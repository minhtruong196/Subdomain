from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subdomain_config import MachineConfig
from subdomain_geometry import selected_segment_radii_m, theta_i
from subdomain_magnetization import magnetization_coefficients


@dataclass(frozen=True)
class UnknownLayout:
    m_count: int
    n_count: int
    slot_count: int

    @property
    def an_count(self) -> int:
        return self.n_count * self.slot_count

    @property
    def off_Am1(self) -> int:
        return 0

    @property
    def off_Cm1(self) -> int:
        return self.m_count

    @property
    def off_Am2(self) -> int:
        return 2 * self.m_count

    @property
    def off_Bm2(self) -> int:
        return 3 * self.m_count

    @property
    def off_Cm2(self) -> int:
        return 4 * self.m_count

    @property
    def off_Dm2(self) -> int:
        return 5 * self.m_count

    @property
    def off_An3(self) -> int:
        return 6 * self.m_count

    @property
    def off_An4(self) -> int:
        return self.off_B03 + self.slot_count

    @property
    def off_B03(self) -> int:
        return self.off_An3 + self.an_count

    @property
    def off_B04(self) -> int:
        return self.off_An4 + self.an_count

    @property
    def total(self) -> int:
        return self.off_B04 + self.slot_count

    def Am1(self, m_idx: int) -> int:
        return self.off_Am1 + m_idx

    def Cm1(self, m_idx: int) -> int:
        return self.off_Cm1 + m_idx

    def Am2(self, m_idx: int) -> int:
        return self.off_Am2 + m_idx

    def Bm2(self, m_idx: int) -> int:
        return self.off_Bm2 + m_idx

    def Cm2(self, m_idx: int) -> int:
        return self.off_Cm2 + m_idx

    def Dm2(self, m_idx: int) -> int:
        return self.off_Dm2 + m_idx

    def An3(self, n_idx: int, slot_idx: int) -> int:
        return self.off_An3 + slot_idx * self.n_count + n_idx

    def An4(self, n_idx: int, slot_idx: int) -> int:
        return self.off_An4 + slot_idx * self.n_count + n_idx

    def B03(self, slot_idx: int) -> int:
        return self.off_B03 + slot_idx

    def B04(self, slot_idx: int) -> int:
        return self.off_B04 + slot_idx


@dataclass(frozen=True)
class SegmentSolutionSet:
    layout: UnknownLayout
    meta: dict[str, np.ndarray | float]
    solutions: np.ndarray


def ratio_p_over_e(order: float, x: float, y: float) -> float:
    """Return P_order(x,y) / E_order(x,y) without overflowing."""

    a = float(order * np.log(x / y))
    if a > 25.0:
        return 1.0
    if a < -25.0:
        return -1.0
    return float(np.cosh(a) / np.sinh(a))


def ratio_e_over_p(order: float, x: float, y: float) -> float:
    return float(np.tanh(order * np.log(x / y)))


def two_over_e(order: float, x: float, y: float) -> float:
    """Return 2 / E_order(x,y), where E = (x/y)^order - (y/x)^order."""

    a = float(order * np.log(x / y))
    if a > 25.0:
        return float(2.0 * np.exp(-a))
    if a < -25.0:
        return float(-2.0 * np.exp(a))
    return float(1.0 / np.sinh(a))


def p_ratio_same_y(order: float, x: float, x_ref: float, y: float) -> float:
    """Return P_order(x,y) / P_order(x_ref,y) using log-sum-exp."""

    a = float(order * np.log(x / y))
    b = float(order * np.log(x_ref / y))
    return float(np.exp(np.logaddexp(a, -a) - np.logaddexp(b, -b)))


def k_mj(r: float, mc: float, Mr_m: float, Mt_m: float, mu0: float) -> float:
    if np.isclose(mc, 1.0):
        return float(-0.5 * mu0 * (Mr_m + Mt_m) * r * np.log(r))
    return float(mu0 * (mc * Mr_m + Mt_m) * r / (mc * mc - 1.0))


def kprime_mj(r: float, mc: float, Mr_m: float, Mt_m: float, mu0: float) -> float:
    if np.isclose(mc, 1.0):
        return float(-0.5 * mu0 * (Mr_m + Mt_m) * (np.log(r) + 1.0))
    return float(mu0 * (mc * Mr_m + Mt_m) / (mc * mc - 1.0))


def x_mj(
    r: float,
    mc: float,
    Mr_m: float,
    Mt_m: float,
    R_l: float,
    R_u: float,
    mu0: float,
) -> float:
    term_l = (R_l / mc) * (R_l / r) ** mc * (kprime_mj(R_l, mc, Mr_m, Mt_m, mu0) + mu0 * Mt_m)
    term_u = (
        (R_l / mc)
        * (R_l / R_u) ** mc
        * (kprime_mj(R_l, mc, Mr_m, Mt_m, mu0) + mu0 * Mt_m)
        + k_mj(R_u, mc, Mr_m, Mt_m, mu0)
    )
    return float(term_l + k_mj(r, mc, Mr_m, Mt_m, mu0) - p_ratio_same_y(mc, r, R_u, R_l) * term_u)


def xprime_mj_at_ru(mc: float, Mr_m: float, Mt_m: float, R_l: float, R_u: float, mu0: float) -> float:
    h = min(1.0e-6, 0.25 * (R_u - R_l))
    return float(
        (x_mj(R_u, mc, Mr_m, Mt_m, R_l, R_u, mu0) - x_mj(R_u - h, mc, Mr_m, Mt_m, R_l, R_u, mu0))
        / h
    )


def G_n(n: int, config: MachineConfig) -> float:
    return float(n * np.pi / config.stator.b_s_rad)


def f_mi(mc: float, slot_number: int, config: MachineConfig) -> float:
    th = theta_i(slot_number, config)
    bs = config.stator.b_s_rad
    return float((np.sin(mc * (th + 0.5 * bs)) - np.sin(mc * (th - 0.5 * bs))) / mc)


def g_mi(mc: float, slot_number: int, config: MachineConfig) -> float:
    th = theta_i(slot_number, config)
    bs = config.stator.b_s_rad
    return float(-(np.cos(mc * (th + 0.5 * bs)) - np.cos(mc * (th - 0.5 * bs))) / mc)


def f_nmi(n: int, mc: float, slot_number: int, config: MachineConfig) -> float:
    th = theta_i(slot_number, config)
    bs = config.stator.b_s_rad
    gn = G_n(n, config)
    if np.isclose(mc, gn):
        return float(0.5 * bs * np.cos(mc * (th - 0.5 * bs)))
    return float(
        mc
        * (np.sin(mc * (th - 0.5 * bs)) - ((-1) ** n) * np.sin(mc * (th + 0.5 * bs)))
        / (gn * gn - mc * mc)
    )


def g_nmi(n: int, mc: float, slot_number: int, config: MachineConfig) -> float:
    th = theta_i(slot_number, config)
    bs = config.stator.b_s_rad
    gn = G_n(n, config)
    if np.isclose(mc, gn):
        return float(0.5 * bs * np.sin(mc * (th - 0.5 * bs)))
    return float(
        mc
        * (((-1) ** n) * np.cos(mc * (th + 0.5 * bs)) - np.cos(mc * (th - 0.5 * bs)))
        / (gn * gn - mc * mc)
    )


def K_c(config: MachineConfig) -> float:
    Rs = config.stator.R_s_m
    Rsc = config.stator.R_sc_m
    Rsa = config.stator.R_sa_m
    Jui = config.current.Jui_A_per_m2
    Jdi = config.current.Jdi_A_per_m2
    mu0 = config.magnet.mu0
    return float(
        -mu0 * Jui * Rs / (2.0 * np.pi)
        + (mu0 * Rsc * Rsc * (Jui - Jdi) / (2.0 * np.pi) + mu0 * Rsa * Rsa * Jdi / (2.0 * np.pi)) / Rs
    )


def slot_current_b04_minus_b03(config: MachineConfig) -> float:
    Rs = config.stator.R_s_m
    Rsc = config.stator.R_sc_m
    Rsa = config.stator.R_sa_m
    Jui = config.current.Jui_A_per_m2
    Jdi = config.current.Jdi_A_per_m2
    mu0 = config.magnet.mu0
    return float(
        mu0 * Jui * (Rs * Rs - Rsc * Rsc) / 4.0
        + (mu0 * Rsc * Rsc * (Jui - Jdi) / 2.0 + mu0 * Rsa * Rsa * Jdi / 2.0) * np.log(Rsc / Rs)
    )


def build_boundary_matrix(
    config: MachineConfig,
    segment_j: int = 1,
    delta_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, UnknownLayout, dict[str, np.ndarray | float]]:
    mc_values, Mr_values, Mt_values = magnetization_coefficients(segment_j, config)

    layout = UnknownLayout(
        m_count=len(mc_values),
        n_count=config.solver.slot_harmonics,
        slot_count=config.reduced_slot_count,
    )
    K = np.zeros((layout.total, layout.total), dtype=float)
    Y = np.zeros(layout.total, dtype=float)

    Ru, Rl = selected_segment_radii_m(config, segment_j)
    Rs = config.stator.R_s_m
    Rsa = config.stator.R_sa_m
    c = config.c_periods
    bs = config.stator.b_s_rad
    kc = K_c(config)
    b04_rhs = slot_current_b04_minus_b03(config)
    mu0 = config.magnet.mu0
    mu_r = config.magnet.mu_r

    row = 0

    for mi, mc in enumerate(mc_values):
        a_coeff = Ru / mc * ratio_p_over_e(mc, Ru, Rs)
        b_coeff = Rs / mc * two_over_e(mc, Rs, Ru)

        K[row, layout.Am1(mi)] = 1.0
        K[row, layout.Am2(mi)] = -a_coeff
        K[row, layout.Bm2(mi)] = -b_coeff
        row += 1

        K[row, layout.Cm1(mi)] = 1.0
        K[row, layout.Cm2(mi)] = -a_coeff
        K[row, layout.Dm2(mi)] = -b_coeff
        row += 1

    for mi, (mc, Mr_m, Mt_m) in enumerate(zip(mc_values, Mr_values, Mt_values)):
        boundary_source = xprime_mj_at_ru(mc, Mr_m, Mt_m, Rl, Ru, mu0) + mu0 * Mt_m
        a_coeff = mc / (mu_r * Ru) * ratio_e_over_p(mc, Ru, Rl)

        K[row, layout.Am2(mi)] = 1.0
        K[row, layout.Am1(mi)] = -a_coeff
        Y[row] = -(boundary_source / mu_r) * np.sin(mc * delta_rad)
        row += 1

        K[row, layout.Cm2(mi)] = 1.0
        K[row, layout.Cm1(mi)] = -a_coeff
        Y[row] = (boundary_source / mu_r) * np.cos(mc * delta_rad)
        row += 1

    for mi, mc in enumerate(mc_values):
        K[row, layout.Bm2(mi)] = 1.0
        Y[row] = sum(2.0 * c * kc * f_mi(mc, slot_i, config) for slot_i in range(1, layout.slot_count + 1))
        for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
            for n_idx, n in enumerate(range(1, layout.n_count + 1)):
                gn = G_n(n, config)
                coeff = gn / Rs * ratio_e_over_p(gn, Rs, Rsa) * (2.0 * c * f_nmi(n, mc, slot_i, config) / np.pi)
                K[row, layout.An3(n_idx, slot_idx)] -= coeff
        row += 1

        K[row, layout.Dm2(mi)] = 1.0
        Y[row] = sum(2.0 * c * kc * g_mi(mc, slot_i, config) for slot_i in range(1, layout.slot_count + 1))
        for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
            for n_idx, n in enumerate(range(1, layout.n_count + 1)):
                gn = G_n(n, config)
                coeff = gn / Rs * ratio_e_over_p(gn, Rs, Rsa) * (2.0 * c * g_nmi(n, mc, slot_i, config) / np.pi)
                K[row, layout.An3(n_idx, slot_idx)] -= coeff
        row += 1

    for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
        for n_idx, n in enumerate(range(1, layout.n_count + 1)):
            for target in (layout.An3(n_idx, slot_idx), layout.An4(n_idx, slot_idx)):
                K[row, target] = 1.0
                for mi, mc in enumerate(mc_values):
                    f_coeff = 2.0 * f_nmi(n, mc, slot_i, config) / bs
                    g_coeff = 2.0 * g_nmi(n, mc, slot_i, config) / bs

                    a_part = Ru / mc * two_over_e(mc, Ru, Rs)
                    b_part = Rs / mc * ratio_p_over_e(mc, Rs, Ru)

                    K[row, layout.Am2(mi)] -= a_part * f_coeff
                    K[row, layout.Bm2(mi)] -= b_part * f_coeff
                    K[row, layout.Cm2(mi)] -= a_part * g_coeff
                    K[row, layout.Dm2(mi)] -= b_part * g_coeff
                row += 1

    for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
        K[row, layout.B03(slot_idx)] = 1.0
        for mi, mc in enumerate(mc_values):
            f_coeff = f_mi(mc, slot_i, config) / bs
            g_coeff = g_mi(mc, slot_i, config) / bs

            a_part = Ru / mc * two_over_e(mc, Ru, Rs)
            b_part = Rs / mc * ratio_p_over_e(mc, Rs, Ru)

            K[row, layout.Am2(mi)] -= a_part * f_coeff
            K[row, layout.Bm2(mi)] -= b_part * f_coeff
            K[row, layout.Cm2(mi)] -= a_part * g_coeff
            K[row, layout.Dm2(mi)] -= b_part * g_coeff
        row += 1

        K[row, layout.B04(slot_idx)] = 1.0
        K[row, layout.B03(slot_idx)] = -1.0
        Y[row] = b04_rhs
        row += 1

    if row != layout.total:
        raise RuntimeError(f"assembled {row} equations for {layout.total} unknowns")

    meta: dict[str, np.ndarray | float] = {
        "segment_j": float(segment_j),
        "mc_values": mc_values,
        "Mr_values": Mr_values,
        "Mt_values": Mt_values,
        "R_u_m": Ru,
        "R_l_m": Rl,
        "R_s_m": Rs,
        "R_sc_m": config.stator.R_sc_m,
        "R_sa_m": Rsa,
        "b_s_rad": bs,
        "K_c": kc,
    }
    return K, Y, layout, meta


def solve_boundary_matrix(
    config: MachineConfig,
    segment_j: int = 1,
    delta_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, UnknownLayout, dict[str, np.ndarray | float], float]:
    K, Y, layout, meta = build_boundary_matrix(config, segment_j=segment_j, delta_rad=delta_rad)
    solution = np.linalg.solve(K, Y)
    residual = np.linalg.norm(K @ solution - Y) / max(1.0, np.linalg.norm(Y))
    return solution, K, Y, layout, meta, float(residual)


def rhs_for_delta(
    base_y: np.ndarray,
    layout: UnknownLayout,
    meta: dict[str, np.ndarray | float],
    config: MachineConfig,
    delta_rad: float,
) -> np.ndarray:
    """Update the Appendix-2 source terms for a rotor angle Delta."""

    y = base_y.copy()
    mc_values = np.asarray(meta["mc_values"], dtype=float)
    Mr_values = np.asarray(meta["Mr_values"], dtype=float)
    Mt_values = np.asarray(meta["Mt_values"], dtype=float)
    Ru = float(meta["R_u_m"])
    Rl = float(meta["R_l_m"])
    mu0 = config.magnet.mu0
    mu_r = config.magnet.mu_r

    first_appendix_2_row = 2 * layout.m_count
    for mi, (mc, Mr_m, Mt_m) in enumerate(zip(mc_values, Mr_values, Mt_values)):
        source = xprime_mj_at_ru(mc, Mr_m, Mt_m, Rl, Ru, mu0) + mu0 * Mt_m
        y[first_appendix_2_row + 2 * mi] = -(source / mu_r) * np.sin(mc * delta_rad)
        y[first_appendix_2_row + 2 * mi + 1] = (source / mu_r) * np.cos(mc * delta_rad)
    return y


def solve_segments_for_deltas(config: MachineConfig, delta_rad: np.ndarray) -> list[SegmentSolutionSet]:
    """Solve K X = Y for all PM segment pairs and all requested rotor angles."""

    solved: list[SegmentSolutionSet] = []
    for segment_j in range(1, config.geometry.Nz + 1):
        K, Y0, layout, meta = build_boundary_matrix(config, segment_j=segment_j, delta_rad=0.0)
        rhs = np.column_stack([rhs_for_delta(Y0, layout, meta, config, float(delta)) for delta in delta_rad])
        solutions = np.linalg.solve(K, rhs)
        solved.append(SegmentSolutionSet(layout=layout, meta=meta, solutions=solutions))
    return solved

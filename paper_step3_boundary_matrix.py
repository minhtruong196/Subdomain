from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper_step1_geometry import PaperGeometryParams, segment_radii_mm
from paper_step2_magnetization import MagnetizationParams, magnetization_coefficients


@dataclass(frozen=True)
class BoundaryMatrixParams:
    """Step 3 parameters for the Appendix boundary-condition matrix."""

    geometry: PaperGeometryParams = PaperGeometryParams()
    segment_j: int = 1
    max_pole_harmonic: int = 200
    slot_harmonics: int = 10
    # Magnetization model: "parallel" follows Eq. (10)-(12), "radial" follows Eq. (15).
    magnetization_model: str = "radial"

    Br_T: float = 1.065
    mu0: float = 4.0 * np.pi * 1.0e-7

    # Table I gives no PM recoil permeability. Keep it configurable and use
    # the neutral checkpoint value first.
    mu_r: float = 1.0

    # Table I value. A sensitivity check with 87.8e-3 gives a near-1 mm gap,
    # but the paper states Rg = 88.4 mm, which implies Rs = 90 mm.
    stator_inner_radius_m: float = 90.0e-3
    slot_depth_m: float = 33.0e-3
    slot_width_m: float = 4.2e-3

    # No-load checkpoint. Nonzero slot current densities can be inserted later.
    Jui_A_per_m2: float = 0.0
    Jdi_A_per_m2: float = 0.0

    delta_rad: float = 0.0
    output_dir: str = "outputs/step3_boundary_matrix"

    @property
    def c_periods(self) -> int:
        return self.geometry.pole_pairs

    @property
    def reduced_slot_count(self) -> int:
        return self.geometry.slots // (2 * self.c_periods)

    @property
    def R_s_m(self) -> float:
        return self.stator_inner_radius_m

    @property
    def R_sa_m(self) -> float:
        return self.stator_inner_radius_m + self.slot_depth_m

    @property
    def R_sc_m(self) -> float:
        # Fig. 6 uses Rsc as the boundary between upper/lower slot halves.
        return self.stator_inner_radius_m + 0.5 * self.slot_depth_m

    @property
    def b_s_rad(self) -> float:
        # Appendix says b_s is expressed in radians.
        return self.slot_width_m / self.stator_inner_radius_m


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
    # One-sided, inside-PM derivative at r = Ru. Central difference would step
    # outside the subdomain for the upper boundary.
    h = min(1.0e-6, 0.25 * (R_u - R_l))
    return float(
        (x_mj(R_u, mc, Mr_m, Mt_m, R_l, R_u, mu0) - x_mj(R_u - h, mc, Mr_m, Mt_m, R_l, R_u, mu0))
        / h
    )


def theta_i(slot_number: int, params: BoundaryMatrixParams) -> float:
    return float(np.pi / params.geometry.slots * (2 * slot_number - 1))


def G_n(n: int, params: BoundaryMatrixParams) -> float:
    return float(n * np.pi / params.b_s_rad)


def f_mi(mc: float, slot_number: int, params: BoundaryMatrixParams) -> float:
    th = theta_i(slot_number, params)
    bs = params.b_s_rad
    return float((np.sin(mc * (th + 0.5 * bs)) - np.sin(mc * (th - 0.5 * bs))) / mc)


def g_mi(mc: float, slot_number: int, params: BoundaryMatrixParams) -> float:
    th = theta_i(slot_number, params)
    bs = params.b_s_rad
    return float(-(np.cos(mc * (th + 0.5 * bs)) - np.cos(mc * (th - 0.5 * bs))) / mc)


def f_nmi(n: int, mc: float, slot_number: int, params: BoundaryMatrixParams) -> float:
    th = theta_i(slot_number, params)
    bs = params.b_s_rad
    gn = G_n(n, params)
    if np.isclose(mc, gn):
        return float(0.5 * bs * np.cos(mc * (th - 0.5 * bs)))
    return float(
        mc
        * (np.sin(mc * (th - 0.5 * bs)) - ((-1) ** n) * np.sin(mc * (th + 0.5 * bs)))
        / (gn * gn - mc * mc)
    )


def g_nmi(n: int, mc: float, slot_number: int, params: BoundaryMatrixParams) -> float:
    th = theta_i(slot_number, params)
    bs = params.b_s_rad
    gn = G_n(n, params)
    if np.isclose(mc, gn):
        return float(0.5 * bs * np.sin(mc * (th - 0.5 * bs)))
    return float(
        mc
        * (((-1) ** n) * np.cos(mc * (th + 0.5 * bs)) - np.cos(mc * (th - 0.5 * bs)))
        / (gn * gn - mc * mc)
    )


def K_c(params: BoundaryMatrixParams) -> float:
    Rs = params.R_s_m
    Rsc = params.R_sc_m
    Rsa = params.R_sa_m
    Jui = params.Jui_A_per_m2
    Jdi = params.Jdi_A_per_m2
    return float(
        -params.mu0 * Jui * Rs / (2.0 * np.pi)
        + (
            params.mu0 * Rsc * Rsc * (Jui - Jdi) / (2.0 * np.pi)
            + params.mu0 * Rsa * Rsa * Jdi / (2.0 * np.pi)
        )
        / Rs
    )


def slot_current_b04_minus_b03(params: BoundaryMatrixParams) -> float:
    Rs = params.R_s_m
    Rsc = params.R_sc_m
    Rsa = params.R_sa_m
    Jui = params.Jui_A_per_m2
    Jdi = params.Jdi_A_per_m2
    return float(
        params.mu0 * Jui * (Rs * Rs - Rsc * Rsc) / 4.0
        + (params.mu0 * Rsc * Rsc * (Jui - Jdi) / 2.0 + params.mu0 * Rsa * Rsa * Jdi / 2.0)
        * np.log(Rsc / Rs)
    )


def selected_segment_radii_m(params: BoundaryMatrixParams) -> tuple[float, float]:
    _, Ru_mm, Rl_mm = segment_radii_mm(params.geometry)
    idx = params.segment_j - 1
    if idx < 0 or idx >= len(Ru_mm):
        raise ValueError(f"segment_j must be in [1, {len(Ru_mm)}]")
    return float(Ru_mm[idx] * 1.0e-3), float(Rl_mm[idx] * 1.0e-3)


def build_boundary_matrix(params: BoundaryMatrixParams) -> tuple[np.ndarray, np.ndarray, UnknownLayout, dict[str, np.ndarray | float]]:
    mag_params = MagnetizationParams(
        geometry=params.geometry,
        Br_T=params.Br_T,
        mu0=params.mu0,
        max_pole_harmonic=params.max_pole_harmonic,
        delta_rad=params.delta_rad,
        magnetization_model=params.magnetization_model,
    )
    mc_values, Mr_values, Mt_values = magnetization_coefficients(params.segment_j, mag_params)

    layout = UnknownLayout(
        m_count=len(mc_values),
        n_count=params.slot_harmonics,
        slot_count=params.reduced_slot_count,
    )
    K = np.zeros((layout.total, layout.total), dtype=float)
    Y = np.zeros(layout.total, dtype=float)

    Ru, Rl = selected_segment_radii_m(params)
    Rs = params.R_s_m
    Rsa = params.R_sa_m
    c = params.c_periods
    bs = params.b_s_rad
    kc = K_c(params)
    b04_rhs = slot_current_b04_minus_b03(params)

    row = 0

    # Appendix 1: equations of Am1 and Cm1.
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

    # Appendix 2: equations of Am2 and Cm2, including the magnetization source.
    for mi, (mc, Mr_m, Mt_m) in enumerate(zip(mc_values, Mr_values, Mt_values)):
        boundary_source = xprime_mj_at_ru(mc, Mr_m, Mt_m, Rl, Ru, params.mu0)
        a_coeff = mc / (params.mu_r * Ru) * ratio_e_over_p(mc, Ru, Rl)

        K[row, layout.Am2(mi)] = 1.0
        K[row, layout.Am1(mi)] = -a_coeff
        Y[row] = -(boundary_source / params.mu_r) * np.sin(mc * params.delta_rad)
        row += 1

        K[row, layout.Cm2(mi)] = 1.0
        K[row, layout.Cm1(mi)] = -a_coeff
        Y[row] = (boundary_source / params.mu_r) * np.cos(mc * params.delta_rad)
        row += 1

    # Appendix 3: odd periodic equations of Bm2 and Dm2.
    for mi, mc in enumerate(mc_values):
        K[row, layout.Bm2(mi)] = 1.0
        Y[row] = sum(2.0 * c * kc * f_mi(mc, slot_i, params) for slot_i in range(1, layout.slot_count + 1))
        for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
            for n_idx, n in enumerate(range(1, layout.n_count + 1)):
                gn = G_n(n, params)
                coeff = gn / Rs * ratio_e_over_p(gn, Rs, Rsa) * (2.0 * c * f_nmi(n, mc, slot_i, params) / np.pi)
                K[row, layout.An3(n_idx, slot_idx)] -= coeff
        row += 1

        K[row, layout.Dm2(mi)] = 1.0
        Y[row] = sum(2.0 * c * kc * g_mi(mc, slot_i, params) for slot_i in range(1, layout.slot_count + 1))
        for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
            for n_idx, n in enumerate(range(1, layout.n_count + 1)):
                gn = G_n(n, params)
                coeff = gn / Rs * ratio_e_over_p(gn, Rs, Rsa) * (2.0 * c * g_nmi(n, mc, slot_i, params) / np.pi)
                K[row, layout.An3(n_idx, slot_idx)] -= coeff
        row += 1

    # Appendix 4: equations of An3i and An4i.
    for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
        for n_idx, n in enumerate(range(1, layout.n_count + 1)):
            for target in (layout.An3(n_idx, slot_idx), layout.An4(n_idx, slot_idx)):
                K[row, target] = 1.0
                for mi, mc in enumerate(mc_values):
                    f_coeff = 2.0 * f_nmi(n, mc, slot_i, params) / bs
                    g_coeff = 2.0 * g_nmi(n, mc, slot_i, params) / bs

                    a_part = Ru / mc * two_over_e(mc, Ru, Rs)
                    b_part = Rs / mc * ratio_p_over_e(mc, Rs, Ru)

                    K[row, layout.Am2(mi)] -= a_part * f_coeff
                    K[row, layout.Bm2(mi)] -= b_part * f_coeff
                    K[row, layout.Cm2(mi)] -= a_part * g_coeff
                    K[row, layout.Dm2(mi)] -= b_part * g_coeff
                row += 1

    # Appendix 5: equations of B03i and B04i.
    for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
        K[row, layout.B03(slot_idx)] = 1.0
        for mi, mc in enumerate(mc_values):
            f_coeff = f_mi(mc, slot_i, params) / bs
            g_coeff = g_mi(mc, slot_i, params) / bs

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
        "mc_values": mc_values,
        "Mr_values": Mr_values,
        "Mt_values": Mt_values,
        "R_u_m": Ru,
        "R_l_m": Rl,
        "R_s_m": Rs,
        "R_sc_m": params.R_sc_m,
        "R_sa_m": Rsa,
        "b_s_rad": bs,
        "K_c": kc,
    }
    return K, Y, layout, meta


def solve_boundary_matrix(params: BoundaryMatrixParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, UnknownLayout, dict[str, np.ndarray | float], float]:
    K, Y, layout, meta = build_boundary_matrix(params)
    solution = np.linalg.solve(K, Y)
    residual = np.linalg.norm(K @ solution - Y) / max(1.0, np.linalg.norm(Y))
    return solution, K, Y, layout, meta, float(residual)


def save_solution(
    solution: np.ndarray,
    K: np.ndarray,
    Y: np.ndarray,
    layout: UnknownLayout,
    meta: dict[str, np.ndarray | float],
    residual: float,
    params: BoundaryMatrixParams,
) -> Path:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"step3_solution_j{params.segment_j}_m{params.max_pole_harmonic}_n{params.slot_harmonics}.npz"
    np.savez_compressed(
        path,
        solution=solution,
        K=K,
        Y=Y,
        residual=residual,
        total_unknowns=layout.total,
        m_count=layout.m_count,
        n_count=layout.n_count,
        slot_count=layout.slot_count,
        **meta,
    )
    return path


def main() -> None:
    params = BoundaryMatrixParams()
    solution, K, Y, layout, meta, residual = solve_boundary_matrix(params)
    output_path = save_solution(solution, K, Y, layout, meta, residual, params)

    print("Step 3 boundary-condition matrix")
    print(f"segment j                 : {params.segment_j}")
    print(f"odd periodic c            : {params.c_periods}")
    print(f"reduced slot count        : {layout.slot_count}")
    print(f"pole harmonics kept       : {layout.m_count} (nu <= {params.max_pole_harmonic}, odd only)")
    print(f"magnetization model       : {params.magnetization_model}")
    print(f"slot harmonics n          : {layout.n_count}")
    print(f"matrix shape              : {K.shape}")
    print(f"rhs norm                  : {np.linalg.norm(Y):.6e}")
    print(f"relative residual         : {residual:.6e}")
    print(f"Ru, Rl [m]                : {meta['R_u_m']:.9f}, {meta['R_l_m']:.9f}")
    print(f"Rs, Rsc, Rsa [m]          : {meta['R_s_m']:.9f}, {meta['R_sc_m']:.9f}, {meta['R_sa_m']:.9f}")
    print(f"bs [rad]                  : {meta['b_s_rad']:.9e}")
    print(f"mu_r used                 : {params.mu_r:.6g}")
    print(f"Jui, Jdi [A/m^2]          : {params.Jui_A_per_m2:.6e}, {params.Jdi_A_per_m2:.6e}")
    print()
    print("first solved coefficients")
    print(f"Am1[0]                    : {solution[layout.Am1(0)]: .6e}")
    print(f"Cm1[0]                    : {solution[layout.Cm1(0)]: .6e}")
    print(f"Am2[0]                    : {solution[layout.Am2(0)]: .6e}")
    print(f"Bm2[0]                    : {solution[layout.Bm2(0)]: .6e}")
    print(f"Cm2[0]                    : {solution[layout.Cm2(0)]: .6e}")
    print(f"Dm2[0]                    : {solution[layout.Dm2(0)]: .6e}")
    print()
    print(f"saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()

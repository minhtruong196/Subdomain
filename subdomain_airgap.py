from __future__ import annotations

import numpy as np

from subdomain_boundary import SegmentSolutionSet, UnknownLayout, solve_boundary_matrix
from subdomain_config import MachineConfig


def log_abs_e(order: float, x: float, y: float) -> tuple[float, float]:
    """Return sign and log(abs(E_order(x,y))) without overflowing."""

    a = float(order * np.log(x / y))
    if np.isclose(a, 0.0):
        return 0.0, -np.inf
    sign = 1.0 if a > 0.0 else -1.0
    abs_a = abs(a)
    if abs_a > 25.0:
        return sign, abs_a
    return sign, float(np.log(abs(2.0 * np.sinh(a))))


def p_over_e(order: float, p_x: float, p_y: float, e_x: float, e_y: float) -> float:
    """Return P_order(p_x,p_y) / E_order(e_x,e_y)."""

    a = float(order * np.log(p_x / p_y))
    log_p = float(np.logaddexp(a, -a))
    sign_e, log_abs_den = log_abs_e(order, e_x, e_y)
    return float(sign_e * np.exp(log_p - log_abs_den))


def e_over_e(order: float, n_x: float, n_y: float, d_x: float, d_y: float) -> float:
    """Return E_order(n_x,n_y) / E_order(d_x,d_y)."""

    sign_num, log_abs_num = log_abs_e(order, n_x, n_y)
    if sign_num == 0.0:
        return 0.0
    sign_den, log_abs_den = log_abs_e(order, d_x, d_y)
    return float(sign_num * sign_den * np.exp(log_abs_num - log_abs_den))


def az2_radial_terms(
    r: float,
    Ru: float,
    Rs: float,
    mc_values: np.ndarray,
    Am2: np.ndarray,
    Bm2: np.ndarray,
    Cm2: np.ndarray,
    Dm2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return alpha, beta and their r-derivatives for Eq. (21)."""

    alpha = np.empty_like(mc_values, dtype=float)
    beta = np.empty_like(mc_values, dtype=float)
    alpha_prime = np.empty_like(mc_values, dtype=float)
    beta_prime = np.empty_like(mc_values, dtype=float)

    for idx, mc in enumerate(mc_values):
        pm_rs = p_over_e(mc, r, Rs, Ru, Rs)
        pm_ru = p_over_e(mc, r, Ru, Rs, Ru)
        em_rs = e_over_e(mc, r, Rs, Ru, Rs)
        em_ru = e_over_e(mc, r, Ru, Rs, Ru)

        a_part = Ru / mc * pm_rs
        b_part = Rs / mc * pm_ru
        a_prime_part = Ru / r * em_rs
        b_prime_part = Rs / r * em_ru

        alpha[idx] = Am2[idx] * a_part + Bm2[idx] * b_part
        beta[idx] = Cm2[idx] * a_part + Dm2[idx] * b_part
        alpha_prime[idx] = Am2[idx] * a_prime_part + Bm2[idx] * b_prime_part
        beta_prime[idx] = Cm2[idx] * a_prime_part + Dm2[idx] * b_prime_part

    return alpha, beta, alpha_prime, beta_prime


def az2_airgap(
    r: float,
    theta_mech_rad: np.ndarray,
    solution: np.ndarray,
    layout: UnknownLayout,
    mc_values: np.ndarray,
    Ru: float,
    Rs: float,
) -> np.ndarray:
    """Magnetic vector potential in Subdomain II from Eq. (21)."""

    Am2 = solution[layout.off_Am2 : layout.off_Am2 + layout.m_count]
    Bm2 = solution[layout.off_Bm2 : layout.off_Bm2 + layout.m_count]
    Cm2 = solution[layout.off_Cm2 : layout.off_Cm2 + layout.m_count]
    Dm2 = solution[layout.off_Dm2 : layout.off_Dm2 + layout.m_count]
    alpha, beta, _, _ = az2_radial_terms(r, Ru, Rs, mc_values, Am2, Bm2, Cm2, Dm2)

    phase = np.outer(mc_values, theta_mech_rad)
    return np.sum(alpha[:, None] * np.cos(phase) + beta[:, None] * np.sin(phase), axis=0)


def segment_flux_density(
    r: float,
    theta_mech_rad: np.ndarray,
    solution: np.ndarray,
    layout: UnknownLayout,
    mc_values: np.ndarray,
    Ru: float,
    Rs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Air-gap Br(j), Btheta(j) from Eq. (28)."""

    Am2 = solution[layout.off_Am2 : layout.off_Am2 + layout.m_count]
    Bm2 = solution[layout.off_Bm2 : layout.off_Bm2 + layout.m_count]
    Cm2 = solution[layout.off_Cm2 : layout.off_Cm2 + layout.m_count]
    Dm2 = solution[layout.off_Dm2 : layout.off_Dm2 + layout.m_count]
    alpha, beta, alpha_prime, beta_prime = az2_radial_terms(r, Ru, Rs, mc_values, Am2, Bm2, Cm2, Dm2)

    phase = np.outer(mc_values, theta_mech_rad)
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)

    Br = np.sum((-mc_values * alpha / r)[:, None] * sin_phase + (mc_values * beta / r)[:, None] * cos_phase, axis=0)
    Btheta = -np.sum(alpha_prime[:, None] * cos_phase + beta_prime[:, None] * sin_phase, axis=0)
    return Br, Btheta


def total_no_load_flux_density(config: MachineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Superpose all Nz segment-pair fields according to Eq. (29)."""

    p = config.geometry.pole_pairs
    theta_elec_deg = np.linspace(-90.0, 90.0, config.solver.airgap_sample_count)
    theta_mech_rad = np.deg2rad(theta_elec_deg) / p

    Br_total = np.zeros_like(theta_mech_rad)
    Btheta_total = np.zeros_like(theta_mech_rad)
    residuals: list[float] = []

    for segment_j in range(1, config.geometry.Nz + 1):
        solution, _, _, layout, meta, residual = solve_boundary_matrix(config, segment_j=segment_j)
        residuals.append(residual)
        mc_values = np.asarray(meta["mc_values"], dtype=float)

        Br_j, Btheta_j = segment_flux_density(
            config.airgap_radius_m,
            theta_mech_rad,
            solution,
            layout,
            mc_values,
            float(meta["R_u_m"]),
            config.stator.R_s_m,
        )
        Br_total += Br_j
        Btheta_total += Btheta_j

    return theta_elec_deg, Br_total, Btheta_total, residuals


def full_airgap_flux_density(
    solved_segments: list[SegmentSolutionSet],
    delta_index: int,
    theta_mech_rad: np.ndarray,
    config: MachineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    Br_total = np.zeros_like(theta_mech_rad)
    Btheta_total = np.zeros_like(theta_mech_rad)

    for segment in solved_segments:
        solution = segment.solutions[:, delta_index]
        mc_values = np.asarray(segment.meta["mc_values"], dtype=float)
        Br_j, Btheta_j = segment_flux_density(
            config.airgap_radius_m,
            theta_mech_rad,
            solution,
            segment.layout,
            mc_values,
            float(segment.meta["R_u_m"]),
            float(segment.meta["R_s_m"]),
        )
        Br_total += Br_j
        Btheta_total += Btheta_j

    return Br_total, Btheta_total

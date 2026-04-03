from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import (
    BaselineSolution,
    CurrentLoadingSpectrum,
    HarmonicSpectrum,
    LinearPMModelParams,
    PMOnly5RegionSolution,
)
from excitation import (
    balanced_phase_currents,
    complex_fourier_coefficients,
    dq_currents,
    exact_radial_magnetization,
    mechanical_angle_grid,
    pm_cfs_coefficients,
    radial_pm_harmonics,
    reconstruct_radial_magnetization,
    slot_current_density_waveform,
    slot_current_loading_harmonics,
    slot_current_loading_waveform,
)
from permeability import k_theta_matrix, signed_nonzero_harmonic_orders
from post_processing import (
    average_torque_maxwell,
    equivalent_pm_flux_linkage,
    phase_back_emf_constants,
)
from subdomain_matrix import (
    assemble_linear_5region_system,
    assemble_coupled_pm_only_system,
    build_linear_subdomain_scaffold,
    build_pm_only_coupled_region_operators,
    evaluate_coupled_region_solution,
    particular_coefficients,
    region_absolute_mu,
    region_bounds,
)


def equivalent_air_gap_field(
    theta_mech: np.ndarray,
    radius: float,
    params: LinearPMModelParams,
    spectrum: HarmonicSpectrum,
    odd_harmonic_limit: int | None = None,
    rotor_electrical_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not (params.air_gap_inner_radius <= radius <= params.air_gap_outer_radius):
        raise ValueError("radius must lie inside the smooth air-gap interval")

    if odd_harmonic_limit is None:
        nu = spectrum.electrical_orders
        amp = spectrum.amplitudes
    else:
        keep = spectrum.electrical_orders <= odd_harmonic_limit
        nu = spectrum.electrical_orders[keep]
        amp = spectrum.amplitudes[keep]

    k = nu * params.pole_pairs
    rho = radius / params.air_gap_inner_radius
    gamma = params.air_gap_outer_radius / params.air_gap_inner_radius
    gamma_2k = gamma ** (2 * k)

    radial_transfer = (rho ** (k - 1) + gamma_2k * rho ** (-k - 1)) / (1.0 + gamma_2k)
    tangential_transfer = (
        gamma_2k * rho ** (-k - 1) - rho ** (k - 1)
    ) / (1.0 + gamma_2k)

    boundary_flux_amp = params.mu_0 * amp
    phase = np.outer(k, theta_mech) - np.outer(
        nu, np.full_like(theta_mech, rotor_electrical_angle)
    )

    Br = np.sum((boundary_flux_amp * radial_transfer)[:, None] * np.cos(phase), axis=0)
    Btheta = np.sum(
        (boundary_flux_amp * tangential_transfer)[:, None] * np.sin(phase),
        axis=0,
    )
    return Br, Btheta


def radial_flux_density_from_potential(
    Az_coeff: np.ndarray, radius: float, harmonic_orders: np.ndarray
) -> np.ndarray:
    K_theta = k_theta_matrix(harmonic_orders)
    return (-1j / radius) * (K_theta @ Az_coeff)


def tangential_flux_density_from_potential(dAz_dr_coeff: np.ndarray) -> np.ndarray:
    return -dAz_dr_coeff


def field_strength_from_potential(
    Az_coeff: np.ndarray,
    dAz_dr_coeff: np.ndarray,
    radius: float,
    harmonic_orders: np.ndarray,
    mu_c_r_inv: np.ndarray,
    mu_c_theta_inv: np.ndarray,
    M_r_coeff: np.ndarray,
    M_theta_coeff: np.ndarray,
    params: LinearPMModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper eqs. (35)-(36)."""

    K_theta = k_theta_matrix(harmonic_orders)
    H_r = (-1j / radius) * (mu_c_r_inv @ K_theta @ Az_coeff) - params.mu_0 * (
        mu_c_r_inv @ M_r_coeff
    )
    H_theta = -(mu_c_theta_inv @ dAz_dr_coeff) - params.mu_0 * (
        mu_c_theta_inv @ M_theta_coeff
    )
    return H_r, H_theta


def evaluate_region_potential_coefficients(
    order: int,
    radius: float,
    region_index: int,
    coefficients: np.ndarray,
    params: LinearPMModelParams,
    M_r: complex = 0.0j,
    M_theta: complex = 0.0j,
    Jz_iv: complex = 0.0j,
    Jz_v: complex = 0.0j,
) -> tuple[complex, complex]:
    bounds = region_bounds(params)
    G_I, F_IV, F_V = particular_coefficients(order, params, M_r, M_theta, Jz_iv, Jz_v)

    a = coefficients[2 * region_index]
    b = coefficients[2 * region_index + 1]
    inner_radius, outer_radius = bounds[region_index]
    phi_a = (radius / outer_radius) ** order
    phi_b = (radius / inner_radius) ** (-order)
    Az = a * phi_a + b * phi_b
    dAz_dr = (order / radius) * a * phi_a - (order / radius) * b * phi_b

    if region_index == 0:
        Az += radius * G_I
        dAz_dr += G_I
    elif region_index == 3:
        Az += radius**2 * F_IV
        dAz_dr += 2.0 * radius * F_IV
    elif region_index == 4:
        Az += radius**2 * F_V
        dAz_dr += 2.0 * radius * F_V

    return Az, dAz_dr


def solve_linear_5region_preview(
    params: LinearPMModelParams,
    radius: float | None = None,
    include_current: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assumption-driven linear 5-region preview.

    This uses:
    - constant scalar permeability in each region
    - PM CFS from eqs. (24)-(29)
    - PM source terms are reliable enough for preview
    - current source terms are optional and remain experimental until the
      mixed slot/tooth permeability of regions IV/V is modeled properly
    - BC assembly from eqs. (41)-(42)
    """

    theta = mechanical_angle_grid(params)
    eval_radius = params.mid_gap_radius if radius is None else radius
    bounds = region_bounds(params)
    mu = region_absolute_mu(params)
    harmonic_orders = np.arange(1, params.max_current_harmonic + 1, dtype=int)

    _, M_r_coeffs, M_theta_coeffs = pm_cfs_coefficients(params, harmonic_orders)
    if include_current:
        Jz_waveform, _, _ = slot_current_density_waveform(theta, params)
        Jz_coeffs = complex_fourier_coefficients(theta, Jz_waveform, harmonic_orders)
    else:
        Jz_coeffs = np.zeros(len(harmonic_orders), dtype=complex)

    Br = np.zeros_like(theta, dtype=float)
    Btheta = np.zeros_like(theta, dtype=float)

    for idx, order in enumerate(harmonic_orders):
        matrix, rhs = assemble_linear_5region_system(
            order=int(order),
            params=params,
            M_r=M_r_coeffs[idx],
            M_theta=M_theta_coeffs[idx],
            Jz_iv=Jz_coeffs[idx],
            Jz_v=Jz_coeffs[idx],
        )
        try:
            coeffs = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue

        Az_n, dAz_dr_n = evaluate_region_potential_coefficients(
            order=int(order),
            radius=eval_radius,
            region_index=1,
            coefficients=coeffs,
            params=params,
            M_r=M_r_coeffs[idx],
            M_theta=M_theta_coeffs[idx],
            Jz_iv=Jz_coeffs[idx],
            Jz_v=Jz_coeffs[idx],
        )
        Br_coeff = -1j * order * Az_n / eval_radius
        Btheta_coeff = -dAz_dr_n
        Br += 2.0 * np.real(Br_coeff * np.exp(1j * order * theta))
        Btheta += 2.0 * np.real(Btheta_coeff * np.exp(1j * order * theta))

    return theta, Br, Btheta


def cogging_period_mech(params: LinearPMModelParams) -> float:
    return 2.0 * np.pi / np.lcm(params.stator_slots, params.pole_count)


def solve_linear_5region_pm_only(
    params: LinearPMModelParams,
    radius: float | None = None,
    cogging_point_count: int | None = None,
) -> PMOnly5RegionSolution:
    eval_radius = params.mid_gap_radius if radius is None else radius
    harmonic_orders = signed_nonzero_harmonic_orders(params.pm_only_coupled_max_harmonic)
    harmonic_count = len(harmonic_orders)
    cogging_point_count = (
        params.pm_only_cogging_point_count
        if cogging_point_count is None
        else cogging_point_count
    )

    def solve_single(rotated_params: LinearPMModelParams) -> tuple[np.ndarray, np.ndarray, float]:
        theta = mechanical_angle_grid(rotated_params)
        operators = build_pm_only_coupled_region_operators(
            theta_mech=theta,
            params=rotated_params,
            harmonic_orders=harmonic_orders,
        )
        magnetization = exact_radial_magnetization(theta, rotated_params)
        M_r = complex_fourier_coefficients(theta, magnetization, harmonic_orders)
        M_theta = np.zeros_like(M_r)
        matrix, rhs = assemble_coupled_pm_only_system(
            params=rotated_params,
            operators=operators,
            harmonic_orders=harmonic_orders,
            M_r=M_r,
            M_theta=M_theta,
        )
        try:
            solution_vector = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solution_vector, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residual = np.linalg.norm(matrix @ solution_vector - rhs) / (
            1.0 + np.linalg.norm(rhs)
        )

        region_ii_a = solution_vector[2 * harmonic_count : 3 * harmonic_count]
        region_ii_b = solution_vector[3 * harmonic_count : 4 * harmonic_count]
        Az_coeff, dAz_dr_coeff = evaluate_coupled_region_solution(
            operator=operators["II"],
            radius=eval_radius,
            a_coeff=region_ii_a,
            b_coeff=region_ii_b,
            M_r=M_r,
            M_theta=M_theta,
            mu_0=rotated_params.mu_0,
        )
        phase = np.exp(1j * np.outer(harmonic_orders, theta))
        Br_coeff = (-1j / eval_radius) * (harmonic_orders.astype(complex) * Az_coeff)
        Btheta_coeff = -dAz_dr_coeff
        Br_gap = np.real(np.sum(Br_coeff[:, None] * phase, axis=0))
        Btheta_gap = np.real(np.sum(Btheta_coeff[:, None] * phase, axis=0))
        return Br_gap, Btheta_gap, float(residual)

    theta = mechanical_angle_grid(params)
    Br_gap, Btheta_gap, solve_residual = solve_single(params)

    cogging_period = cogging_period_mech(params)
    rotor_angles_mech = np.linspace(0.0, cogging_period, cogging_point_count)
    cogging_torque_nm = np.empty(cogging_point_count, dtype=float)

    for idx, rotor_angle in enumerate(rotor_angles_mech):
        rotated_params = replace(params, initial_rotor_angle_mech=float(rotor_angle))
        Br_rot, Btheta_rot, _ = solve_single(rotated_params)
        cogging_torque_nm[idx] = average_torque_maxwell(
            theta_mech=theta,
            Br=Br_rot,
            Btheta=Btheta_rot,
            radius=eval_radius,
            params=rotated_params,
        )

    return PMOnly5RegionSolution(
        params=params,
        theta_mech=theta,
        harmonic_orders=harmonic_orders,
        eval_radius=eval_radius,
        Br_gap=Br_gap,
        Btheta_gap=Btheta_gap,
        rotor_angles_mech=rotor_angles_mech,
        cogging_torque_nm=cogging_torque_nm,
        system_size=10 * harmonic_count,
        solve_residual=solve_residual,
    )


def slot_current_air_gap_field(
    theta_mech: np.ndarray,
    radius: float,
    params: LinearPMModelParams,
    current_spectrum: CurrentLoadingSpectrum,
) -> tuple[np.ndarray, np.ndarray]:
    if not (params.air_gap_inner_radius <= radius <= params.air_gap_outer_radius):
        raise ValueError("radius must lie inside the smooth air-gap interval")

    orders = current_spectrum.orders
    a = current_spectrum.cosine
    b = current_spectrum.sine

    R2 = params.air_gap_inner_radius
    R3 = params.air_gap_outer_radius
    denom = R3 ** (2 * orders) - R2 ** (2 * orders)
    prefactor = params.mu_0 * R3 ** (orders + 1) / denom

    radial_transfer = radius ** (orders - 1) + R2 ** (2 * orders) * radius ** (-orders - 1)
    tangential_transfer = radius ** (orders - 1) - R2 ** (2 * orders) * radius ** (-orders - 1)
    phase = np.outer(orders, theta_mech)

    Ktheta = a[:, None] * np.cos(phase) + b[:, None] * np.sin(phase)
    quadrature = a[:, None] * np.sin(phase) - b[:, None] * np.cos(phase)

    Br = np.sum((prefactor * radial_transfer)[:, None] * quadrature, axis=0)
    Btheta = np.sum((prefactor * tangential_transfer)[:, None] * Ktheta, axis=0)
    return Br, Btheta


def total_air_gap_field(
    theta_mech: np.ndarray,
    radius: float,
    params: LinearPMModelParams,
    pm_spectrum: HarmonicSpectrum,
    current_spectrum: CurrentLoadingSpectrum,
    odd_harmonic_limit: int | None = None,
    rotor_electrical_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    Br_pm, Btheta_pm = equivalent_air_gap_field(
        theta_mech=theta_mech,
        radius=radius,
        params=params,
        spectrum=pm_spectrum,
        odd_harmonic_limit=odd_harmonic_limit,
        rotor_electrical_angle=rotor_electrical_angle,
    )
    Br_cs, Btheta_cs = slot_current_air_gap_field(
        theta_mech=theta_mech,
        radius=radius,
        params=params,
        current_spectrum=current_spectrum,
    )
    return Br_pm + Br_cs, Btheta_pm + Btheta_cs


def solve_smooth_air_gap_baseline(params: LinearPMModelParams) -> BaselineSolution:
    theta = mechanical_angle_grid(params)
    pm_spectrum = radial_pm_harmonics(params)
    exact_Mr = exact_radial_magnetization(theta, params)
    recon_Mr = reconstruct_radial_magnetization(theta, params, pm_spectrum)
    phase_currents = balanced_phase_currents(params)
    current_loading, J1_slots, J2_slots = slot_current_loading_waveform(theta, params)
    current_spectrum = slot_current_loading_harmonics(theta, current_loading, params)

    Br_pm_gap, Btheta_pm_gap = equivalent_air_gap_field(
        theta_mech=theta,
        radius=params.mid_gap_radius,
        params=params,
        spectrum=pm_spectrum,
    )
    Br_current_gap, Btheta_current_gap = slot_current_air_gap_field(
        theta_mech=theta,
        radius=params.mid_gap_radius,
        params=params,
        current_spectrum=current_spectrum,
    )
    Br_total_gap, Btheta_total_gap = total_air_gap_field(
        theta_mech=theta,
        radius=params.mid_gap_radius,
        params=params,
        pm_spectrum=pm_spectrum,
        current_spectrum=current_spectrum,
    )

    torque_pm = average_torque_maxwell(
        theta_mech=theta,
        Br=Br_pm_gap,
        Btheta=Btheta_pm_gap,
        radius=params.mid_gap_radius,
        params=params,
    )
    torque_current = average_torque_maxwell(
        theta_mech=theta,
        Br=Br_current_gap,
        Btheta=Btheta_current_gap,
        radius=params.mid_gap_radius,
        params=params,
    )
    torque_total = average_torque_maxwell(
        theta_mech=theta,
        Br=Br_total_gap,
        Btheta=Btheta_total_gap,
        radius=params.mid_gap_radius,
        params=params,
    )
    torque_interaction = torque_total - torque_pm - torque_current

    id_current, iq_current = dq_currents(params)
    lambda_pm_equiv = equivalent_pm_flux_linkage(torque_interaction, params)
    emf_phase_peak, emf_phase_rms = phase_back_emf_constants(lambda_pm_equiv, params)
    mech_power = torque_total * params.omega_mech_rad_s

    return BaselineSolution(
        params=params,
        theta_mech=theta,
        pm_spectrum=pm_spectrum,
        current_spectrum=current_spectrum,
        exact_Mr=exact_Mr,
        recon_Mr=recon_Mr,
        phase_currents=phase_currents,
        J1_slots=J1_slots,
        J2_slots=J2_slots,
        current_loading=current_loading,
        Br_pm_gap=Br_pm_gap,
        Btheta_pm_gap=Btheta_pm_gap,
        Br_current_gap=Br_current_gap,
        Btheta_current_gap=Btheta_current_gap,
        Br_total_gap=Br_total_gap,
        Btheta_total_gap=Btheta_total_gap,
        torque_pm=torque_pm,
        torque_current=torque_current,
        torque_total=torque_total,
        torque_interaction=torque_interaction,
        id_current=id_current,
        iq_current=iq_current,
        lambda_pm_equiv=lambda_pm_equiv,
        emf_phase_peak=emf_phase_peak,
        emf_phase_rms=emf_phase_rms,
        mech_power=mech_power,
    )


def prepare_linear_subdomain_scaffold(params: LinearPMModelParams):
    return build_linear_subdomain_scaffold(params)

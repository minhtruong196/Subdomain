from __future__ import annotations

import numpy as np

from config import CoupledRegionOperators, LinearPMModelParams, SubdomainScaffold
from permeability import (
    convolution_matrices_from_waveforms,
    k_theta_matrix,
    sample_constant_permeability,
    sample_region_i_permeability,
    sample_region_iii_permeability,
    sample_region_iv_v_permeability,
)


def build_linear_subdomain_scaffold(params: LinearPMModelParams) -> SubdomainScaffold:
    return SubdomainScaffold(
        region_names=("I", "II", "III", "IV", "V"),
        interface_radii=(
            params.rotor_outer_radius,
            params.air_gap_inner_radius,
            params.air_gap_outer_radius,
            params.slot_opening_radius,
            params.region_iv_boundary_radius,
            params.stator_outer_radius,
        ),
        unknowns_per_harmonic=10,
        boundary_condition_count=10,
        missing_equations=(
            "Current-carrying 5-region solve still needs the mixed slot/tooth permeability treatment in regions IV/V",
            "Full paper-accurate matrix form still needs the coupled convolution-matrix implementation instead of the current scalar-mu simplification",
        ),
        notes=(
            "This scaffold maps the 5 regions and interface radii without pretending the full paper matrix is done",
            "The verified executable path remains the smooth-air-gap baseline solver",
            "PM CFS source terms from equations (24)-(29) are now available in excitation.py",
            "Permeability CFS helpers from equations (11)-(15) and H(Az) relations from (35)-(36) are now available",
            "Region III/IV/V geometry is now parameterized, but still uses explicit assumptions until exact slot-opening angles and subsection profiles are confirmed",
            "A linear scalar-mu 5-region PM-only solve is now executable for field and cogging-torque preview",
            "Current-carrying 5-region solve is still deferred because regions IV/V need mixed slot-tooth permeability, not a single scalar mu",
            "When the missing blocks are filled, this file becomes the matrix core of the paper model",
        ),
    )


def region_bounds(params: LinearPMModelParams) -> tuple[tuple[float, float], ...]:
    return (
        (params.rotor_outer_radius, params.air_gap_inner_radius),   # I
        (params.air_gap_inner_radius, params.air_gap_outer_radius), # II
        (params.air_gap_outer_radius, params.slot_opening_radius),  # III
        (params.slot_opening_radius, params.region_iv_boundary_radius), # IV
        (params.region_iv_boundary_radius, params.stator_outer_radius), # V
    )


def region_absolute_mu(params: LinearPMModelParams) -> tuple[float, ...]:
    return (
        params.mu_0 * params.magnet_mu_r_linear,
        params.mu_0,
        params.mu_0,
        params.mu_0 * params.core_mu_r_linear,
        params.mu_0 * params.core_mu_r_linear,
    )


def particular_coefficients(
    order: int,
    params: LinearPMModelParams,
    M_r: complex,
    M_theta: complex,
    Jz_iv: complex,
    Jz_v: complex,
) -> tuple[complex, complex, complex]:
    """Return G_I, F_IV, F_V for the linear scalar-mu case."""

    if order in (0, 1):
        G_I = 0.0j
    else:
        G_I = 1j * params.mu_0 * M_r / (order * (order**2 - 1)) + params.mu_0 * M_theta / (
            order**2 - 1
        )

    mu_iv = params.mu_0 * params.core_mu_r_linear
    if order in (0, 2):
        F_IV = 0.0j
        F_V = 0.0j
    else:
        F_IV = Jz_iv / (mu_iv * (order**2 - 4))
        F_V = Jz_v / (mu_iv * (order**2 - 4))

    return G_I, F_IV, F_V


def _basis_values(order: int, radius: float, inner_radius: float, outer_radius: float) -> tuple[complex, complex]:
    return (radius / outer_radius) ** order, (radius / inner_radius) ** (-order)


def _basis_derivatives(
    order: int, radius: float, inner_radius: float, outer_radius: float
) -> tuple[complex, complex]:
    phi_a, phi_b = _basis_values(order, radius, inner_radius, outer_radius)
    return (order / radius) * phi_a, (-order / radius) * phi_b


def assemble_linear_5region_system(
    order: int,
    params: LinearPMModelParams,
    M_r: complex = 0.0j,
    M_theta: complex = 0.0j,
    Jz_iv: complex = 0.0j,
    Jz_v: complex = 0.0j,
) -> tuple[np.ndarray, np.ndarray]:
    if order <= 0:
        raise ValueError("order must be positive for the linear 5-region harmonic solve")

    bounds = region_bounds(params)
    mu = region_absolute_mu(params)
    G_I, F_IV, F_V = particular_coefficients(order, params, M_r, M_theta, Jz_iv, Jz_v)

    matrix = np.zeros((10, 10), dtype=complex)
    rhs = np.zeros(10, dtype=complex)

    # Unknown ordering: [a1,b1,a2,b2,a3,b3,a4,b4,a5,b5]
    def col(region_index: int, coeff: str) -> int:
        return 2 * region_index + (0 if coeff == "a" else 1)

    # Particular Az and dAz/dr terms
    def particular_A(region_index: int, radius: float) -> complex:
        if region_index == 0:
            return radius * G_I
        if region_index == 3:
            return radius**2 * F_IV
        if region_index == 4:
            return radius**2 * F_V
        return 0.0j

    def particular_dA(region_index: int, radius: float) -> complex:
        if region_index == 0:
            return G_I
        if region_index == 3:
            return 2.0 * radius * F_IV
        if region_index == 4:
            return 2.0 * radius * F_V
        return 0.0j

    def particular_Htheta(region_index: int, radius: float) -> complex:
        M_theta_region = M_theta if region_index == 0 else 0.0j
        return -(particular_dA(region_index, radius) / mu[region_index]) - (
            params.mu_0 / mu[region_index]
        ) * M_theta_region

    # (41) at R1: Htheta^I = 0
    R1 = bounds[0][0]
    da, db = _basis_derivatives(order, R1, *bounds[0])
    matrix[0, col(0, "a")] = -(da / mu[0])
    matrix[0, col(0, "b")] = -(db / mu[0])
    rhs[0] = -particular_Htheta(0, R1)

    # Interfaces R2..R5
    interfaces = [
        (bounds[0][1], 0, 1),
        (bounds[1][1], 1, 2),
        (bounds[2][1], 2, 3),
        (bounds[3][1], 3, 4),
    ]

    row = 1
    for radius, left_region, right_region in interfaces:
        # Continuity of Az
        left_a, left_b = _basis_values(order, radius, *bounds[left_region])
        right_a, right_b = _basis_values(order, radius, *bounds[right_region])
        matrix[row, col(left_region, "a")] = left_a
        matrix[row, col(left_region, "b")] = left_b
        matrix[row, col(right_region, "a")] = -right_a
        matrix[row, col(right_region, "b")] = -right_b
        rhs[row] = particular_A(right_region, radius) - particular_A(left_region, radius)
        row += 1

        # Continuity of Htheta
        left_da, left_db = _basis_derivatives(order, radius, *bounds[left_region])
        right_da, right_db = _basis_derivatives(order, radius, *bounds[right_region])
        matrix[row, col(left_region, "a")] = -(left_da / mu[left_region])
        matrix[row, col(left_region, "b")] = -(left_db / mu[left_region])
        matrix[row, col(right_region, "a")] = right_da / mu[right_region]
        matrix[row, col(right_region, "b")] = right_db / mu[right_region]
        rhs[row] = particular_Htheta(right_region, radius) - particular_Htheta(left_region, radius)
        row += 1

    # (41) at R6: Htheta^V = 0
    R6 = bounds[4][1]
    da, db = _basis_derivatives(order, R6, *bounds[4])
    matrix[row, col(4, "a")] = -(da / mu[4])
    matrix[row, col(4, "b")] = -(db / mu[4])
    rhs[row] = -particular_Htheta(4, R6)

    return matrix, rhs


def pm_only_region_waveforms(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
) -> dict[str, np.ndarray]:
    return {
        "I": sample_region_i_permeability(theta_mech, params),
        "II": sample_constant_permeability(theta_mech, 1.0),
        "III": sample_region_iii_permeability(theta_mech, params),
        "IV": sample_region_iv_v_permeability(theta_mech, params),
        "V": sample_region_iv_v_permeability(theta_mech, params),
    }


def _build_region_operator(
    region_name: str,
    inner_radius: float,
    outer_radius: float,
    theta_mech: np.ndarray,
    permeability_waveform: np.ndarray,
    harmonic_orders: np.ndarray,
) -> CoupledRegionOperators:
    mu_c_r, mu_c_r_inv, mu_c_theta, mu_c_theta_inv = convolution_matrices_from_waveforms(
        theta_mech=theta_mech,
        permeability_waveform=permeability_waveform,
        orders=harmonic_orders,
    )
    K_theta = k_theta_matrix(harmonic_orders)
    V = mu_c_theta @ K_theta @ mu_c_r_inv @ K_theta
    U = mu_c_theta @ K_theta @ mu_c_r_inv
    eigenvalues, W = np.linalg.eig(V)
    lambda_values = np.sqrt(eigenvalues.astype(complex))
    W_inv = np.linalg.inv(W)
    return CoupledRegionOperators(
        region_name=region_name,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        mu_c_r=mu_c_r,
        mu_c_r_inv=mu_c_r_inv,
        mu_c_theta=mu_c_theta,
        mu_c_theta_inv=mu_c_theta_inv,
        V=V,
        U=U,
        W=W,
        W_inv=W_inv,
        lambda_values=lambda_values,
    )


def build_pm_only_coupled_region_operators(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
    harmonic_orders: np.ndarray,
) -> dict[str, CoupledRegionOperators]:
    bounds = region_bounds(params)
    waveforms = pm_only_region_waveforms(theta_mech, params)
    region_names = ("I", "II", "III", "IV", "V")
    operators: dict[str, CoupledRegionOperators] = {}
    for idx, region_name in enumerate(region_names):
        inner_radius, outer_radius = bounds[idx]
        operators[region_name] = _build_region_operator(
            region_name=region_name,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            theta_mech=theta_mech,
            permeability_waveform=waveforms[region_name],
            harmonic_orders=harmonic_orders,
        )
    return operators


def _block_slice(region_index: int, coeff_kind: str, harmonic_count: int) -> slice:
    offset = 2 * region_index * harmonic_count
    if coeff_kind == "a":
        return slice(offset, offset + harmonic_count)
    return slice(offset + harmonic_count, offset + 2 * harmonic_count)


def _homogeneous_matrices(
    operator: CoupledRegionOperators,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lambda_values = operator.lambda_values
    ratio_a = np.power(radius / operator.outer_radius, lambda_values)
    ratio_b = np.power(radius / operator.inner_radius, -lambda_values)
    Az_a = operator.W @ np.diag(ratio_a)
    Az_b = operator.W @ np.diag(ratio_b)
    dAz_a = operator.W @ np.diag((lambda_values / radius) * ratio_a)
    dAz_b = operator.W @ np.diag((-lambda_values / radius) * ratio_b)
    return Az_a, Az_b, dAz_a, dAz_b


def _pm_only_particular_vectors(
    operator: CoupledRegionOperators,
    radius: float,
    M_r: np.ndarray,
    M_theta: np.ndarray,
    mu_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if operator.region_name != "I":
        zeros = np.zeros(len(M_r), dtype=complex)
        return zeros, zeros, zeros

    identity = np.eye(len(M_r), dtype=complex)
    rhs = 1j * mu_0 * (operator.U @ M_r) + mu_0 * M_theta
    G_I = np.linalg.solve(operator.V - identity, rhs)
    Az_part = radius * G_I
    dAz_part = G_I
    Htheta_part = -(operator.mu_c_theta_inv @ dAz_part) - mu_0 * (
        operator.mu_c_theta_inv @ M_theta
    )
    return Az_part, dAz_part, Htheta_part


def assemble_coupled_pm_only_system(
    params: LinearPMModelParams,
    operators: dict[str, CoupledRegionOperators],
    harmonic_orders: np.ndarray,
    M_r: np.ndarray,
    M_theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    harmonic_count = len(harmonic_orders)
    system_size = 10 * harmonic_count
    matrix = np.zeros((system_size, system_size), dtype=complex)
    rhs = np.zeros(system_size, dtype=complex)
    region_names = ("I", "II", "III", "IV", "V")
    region_index = {name: idx for idx, name in enumerate(region_names)}

    def htheta_basis(
        operator: CoupledRegionOperators,
        dAz_a: np.ndarray,
        dAz_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            -(operator.mu_c_theta_inv @ dAz_a),
            -(operator.mu_c_theta_inv @ dAz_b),
        )

    row = 0
    region_I = operators["I"]
    _, _, dAz_a, dAz_b = _homogeneous_matrices(region_I, region_I.inner_radius)
    H_a, H_b = htheta_basis(region_I, dAz_a, dAz_b)
    _, _, H_part = _pm_only_particular_vectors(
        region_I, region_I.inner_radius, M_r, M_theta, params.mu_0
    )
    row_slice = slice(row, row + harmonic_count)
    matrix[row_slice, _block_slice(region_index["I"], "a", harmonic_count)] = H_a
    matrix[row_slice, _block_slice(region_index["I"], "b", harmonic_count)] = H_b
    rhs[row_slice] = -H_part
    row += harmonic_count

    interfaces = (("I", "II"), ("II", "III"), ("III", "IV"), ("IV", "V"))
    interface_radii = (
        params.air_gap_inner_radius,
        params.air_gap_outer_radius,
        params.slot_opening_radius,
        params.region_iv_boundary_radius,
    )

    for radius, (left_name, right_name) in zip(interface_radii, interfaces):
        left = operators[left_name]
        right = operators[right_name]
        left_idx = region_index[left_name]
        right_idx = region_index[right_name]

        Az_a_l, Az_b_l, dAz_a_l, dAz_b_l = _homogeneous_matrices(left, radius)
        Az_a_r, Az_b_r, dAz_a_r, dAz_b_r = _homogeneous_matrices(right, radius)
        H_a_l, H_b_l = htheta_basis(left, dAz_a_l, dAz_b_l)
        H_a_r, H_b_r = htheta_basis(right, dAz_a_r, dAz_b_r)

        Az_part_l, _, H_part_l = _pm_only_particular_vectors(
            left, radius, M_r, M_theta, params.mu_0
        )
        Az_part_r, _, H_part_r = _pm_only_particular_vectors(
            right, radius, M_r, M_theta, params.mu_0
        )

        row_slice = slice(row, row + harmonic_count)
        matrix[row_slice, _block_slice(left_idx, "a", harmonic_count)] = Az_a_l
        matrix[row_slice, _block_slice(left_idx, "b", harmonic_count)] = Az_b_l
        matrix[row_slice, _block_slice(right_idx, "a", harmonic_count)] = -Az_a_r
        matrix[row_slice, _block_slice(right_idx, "b", harmonic_count)] = -Az_b_r
        rhs[row_slice] = Az_part_r - Az_part_l
        row += harmonic_count

        row_slice = slice(row, row + harmonic_count)
        matrix[row_slice, _block_slice(left_idx, "a", harmonic_count)] = H_a_l
        matrix[row_slice, _block_slice(left_idx, "b", harmonic_count)] = H_b_l
        matrix[row_slice, _block_slice(right_idx, "a", harmonic_count)] = -H_a_r
        matrix[row_slice, _block_slice(right_idx, "b", harmonic_count)] = -H_b_r
        rhs[row_slice] = H_part_r - H_part_l
        row += harmonic_count

    region_V = operators["V"]
    _, _, dAz_a, dAz_b = _homogeneous_matrices(region_V, region_V.outer_radius)
    H_a, H_b = htheta_basis(region_V, dAz_a, dAz_b)
    _, _, H_part = _pm_only_particular_vectors(
        region_V, region_V.outer_radius, M_r, M_theta, params.mu_0
    )
    row_slice = slice(row, row + harmonic_count)
    matrix[row_slice, _block_slice(region_index["V"], "a", harmonic_count)] = H_a
    matrix[row_slice, _block_slice(region_index["V"], "b", harmonic_count)] = H_b
    rhs[row_slice] = -H_part
    return matrix, rhs


def evaluate_coupled_region_solution(
    operator: CoupledRegionOperators,
    radius: float,
    a_coeff: np.ndarray,
    b_coeff: np.ndarray,
    M_r: np.ndarray,
    M_theta: np.ndarray,
    mu_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    Az_a, Az_b, dAz_a, dAz_b = _homogeneous_matrices(operator, radius)
    Az_part, dAz_part, _ = _pm_only_particular_vectors(
        operator, radius, M_r, M_theta, mu_0
    )
    Az_coeff = Az_a @ a_coeff + Az_b @ b_coeff + Az_part
    dAz_dr_coeff = dAz_a @ a_coeff + dAz_b @ b_coeff + dAz_part
    return Az_coeff, dAz_dr_coeff

from __future__ import annotations

import numpy as np

from config import LinearPMModelParams, RegionPermeabilityDescriptor
from excitation import exact_radial_magnetization


def constant_region_mu_r(params: LinearPMModelParams) -> dict[str, float]:
    return {
        "I": 1.0,
        "II": params.magnet_mu_r_linear,
        "III": 1.0,
        "IV": params.core_mu_r_linear,
        "V": params.core_mu_r_linear,
    }


def constant_convolution_matrix(mu_r: float, harmonic_count: int) -> np.ndarray:
    return mu_r * np.eye(harmonic_count, dtype=float)


def constant_permeability_matrices(
    params: LinearPMModelParams, harmonic_count: int
) -> dict[str, np.ndarray]:
    return {
        region: constant_convolution_matrix(mu_r, harmonic_count)
        for region, mu_r in constant_region_mu_r(params).items()
    }


def harmonic_order_vector(max_harmonic: int) -> np.ndarray:
    return np.arange(-max_harmonic, max_harmonic + 1, dtype=int)


def signed_nonzero_harmonic_orders(max_harmonic: int) -> np.ndarray:
    return np.concatenate(
        (
            np.arange(-max_harmonic, 0, dtype=int),
            np.arange(1, max_harmonic + 1, dtype=int),
        )
    )


def k_theta_matrix(orders: np.ndarray) -> np.ndarray:
    return np.diag(orders.astype(complex))


def toeplitz_from_cfs(coefficients: dict[int, complex], orders: np.ndarray) -> np.ndarray:
    size = len(orders)
    matrix = np.zeros((size, size), dtype=complex)
    for row_idx, row_order in enumerate(orders):
        for col_idx, col_order in enumerate(orders):
            matrix[row_idx, col_idx] = coefficients.get(int(row_order - col_order), 0.0)
    return matrix


def inverse_cfs_coefficients(coefficients: dict[int, complex]) -> dict[int, complex]:
    inverse = {}
    for order, value in coefficients.items():
        if np.isclose(value, 0.0):
            inverse[order] = 0.0
        else:
            inverse[order] = 1.0 / value
    return inverse


def region_i_mu_coefficients(
    params: LinearPMModelParams, max_harmonic: int
) -> dict[int, complex]:
    """Paper eqs. (12)-(13) for region I under the linear baseline geometry."""

    coeffs: dict[int, complex] = {}
    alpha_rl = params.magnet_air_gap_arc_mech
    alpha_rt = params.magnet_arc_mech
    pole_count = params.pole_count

    coeffs[0] = (
        params.mu_0 * alpha_rl + params.mu_0 * params.magnet_mu_r_linear * alpha_rt
    ) / (alpha_rl + alpha_rt)

    for n in harmonic_order_vector(max_harmonic):
        if n == 0:
            continue
        total = 0.0j
        for v in range(1, pole_count + 1):
            theta_v = (2 * v - 1) * np.pi / pole_count + params.initial_rotor_angle_mech
            phase = np.exp(1j * n * (theta_v + alpha_rl / 2.0))
            air_term = params.mu_0 * (1.0 - np.exp(-1j * n * alpha_rl))
            magnet_term = params.mu_0 * params.magnet_mu_r_linear * (
                np.exp(1j * n * alpha_rt) - 1.0
            )
            total += phase * (air_term + magnet_term)
        coeffs[int(n)] = total / (2.0 * np.pi * 1j * n)

    return coeffs


def generic_region_mu_coefficients(
    theta_i: np.ndarray,
    beta_sl: float,
    subsection_width: float,
    subsection_mu: np.ndarray,
    max_harmonic: int,
    mu_0: float,
) -> dict[int, complex]:
    """Paper eqs. (14)-(15) in a geometry-driven generic form.

    Parameters:
    - theta_i: center angle of each slot opening for the target region
    - beta_sl: slot-opening width in that region
    - subsection_width: beta_sub = beta_sl / k
    - subsection_mu: shape (Ns, k), relative permeability for each subsection
    """

    coeffs: dict[int, complex] = {}
    theta_i = np.asarray(theta_i, dtype=float)
    subsection_mu = np.asarray(subsection_mu, dtype=float)
    slot_count, subsection_count = subsection_mu.shape

    coeffs[0] = (
        np.sum(mu_0 * beta_sl + np.sum(subsection_mu, axis=1) * subsection_width)
        / (2.0 * np.pi)
    )

    for n in harmonic_order_vector(max_harmonic):
        if n == 0:
            continue
        total = 0.0j
        for slot_idx in range(slot_count):
            theta_center = theta_i[slot_idx]
            air_term = mu_0 * (
                np.exp(1j * n * (theta_center + beta_sl / 2.0))
                - np.exp(1j * n * (theta_center - beta_sl / 2.0))
            )
            total += air_term
            for p in range(1, subsection_count + 1):
                start = theta_center + beta_sl / 2.0 + p * subsection_width
                end = theta_center + beta_sl / 2.0 + (p - 1) * subsection_width
                total += subsection_mu[slot_idx, p - 1] * (
                    np.exp(1j * n * start) - np.exp(1j * n * end)
                )
        coeffs[int(n)] = total / (2.0 * np.pi * 1j * n)

    return coeffs


def permeability_convolution_matrices_from_coefficients(
    coefficients: dict[int, complex], orders: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu_c = toeplitz_from_cfs(coefficients, orders)
    mu_c_inv = toeplitz_from_cfs(inverse_cfs_coefficients(coefficients), orders)
    return mu_c, mu_c_inv


def coefficient_dict_from_waveform(
    theta_mech: np.ndarray,
    waveform: np.ndarray,
    max_order: int,
) -> dict[int, complex]:
    coeffs: dict[int, complex] = {}
    for order in range(-max_order, max_order + 1):
        coeffs[order] = np.mean(waveform * np.exp(-1j * order * theta_mech))
    return coeffs


def convolution_matrices_from_waveforms(
    theta_mech: np.ndarray,
    permeability_waveform: np.ndarray,
    orders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_diff = int(np.max(np.abs(orders[:, None] - orders[None, :])))
    mu_coeffs = coefficient_dict_from_waveform(theta_mech, permeability_waveform, max_diff)
    mu_rec_coeffs = coefficient_dict_from_waveform(
        theta_mech,
        1.0 / permeability_waveform,
        max_diff,
    )
    mu_c_r = toeplitz_from_cfs(mu_coeffs, orders)
    mu_c_theta = toeplitz_from_cfs(mu_rec_coeffs, orders)
    mu_c_r_inv = np.linalg.inv(mu_c_r)
    mu_c_theta_inv = np.linalg.inv(mu_c_theta)
    return mu_c_r, mu_c_r_inv, mu_c_theta, mu_c_theta_inv


def sample_region_i_permeability(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
) -> np.ndarray:
    magnet_mask = np.abs(exact_radial_magnetization(theta_mech, params)) > 0.5 * params.M0
    waveform = np.ones_like(theta_mech, dtype=float)
    waveform[magnet_mask] = params.magnet_mu_r_linear
    return waveform


def sample_region_iii_permeability(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
) -> np.ndarray:
    theta_mod = np.mod(theta_mech, 2.0 * np.pi)
    local_angle = (
        theta_mod - (np.floor(theta_mod / params.slot_pitch_mech) + 0.5) * params.slot_pitch_mech
    )
    local_angle = (local_angle + np.pi) % (2.0 * np.pi) - np.pi
    air_mask = np.abs(local_angle) <= 0.5 * params.slot_opening_angle_mech
    waveform = np.full_like(theta_mech, params.core_mu_r_linear, dtype=float)
    waveform[air_mask] = 1.0
    return waveform


def sample_region_iv_v_permeability(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
) -> np.ndarray:
    theta_mod = np.mod(theta_mech, 2.0 * np.pi)
    local_angle = (
        theta_mod - (np.floor(theta_mod / params.slot_pitch_mech) + 0.5) * params.slot_pitch_mech
    )
    local_angle = (local_angle + np.pi) % (2.0 * np.pi) - np.pi
    air_mask = np.abs(local_angle) <= 0.5 * params.slot_body_angle_mech
    waveform = np.full_like(theta_mech, params.core_mu_r_linear, dtype=float)
    waveform[air_mask] = params.slot_region_mu_r_linear
    return waveform


def sample_constant_permeability(
    theta_mech: np.ndarray,
    relative_mu: float,
) -> np.ndarray:
    return np.full_like(theta_mech, relative_mu, dtype=float)


def slot_center_angles(params: LinearPMModelParams) -> np.ndarray:
    slot_ids = np.arange(params.stator_slots, dtype=float)
    return (slot_ids + 0.5) * params.slot_pitch_mech


def assumed_region_descriptor(
    params: LinearPMModelParams, region_name: str
) -> RegionPermeabilityDescriptor:
    """Geometry-driven placeholder for Fig. 5(b) regions.

    This is the remaining inferred part of the 5-region setup:
    - region III uses the assumed slot-opening angle
    - regions IV and V use one full slot pitch
    - subsection permeabilities are set to linear core values
    """

    theta_i = slot_center_angles(params)

    if region_name == "III":
        subsection_count = params.region_iii_subsections
        beta_sl = params.slot_opening_angle_mech
    elif region_name == "IV":
        subsection_count = params.region_iv_subsections
        beta_sl = params.slot_pitch_mech
    elif region_name == "V":
        subsection_count = params.region_v_subsections
        beta_sl = params.slot_pitch_mech
    else:
        raise ValueError(f"unsupported region name: {region_name}")

    beta_sub = beta_sl / subsection_count
    subsection_mu = params.mu_0 * params.core_mu_r_linear * np.ones(
        (params.stator_slots, subsection_count),
        dtype=float,
    )
    return RegionPermeabilityDescriptor(
        region_name=region_name,
        theta_i=theta_i,
        beta_sl=beta_sl,
        beta_sub=beta_sub,
        subsection_mu=subsection_mu,
        subsection_count=subsection_count,
    )


def assumed_region_coefficients(
    params: LinearPMModelParams, region_name: str, max_harmonic: int
) -> tuple[RegionPermeabilityDescriptor, dict[int, complex]]:
    descriptor = assumed_region_descriptor(params, region_name)
    coeffs = generic_region_mu_coefficients(
        theta_i=descriptor.theta_i,
        beta_sl=descriptor.beta_sl,
        subsection_width=descriptor.beta_sub,
        subsection_mu=descriptor.subsection_mu,
        max_harmonic=max_harmonic,
        mu_0=params.mu_0,
    )
    return descriptor, coeffs

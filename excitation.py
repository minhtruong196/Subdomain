from __future__ import annotations

import numpy as np

from config import CurrentLoadingSpectrum, HarmonicSpectrum, LinearPMModelParams


def mechanical_angle_grid(params: LinearPMModelParams) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, params.sample_count, endpoint=False)


def exact_radial_magnetization(
    theta_mech: np.ndarray, params: LinearPMModelParams
) -> np.ndarray:
    theta_rotor = theta_mech - params.initial_rotor_angle_mech
    phi = np.mod(params.pole_pairs * theta_rotor, 2.0 * np.pi)
    phi_wrapped = (phi + np.pi) % (2.0 * np.pi) - np.pi

    half_arc = 0.5 * params.magnet_arc_elec
    positive_pole = np.abs(phi_wrapped) <= half_arc
    negative_pole = np.abs(np.abs(phi_wrapped) - np.pi) <= half_arc

    Mr = np.zeros_like(theta_mech)
    Mr[positive_pole] = params.M0
    Mr[negative_pole] = -params.M0
    return Mr


def radial_pm_harmonics(params: LinearPMModelParams) -> HarmonicSpectrum:
    nu = params.odd_electrical_harmonics
    k = nu * params.pole_pairs
    amp = (4.0 * params.M0 / (np.pi * nu)) * np.sin(0.5 * nu * params.magnet_arc_elec)
    return HarmonicSpectrum(
        electrical_orders=nu,
        mechanical_orders=k,
        amplitudes=amp,
    )


def _delta_f1(theta_1: float, theta_2: float, n: int, theta_v_prime: float) -> complex:
    if n == 1:
        regular = (
            np.exp(1j * (2.0 * theta_2 - theta_v_prime))
            - np.exp(1j * (2.0 * theta_1 - theta_v_prime))
        ) / 2.0
        singular_limit = 1j * np.exp(1j * theta_v_prime) * (theta_2 - theta_1)
        return (regular + singular_limit) / (4.0 * np.pi * 1j)

    term_1 = (
        np.exp(1j * theta_2 * (n + 1) - 1j * theta_v_prime)
        - np.exp(1j * theta_1 * (n + 1) - 1j * theta_v_prime)
    ) / (n + 1)
    term_2 = (
        np.exp(1j * theta_2 * (n - 1) + 1j * theta_v_prime)
        - np.exp(1j * theta_1 * (n - 1) + 1j * theta_v_prime)
    ) / (n - 1)
    return (term_1 + term_2) / (4.0 * np.pi * 1j)


def _delta_f2(theta_1: float, theta_2: float, n: int, theta_v_prime: float) -> complex:
    if n == 1:
        regular = (
            np.exp(1j * (2.0 * theta_2 - theta_v_prime))
            - np.exp(1j * (2.0 * theta_1 - theta_v_prime))
        ) / 2.0
        singular_limit = 1j * np.exp(1j * theta_v_prime) * (theta_2 - theta_1)
        return (regular - singular_limit) / (4.0 * np.pi)

    term_1 = (
        np.exp(1j * theta_2 * (n + 1) - 1j * theta_v_prime)
        - np.exp(1j * theta_1 * (n + 1) - 1j * theta_v_prime)
    ) / (n + 1)
    term_2 = (
        np.exp(1j * theta_2 * (n - 1) + 1j * theta_v_prime)
        - np.exp(1j * theta_1 * (n - 1) + 1j * theta_v_prime)
    ) / (n - 1)
    return (term_1 - term_2) / (4.0 * np.pi)


def pm_cfs_coefficients(
    params: LinearPMModelParams,
    mechanical_orders: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex Fourier-series PM coefficients from paper eqs. (24)-(29).

    Current implementation keeps the linear baseline simplification:
    - one radial layer
    - one tangential segment per pole
    - uniform Br across magnets
    """

    if mechanical_orders is None:
        electrical_orders = params.odd_electrical_harmonics.copy()
        orders = electrical_orders * params.pole_pairs
    else:
        orders = np.asarray(mechanical_orders, dtype=int)
    M_r = np.zeros(len(orders), dtype=complex)
    M_theta = np.zeros(len(orders), dtype=complex)

    gamma = params.magnet_arc_mech / params.magnet_tangential_segments

    for order_idx, n in enumerate(orders):
        for v in range(1, params.pole_count + 1):
            theta_v_prime = (
                (2 * v - 1) * np.pi / params.pole_count + params.initial_rotor_angle_mech
            )
            vartheta = theta_v_prime - params.magnet_arc_mech / 2.0

            for k1 in range(1, params.magnet_tangential_segments + 1):
                theta_1 = vartheta + (k1 - 1) * gamma
                theta_2 = vartheta + k1 * gamma
                M0_vk = ((-1) ** v) * params.Br_60C / params.mu_0

                M_r[order_idx] += M0_vk * _delta_f1(theta_1, theta_2, int(n), theta_v_prime)
                M_theta[order_idx] += M0_vk * _delta_f2(
                    theta_1, theta_2, int(n), theta_v_prime
                )

    return orders, M_r, M_theta


def slot_current_density_waveform(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
    electrical_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    J1, J2 = slot_current_densities(params, electrical_angle=electrical_angle)
    theta_mod = np.mod(theta_mech, 2.0 * np.pi)
    slot_index = np.floor(theta_mod / params.slot_pitch_mech).astype(int) % params.stator_slots
    local_angle = theta_mod - slot_index * params.slot_pitch_mech
    Jz = np.where(local_angle < params.coil_width_mech, J1[slot_index], J2[slot_index])
    return Jz, J1, J2


def complex_fourier_coefficients(
    theta_mech: np.ndarray,
    waveform: np.ndarray,
    mechanical_orders: np.ndarray,
) -> np.ndarray:
    coeffs = np.empty(len(mechanical_orders), dtype=complex)
    for idx, order in enumerate(mechanical_orders):
        coeffs[idx] = np.mean(waveform * np.exp(-1j * order * theta_mech))
    return coeffs


def balanced_phase_currents(
    params: LinearPMModelParams, electrical_angle: float | None = None
) -> tuple[float, float, float]:
    delta = params.torque_angle_elec if electrical_angle is None else electrical_angle
    delta += params.stator_q_axis_offset_elec
    ia = params.phase_current_peak * np.cos(delta)
    ib = params.phase_current_peak * np.cos(delta - 2.0 * np.pi / 3.0)
    ic = params.phase_current_peak * np.cos(delta + 2.0 * np.pi / 3.0)
    return float(ia), float(ib), float(ic)


def dq_currents(
    params: LinearPMModelParams, electrical_angle: float | None = None
) -> tuple[float, float]:
    delta = params.torque_angle_elec if electrical_angle is None else electrical_angle
    iq = params.phase_current_peak * np.cos(delta)
    id_ = params.phase_current_peak * np.sin(delta)
    return float(id_), float(iq)


def reconstruct_radial_magnetization(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
    spectrum: HarmonicSpectrum,
    odd_harmonic_limit: int | None = None,
) -> np.ndarray:
    if odd_harmonic_limit is None:
        nu = spectrum.electrical_orders
        amp = spectrum.amplitudes
    else:
        keep = spectrum.electrical_orders <= odd_harmonic_limit
        nu = spectrum.electrical_orders[keep]
        amp = spectrum.amplitudes[keep]

    phase = np.outer(nu * params.pole_pairs, theta_mech)
    return np.sum(amp[:, None] * np.cos(phase), axis=0)


def slot_connection_matrices() -> tuple[np.ndarray, np.ndarray]:
    C1 = np.array(
        [
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    C2 = -np.array(
        [
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        ],
        dtype=float,
    )
    return C1, C2


def slot_current_densities(
    params: LinearPMModelParams, electrical_angle: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    ia, ib, ic = balanced_phase_currents(params, electrical_angle=electrical_angle)
    phase_currents = np.array([ia, ib, ic], dtype=float)
    C1, C2 = slot_connection_matrices()
    scale = params.turns_per_coil_in_slot / params.slot_area_region_v
    J1 = scale * C1.T @ phase_currents
    J2 = scale * C2.T @ phase_currents
    return J1, J2


def slot_current_loading_waveform(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
    electrical_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    J1, J2 = slot_current_densities(params, electrical_angle=electrical_angle)
    theta_mod = np.mod(theta_mech, 2.0 * np.pi)
    slot_index = np.floor(theta_mod / params.slot_pitch_mech).astype(int) % params.stator_slots
    local_angle = theta_mod - slot_index * params.slot_pitch_mech

    K1 = params.slot_current_to_surface_factor * J1
    K2 = params.slot_current_to_surface_factor * J2
    Kz = np.where(local_angle < params.coil_width_mech, K1[slot_index], K2[slot_index])
    return Kz, J1, J2


def slot_current_loading_harmonics(
    theta_mech: np.ndarray,
    current_loading: np.ndarray,
    params: LinearPMModelParams,
) -> CurrentLoadingSpectrum:
    orders = np.arange(1, params.max_current_harmonic + 1, dtype=int)
    cosine = np.empty_like(orders, dtype=float)
    sine = np.empty_like(orders, dtype=float)

    for idx, order in enumerate(orders):
        cosine[idx] = 2.0 * np.mean(current_loading * np.cos(order * theta_mech))
        sine[idx] = 2.0 * np.mean(current_loading * np.sin(order * theta_mech))

    return CurrentLoadingSpectrum(orders=orders, cosine=cosine, sine=sine)

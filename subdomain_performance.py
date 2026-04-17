from __future__ import annotations

import numpy as np

from subdomain_airgap import full_airgap_flux_density
from subdomain_boundary import SegmentSolutionSet, UnknownLayout, solve_segments_for_deltas
from subdomain_config import MachineConfig


def cogging_torque_waveform(config: MachineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (30): T = Lef Rg^2 / mu0 * integral Br(Rg,theta) Btheta(Rg,theta) dtheta."""

    rotor_pos_deg = np.linspace(0.0, 30.0, config.operating.torque_position_count)
    delta_rad = np.deg2rad(rotor_pos_deg)
    theta = np.linspace(0.0, 2.0 * np.pi, config.operating.torque_theta_count, endpoint=False)

    solved_segments = solve_segments_for_deltas(config, delta_rad)
    torque_nm = np.empty_like(delta_rad)

    for idx in range(len(delta_rad)):
        Br, Btheta = full_airgap_flux_density(solved_segments, idx, theta, config)
        integral = 2.0 * np.pi * np.mean(Br * Btheta)
        torque_nm[idx] = (
            config.operating.active_axial_length_m
            * config.stator.airgap_radius_m**2
            / config.magnet.mu0
            * integral
        )

    torque_nm -= np.mean(torque_nm)
    return rotor_pos_deg, 1.0e3 * torque_nm


def odd_periodic_slot_map(slot_number: int, config: MachineConfig) -> tuple[int, float]:
    """Map a global slot to the reduced Z/(2c) slot index and odd-periodic sign."""

    reduced = config.reduced_slot_count
    zero_based = (slot_number - 1) % config.geometry.slots
    sector = zero_based // reduced
    local_slot_idx = zero_based % reduced
    sign = -1.0 if sector % 2 else 1.0
    return local_slot_idx, sign


def phase_a_coils(config: MachineConfig) -> list[tuple[int, int, float]]:
    """Return (start_slot, return_slot, sign) for the phase-A winding."""

    coils: list[tuple[int, int, float]] = []
    Z = config.geometry.slots
    p = config.geometry.pole_pairs
    y = config.operating.coil_pitch_slots

    for slot in range(1, Z + 1):
        theta_center = (2 * slot - 1) * np.pi / Z
        elec_deg = np.rad2deg(p * theta_center) % 360.0

        if 0.0 <= elec_deg < 60.0:
            return_slot = ((slot + y - 1) % Z) + 1
            coils.append((slot, return_slot, 1.0))

    return coils


def slot_constant_potential(
    solution: np.ndarray,
    layout: UnknownLayout,
    slot_number: int,
    upper_half: bool,
    config: MachineConfig,
) -> float:
    local_slot_idx, periodic_sign = odd_periodic_slot_map(slot_number, config)
    if upper_half:
        return periodic_sign * float(solution[layout.B03(local_slot_idx)])
    return periodic_sign * float(solution[layout.B04(local_slot_idx)])


def phase_a_flux_linkage(
    solved_segments: list[SegmentSolutionSet],
    delta_index: int,
    config: MachineConfig,
) -> float:
    """Eq. (31) checkpoint for phase-A flux linkage."""

    coils = phase_a_coils(config)
    turns_per_coil = config.operating.series_turns_per_phase / len(coils)
    flux = 0.0

    for segment in solved_segments:
        solution = segment.solutions[:, delta_index]
        for start_slot, return_slot, coil_sign in coils:
            a_start = slot_constant_potential(solution, segment.layout, start_slot, True, config)
            a_return = slot_constant_potential(solution, segment.layout, return_slot, False, config)
            flux += coil_sign * config.operating.active_axial_length_m * turns_per_coil * (a_start - a_return)

    return float(flux)


def no_load_back_emf_waveform(config: MachineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elec_deg = np.linspace(0.0, 360.0, config.operating.emf_sample_count, endpoint=False)
    delta_rad = np.deg2rad(elec_deg) / config.geometry.pole_pairs
    solved_segments = solve_segments_for_deltas(config, delta_rad)

    flux_linkage = np.array(
        [phase_a_flux_linkage(solved_segments, idx, config) for idx in range(len(delta_rad))],
        dtype=float,
    )

    step = delta_rad[1] - delta_rad[0]
    dflux_ddelta = (np.roll(flux_linkage, -1) - np.roll(flux_linkage, 1)) / (2.0 * step)
    emf_v = -config.operating.omega_mech_rad_s * dflux_ddelta
    emf_v -= np.mean(emf_v)
    return elec_deg, emf_v, flux_linkage


def periodic_shift_deg(x_deg: np.ndarray, signal: np.ndarray, shift_deg: float) -> np.ndarray:
    """Evaluate a 360-electrical-degree periodic waveform at x_deg - shift_deg."""

    period = 360.0
    x = np.asarray(x_deg, dtype=float)
    y = np.asarray(signal, dtype=float)
    xp = np.concatenate([x, [x[0] + period]])
    fp = np.concatenate([y, [y[0]]])
    return np.interp((x - shift_deg) % period, xp, fp)


def line_to_line_back_emf_from_phase_a(elec_deg: np.ndarray, phase_a_emf_v: np.ndarray) -> np.ndarray:
    """Balanced-Y conversion: e_ab(theta) = e_a(theta) - e_b(theta)."""

    phase_b_emf_v = periodic_shift_deg(elec_deg, phase_a_emf_v, 120.0)
    line_line_emf_v = phase_a_emf_v - phase_b_emf_v
    line_line_emf_v -= np.mean(line_line_emf_v)
    return line_line_emf_v


def fundamental_peak(signal: np.ndarray) -> float:
    coeff = np.fft.rfft(signal) / len(signal)
    return float(2.0 * abs(coeff[1]))

from __future__ import annotations

import numpy as np

from config import CurrentLoadingSpectrum, HarmonicSpectrum, LinearPMModelParams
from excitation import (
    exact_radial_magnetization,
    reconstruct_radial_magnetization,
    slot_current_loading_harmonics,
    slot_current_loading_waveform,
)


def average_torque_maxwell(
    theta_mech: np.ndarray,
    Br: np.ndarray,
    Btheta: np.ndarray,
    radius: float,
    params: LinearPMModelParams,
) -> float:
    if Br.shape != Btheta.shape or Br.shape != theta_mech.shape:
        raise ValueError("theta_mech, Br, and Btheta must have the same shape")

    shear_stress_mean = float(np.mean(Br * Btheta) / params.mu_0)
    air_gap_surface = 2.0 * np.pi * radius * params.stack_length
    return shear_stress_mean * air_gap_surface * radius


def equivalent_pm_flux_linkage(
    torque_nm: float,
    params: LinearPMModelParams,
    electrical_angle: float | None = None,
) -> float:
    delta = params.torque_angle_elec if electrical_angle is None else electrical_angle
    iq = params.phase_current_peak * np.cos(delta)
    if np.isclose(iq, 0.0):
        return float("nan")
    return torque_nm / (1.5 * params.pole_pairs * iq)


def phase_back_emf_constants(
    lambda_pm_peak: float, params: LinearPMModelParams
) -> tuple[float, float]:
    phase_peak = params.omega_elec_rad_s * lambda_pm_peak
    phase_rms = phase_peak / np.sqrt(2.0)
    return float(phase_peak), float(phase_rms)


def rms_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    diff = reference - estimate
    return float(np.sqrt(np.mean(diff * diff)))


def convergence_rows(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
    spectrum: HarmonicSpectrum,
) -> list[tuple[int, float]]:
    exact = exact_radial_magnetization(theta_mech, params)
    rows: list[tuple[int, float]] = []
    trial_limits = [1, 3, 5, 9, 15, 31, spectrum.electrical_orders[-1]]

    for limit in trial_limits:
        reconstructed = reconstruct_radial_magnetization(
            theta_mech=theta_mech,
            params=params,
            spectrum=spectrum,
            odd_harmonic_limit=limit,
        )
        rows.append((limit, rms_error(exact, reconstructed)))

    return rows


def dominant_harmonic_rows(
    spectrum: HarmonicSpectrum, count: int = 8
) -> list[tuple[int, int, float]]:
    amps = np.abs(spectrum.amplitudes)
    order = np.argsort(amps)[::-1][:count]
    return [
        (
            int(spectrum.electrical_orders[idx]),
            int(spectrum.mechanical_orders[idx]),
            float(spectrum.amplitudes[idx]),
        )
        for idx in order
    ]


def dominant_current_loading_rows(
    spectrum: CurrentLoadingSpectrum, count: int = 8
) -> list[tuple[int, float, float, float]]:
    magnitude = np.sqrt(spectrum.cosine**2 + spectrum.sine**2)
    order = np.argsort(magnitude)[::-1][:count]
    return [
        (
            int(spectrum.orders[idx]),
            float(spectrum.cosine[idx]),
            float(spectrum.sine[idx]),
            float(magnitude[idx]),
        )
        for idx in order
    ]


def torque_angle_curve(
    theta_mech: np.ndarray,
    params: LinearPMModelParams,
    pm_spectrum: HarmonicSpectrum,
    total_air_gap_field_fn,
    point_count: int = 181,
) -> tuple[np.ndarray, np.ndarray]:
    electrical_angles_deg = np.linspace(-180.0, 180.0, point_count)
    torques = np.empty(point_count, dtype=float)

    for idx, angle_deg in enumerate(electrical_angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        current_loading, _, _ = slot_current_loading_waveform(
            theta_mech, params, electrical_angle=angle_rad
        )
        current_spectrum = slot_current_loading_harmonics(
            theta_mech, current_loading, params
        )
        Br, Btheta = total_air_gap_field_fn(
            theta_mech=theta_mech,
            radius=params.mid_gap_radius,
            params=params,
            pm_spectrum=pm_spectrum,
            current_spectrum=current_spectrum,
        )
        torques[idx] = average_torque_maxwell(
            theta_mech=theta_mech,
            Br=Br,
            Btheta=Btheta,
            radius=params.mid_gap_radius,
            params=params,
        )

    return electrical_angles_deg, torques

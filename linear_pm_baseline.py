from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LinearPMBaselineParams:
    """Minimal linear baseline for an 8-pole / 12-slot SPM machine.

    Assumptions in this file:
    - Linear magnetic materials
    - Smooth air gap, no slotting
    - Radial magnetization only
    - Slot current loading from the paper, but field solution still uses a
      smooth-air-gap linear baseline
    - High-permeability stator boundary at the air-gap outer radius

    This is a debug-friendly baseline, not the final nonlinear paper model.
    """

    pole_pairs: int = 4
    stator_slots: int = 12

    rotor_outer_radius: float = 38.0e-3
    air_gap_inner_radius: float = 43.0e-3
    air_gap_outer_radius: float = 47.0e-3
    slot_opening_radius: float = 49.8e-3
    region_iv_boundary_radius: float = 53.5e-3
    stator_outer_radius: float = 64.5e-3
    stack_length: float = 30.0e-3

    magnet_arc_ratio: float = 0.80
    Br_60C: float = 1.247
    mu_rec: float = 1.039
    mu_0: float = 4.0 * np.pi * 1.0e-7

    max_odd_harmonic: int = 45
    max_current_harmonic: int = 90
    sample_count: int = 4096

    phase_current_peak: float = 50.0
    turns_per_phase: int = 100
    turns_per_coil_in_slot: int = 14
    winding_factor_fundamental: float = 0.95
    torque_angle_elec: float = 0.0
    stator_q_axis_offset_elec: float = 0.5 * np.pi
    mechanical_speed_rpm: float = 1400.0
    output_dir: str = "outputs/linear_pm_baseline"

    @property
    def M0(self) -> float:
        return self.Br_60C / self.mu_0

    @property
    def pole_pitch_mech(self) -> float:
        return np.pi / self.pole_pairs

    @property
    def magnet_arc_mech(self) -> float:
        return self.magnet_arc_ratio * self.pole_pitch_mech

    @property
    def magnet_arc_elec(self) -> float:
        return self.magnet_arc_ratio * np.pi

    @property
    def mid_gap_radius(self) -> float:
        return 0.5 * (self.air_gap_inner_radius + self.air_gap_outer_radius)

    @property
    def odd_electrical_harmonics(self) -> np.ndarray:
        return np.arange(1, 2 * self.max_odd_harmonic, 2, dtype=int)

    @property
    def slot_pitch_mech(self) -> float:
        return 2.0 * np.pi / self.stator_slots

    @property
    def coil_width_mech(self) -> float:
        return 0.5 * self.slot_pitch_mech

    @property
    def slot_area_region_v(self) -> float:
        return 0.5 * self.coil_width_mech * (
            self.stator_outer_radius**2 - self.slot_opening_radius**2
        )

    @property
    def slot_current_to_surface_factor(self) -> float:
        return (
            self.stator_outer_radius**2 - self.slot_opening_radius**2
        ) / (2.0 * self.air_gap_outer_radius)

    @property
    def omega_mech_rad_s(self) -> float:
        return 2.0 * np.pi * self.mechanical_speed_rpm / 60.0

    @property
    def omega_elec_rad_s(self) -> float:
        return self.pole_pairs * self.omega_mech_rad_s

    @property
    def electrical_frequency_hz(self) -> float:
        return self.omega_elec_rad_s / (2.0 * np.pi)


@dataclass(frozen=True)
class HarmonicSpectrum:
    electrical_orders: np.ndarray
    mechanical_orders: np.ndarray
    amplitudes: np.ndarray


@dataclass(frozen=True)
class CurrentLoadingSpectrum:
    orders: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray


def mechanical_angle_grid(params: LinearPMBaselineParams) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, params.sample_count, endpoint=False)


def exact_radial_magnetization(
    theta_mech: np.ndarray, params: LinearPMBaselineParams
) -> np.ndarray:
    """Piecewise radial PM magnetization in mechanical coordinates."""

    phi = np.mod(params.pole_pairs * theta_mech, 2.0 * np.pi)
    phi_wrapped = (phi + np.pi) % (2.0 * np.pi) - np.pi

    half_arc = 0.5 * params.magnet_arc_elec
    positive_pole = np.abs(phi_wrapped) <= half_arc
    negative_pole = np.abs(np.abs(phi_wrapped) - np.pi) <= half_arc

    Mr = np.zeros_like(theta_mech)
    Mr[positive_pole] = params.M0
    Mr[negative_pole] = -params.M0
    return Mr


def radial_pm_harmonics(params: LinearPMBaselineParams) -> HarmonicSpectrum:
    """Analytical Fourier amplitudes for radial PM magnetization.

    The spectrum is written in electrical harmonics nu = 1, 3, 5, ...
    The corresponding mechanical space order is k = nu * pole_pairs.
    """

    nu = params.odd_electrical_harmonics
    k = nu * params.pole_pairs
    amp = (4.0 * params.M0 / (np.pi * nu)) * np.sin(0.5 * nu * params.magnet_arc_elec)
    return HarmonicSpectrum(
        electrical_orders=nu,
        mechanical_orders=k,
        amplitudes=amp,
    )


def balanced_phase_currents(
    params: LinearPMBaselineParams, electrical_angle: float | None = None
) -> tuple[float, float, float]:
    delta = params.torque_angle_elec if electrical_angle is None else electrical_angle
    delta += params.stator_q_axis_offset_elec
    ia = params.phase_current_peak * np.cos(delta)
    ib = params.phase_current_peak * np.cos(delta - 2.0 * np.pi / 3.0)
    ic = params.phase_current_peak * np.cos(delta + 2.0 * np.pi / 3.0)
    return float(ia), float(ib), float(ic)


def dq_currents(
    params: LinearPMBaselineParams, electrical_angle: float | None = None
) -> tuple[float, float]:
    """Equivalent dq currents for the chosen current-sheet convention.

    In this baseline:
    - torque_angle_elec = 0 means pure positive q-axis current
    - positive torque_angle_elec rotates current toward positive d-axis
    """

    delta = params.torque_angle_elec if electrical_angle is None else electrical_angle
    iq = params.phase_current_peak * np.cos(delta)
    id_ = params.phase_current_peak * np.sin(delta)
    return float(id_), float(iq)


def reconstruct_radial_magnetization(
    theta_mech: np.ndarray,
    params: LinearPMBaselineParams,
    spectrum: HarmonicSpectrum,
    odd_harmonic_limit: int | None = None,
) -> np.ndarray:
    """Reconstruct the radial magnetization waveform from odd harmonics."""

    if odd_harmonic_limit is None:
        nu = spectrum.electrical_orders
        amp = spectrum.amplitudes
    else:
        keep = spectrum.electrical_orders <= odd_harmonic_limit
        nu = spectrum.electrical_orders[keep]
        amp = spectrum.amplitudes[keep]

    phase = np.outer(nu * params.pole_pairs, theta_mech)
    return np.sum(amp[:, None] * np.cos(phase), axis=0)


def equivalent_air_gap_field(
    theta_mech: np.ndarray,
    radius: float,
    params: LinearPMBaselineParams,
    spectrum: HarmonicSpectrum,
    odd_harmonic_limit: int | None = None,
    rotor_electrical_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Br and Btheta in the smooth air gap.

    This uses a Laplace-region transfer from an imposed radial boundary field
    at r = air_gap_inner_radius:

        Br(R2, theta) = mu_0 * Mr(theta)

    It is intentionally simple and should be treated as a baseline proxy.
    """

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
    phase = np.outer(k, theta_mech) - np.outer(nu, np.full_like(theta_mech, rotor_electrical_angle))

    Br = np.sum((boundary_flux_amp * radial_transfer)[:, None] * np.cos(phase), axis=0)
    Btheta = np.sum(
        (boundary_flux_amp * tangential_transfer)[:, None] * np.sin(phase),
        axis=0,
    )
    return Br, Btheta


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
    params: LinearPMBaselineParams, electrical_angle: float | None = None
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
    params: LinearPMBaselineParams,
    electrical_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equivalent surface current loading built from slot current densities."""

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
    params: LinearPMBaselineParams,
) -> CurrentLoadingSpectrum:
    orders = np.arange(1, params.max_current_harmonic + 1, dtype=int)
    cosine = np.empty_like(orders, dtype=float)
    sine = np.empty_like(orders, dtype=float)

    for idx, order in enumerate(orders):
        cosine[idx] = 2.0 * np.mean(current_loading * np.cos(order * theta_mech))
        sine[idx] = 2.0 * np.mean(current_loading * np.sin(order * theta_mech))

    return CurrentLoadingSpectrum(orders=orders, cosine=cosine, sine=sine)


def slot_current_air_gap_field(
    theta_mech: np.ndarray,
    radius: float,
    params: LinearPMBaselineParams,
    current_spectrum: CurrentLoadingSpectrum,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Br and Btheta in the smooth air gap from slot current loading."""

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


def phase_winding_functions(
    theta_mech: np.ndarray, params: LinearPMBaselineParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal sinusoidal phase winding functions [turn/rad]."""

    turns_per_mech_rad_peak = (
        params.winding_factor_fundamental * params.turns_per_phase / np.pi
    )
    electrical_angle = params.pole_pairs * theta_mech

    Na = turns_per_mech_rad_peak * np.cos(electrical_angle)
    Nb = turns_per_mech_rad_peak * np.cos(electrical_angle - 2.0 * np.pi / 3.0)
    Nc = turns_per_mech_rad_peak * np.cos(electrical_angle + 2.0 * np.pi / 3.0)
    return Na, Nb, Nc


def total_air_gap_field(
    theta_mech: np.ndarray,
    radius: float,
    params: LinearPMBaselineParams,
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


def phase_flux_linkages_from_pm(
    theta_mech: np.ndarray,
    params: LinearPMBaselineParams,
    pm_spectrum: HarmonicSpectrum,
    radius: float | None = None,
    rotor_electrical_angle: float = 0.0,
) -> tuple[float, float, float]:
    """Open-circuit phase flux linkages from PM air-gap field and winding functions."""

    eval_radius = params.mid_gap_radius if radius is None else radius
    Br_pm, _ = equivalent_air_gap_field(
        theta_mech=theta_mech,
        radius=eval_radius,
        params=params,
        spectrum=pm_spectrum,
        rotor_electrical_angle=rotor_electrical_angle,
    )
    Na, Nb, Nc = phase_winding_functions(theta_mech, params)
    area_weight = params.stack_length * eval_radius

    lambda_a = area_weight * np.trapezoid(Na * Br_pm, theta_mech)
    lambda_b = area_weight * np.trapezoid(Nb * Br_pm, theta_mech)
    lambda_c = area_weight * np.trapezoid(Nc * Br_pm, theta_mech)
    return float(lambda_a), float(lambda_b), float(lambda_c)


def open_circuit_pm_waveforms(
    params: LinearPMBaselineParams,
    pm_spectrum: HarmonicSpectrum,
    radius: float | None = None,
    rotor_sample_count: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rotor-angle sweep with PM phase flux linkage and phase back-EMF."""

    theta_mech = mechanical_angle_grid(params)
    rotor_electrical_angles = np.linspace(
        0.0, 2.0 * np.pi, rotor_sample_count, endpoint=False
    )
    flux_linkages = np.empty((rotor_sample_count, 3))

    for idx, rotor_electrical_angle in enumerate(rotor_electrical_angles):
        flux_linkages[idx, :] = phase_flux_linkages_from_pm(
            theta_mech=theta_mech,
            params=params,
            pm_spectrum=pm_spectrum,
            radius=radius,
            rotor_electrical_angle=float(rotor_electrical_angle),
        )

    emfs = -params.omega_elec_rad_s * np.gradient(
        flux_linkages,
        rotor_electrical_angles,
        axis=0,
        edge_order=2,
    )
    return rotor_electrical_angles, flux_linkages, emfs, theta_mech


def average_torque_maxwell(
    theta_mech: np.ndarray,
    Br: np.ndarray,
    Btheta: np.ndarray,
    radius: float,
    params: LinearPMBaselineParams,
) -> float:
    """Average electromagnetic torque from Maxwell stress in the air gap."""

    if Br.shape != Btheta.shape or Br.shape != theta_mech.shape:
        raise ValueError("theta_mech, Br, and Btheta must have the same shape")

    shear_stress_mean = float(np.mean(Br * Btheta) / params.mu_0)
    air_gap_surface = 2.0 * np.pi * radius * params.stack_length
    return shear_stress_mean * air_gap_surface * radius


def equivalent_pm_flux_linkage(
    torque_nm: float,
    params: LinearPMBaselineParams,
    electrical_angle: float | None = None,
) -> float:
    """Infer PM flux linkage from Maxwell torque using the PMSM torque law.

    This is intentionally a reduced-order estimate:

        T = 1.5 * p * lambda_pm * iq
    """

    _, iq = dq_currents(params, electrical_angle=electrical_angle)
    if np.isclose(iq, 0.0):
        return float("nan")
    return torque_nm / (1.5 * params.pole_pairs * iq)


def phase_back_emf_constants(
    lambda_pm_peak: float, params: LinearPMBaselineParams
) -> tuple[float, float]:
    """Return phase peak and RMS back-EMF for the configured speed."""

    phase_peak = params.omega_elec_rad_s * lambda_pm_peak
    phase_rms = phase_peak / np.sqrt(2.0)
    return float(phase_peak), float(phase_rms)


def phase_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def output_directory(params: LinearPMBaselineParams) -> Path:
    path = Path(params.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mechanical_degrees(theta_mech: np.ndarray) -> np.ndarray:
    return np.rad2deg(theta_mech)


def torque_angle_curve(
    theta_mech: np.ndarray,
    params: LinearPMBaselineParams,
    pm_spectrum: HarmonicSpectrum,
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
        Br, Btheta = total_air_gap_field(
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


def save_current_loading_png(
    output_dir: Path,
    theta_mech: np.ndarray,
    current_loading: np.ndarray,
    J1_slots: np.ndarray,
    J2_slots: np.ndarray,
) -> Path:
    theta_deg = mechanical_degrees(theta_mech)
    slot_ids = np.arange(1, len(J1_slots) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(theta_deg, current_loading, color="#155E75", linewidth=1.5)
    axes[0].set_title("Equivalent Slot Current Loading")
    axes[0].set_xlabel("Mechanical Angle [deg]")
    axes[0].set_ylabel("Kz [A/m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.0, 360.0)

    width = 0.38
    axes[1].bar(slot_ids - width / 2, J1_slots, width=width, label="J1", color="#DC2626")
    axes[1].bar(slot_ids + width / 2, J2_slots, width=width, label="J2", color="#2563EB")
    axes[1].set_title("Slot Current Densities")
    axes[1].set_xlabel("Slot Index")
    axes[1].set_ylabel("Current Density [A/m^2]")
    axes[1].set_xticks(slot_ids)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    path = output_dir / "current_loading.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_air_gap_fields_png(
    output_dir: Path,
    theta_mech: np.ndarray,
    Br_pm: np.ndarray,
    Br_current: np.ndarray,
    Br_total: np.ndarray,
    Btheta_pm: np.ndarray,
    Btheta_current: np.ndarray,
    Btheta_total: np.ndarray,
) -> Path:
    theta_deg = mechanical_degrees(theta_mech)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(theta_deg, Br_pm, label="PM only", color="#1D4ED8", linewidth=1.2)
    axes[0].plot(theta_deg, Br_current, label="Current only", color="#D97706", linewidth=1.2)
    axes[0].plot(theta_deg, Br_total, label="Total", color="#111827", linewidth=1.5)
    axes[0].set_title("Air-Gap Radial Flux Density")
    axes[0].set_xlabel("Mechanical Angle [deg]")
    axes[0].set_ylabel("Br [T]")
    axes[0].set_xlim(0.0, 360.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(theta_deg, Btheta_pm, label="PM only", color="#1D4ED8", linewidth=1.2)
    axes[1].plot(theta_deg, Btheta_current, label="Current only", color="#D97706", linewidth=1.2)
    axes[1].plot(theta_deg, Btheta_total, label="Total", color="#111827", linewidth=1.5)
    axes[1].set_title("Air-Gap Tangential Flux Density")
    axes[1].set_xlabel("Mechanical Angle [deg]")
    axes[1].set_ylabel("Btheta [T]")
    axes[1].set_xlim(0.0, 360.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    path = output_dir / "air_gap_fields.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_harmonics_png(
    output_dir: Path,
    pm_spectrum: HarmonicSpectrum,
    current_spectrum: CurrentLoadingSpectrum,
    pm_count: int = 12,
    current_count: int = 12,
) -> Path:
    pm_idx = np.argsort(np.abs(pm_spectrum.amplitudes))[::-1][:pm_count]
    current_mag = np.sqrt(current_spectrum.cosine**2 + current_spectrum.sine**2)
    current_idx = np.argsort(current_mag)[::-1][:current_count]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].bar(
        pm_spectrum.electrical_orders[pm_idx],
        pm_spectrum.amplitudes[pm_idx],
        color="#7C3AED",
    )
    axes[0].set_title("Dominant PM Magnetization Harmonics")
    axes[0].set_xlabel("Electrical Harmonic Order")
    axes[0].set_ylabel("Mr Amplitude [A/m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(
        current_spectrum.orders[current_idx],
        current_mag[current_idx],
        color="#059669",
    )
    axes[1].set_title("Dominant Slot-Current Loading Harmonics")
    axes[1].set_xlabel("Mechanical Harmonic Order")
    axes[1].set_ylabel("|K| [A/m]")
    axes[1].grid(True, alpha=0.3)

    path = output_dir / "harmonics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_torque_angle_png(
    output_dir: Path,
    electrical_angles_deg: np.ndarray,
    torques: np.ndarray,
    operating_angle_deg: float,
    operating_torque_nm: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(electrical_angles_deg, torques, color="#991B1B", linewidth=1.7)
    ax.scatter(
        [operating_angle_deg],
        [operating_torque_nm],
        color="#111827",
        s=45,
        zorder=3,
        label="Operating point",
    )
    ax.set_title("Average Torque vs Electrical Torque Angle")
    ax.set_xlabel("Electrical Torque Angle [deg]")
    ax.set_ylabel("Torque [Nm]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = output_dir / "torque_vs_angle.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def rms_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    diff = reference - estimate
    return float(np.sqrt(np.mean(diff * diff)))


def convergence_rows(
    theta_mech: np.ndarray,
    params: LinearPMBaselineParams,
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


def main() -> None:
    params = LinearPMBaselineParams()
    theta = mechanical_angle_grid(params)
    output_dir = output_directory(params)

    spectrum = radial_pm_harmonics(params)
    ia, ib, ic = balanced_phase_currents(params)
    exact_Mr = exact_radial_magnetization(theta, params)
    recon_Mr = reconstruct_radial_magnetization(theta, params, spectrum)
    K_slot, J1_slots, J2_slots = slot_current_loading_waveform(theta, params)
    current_spectrum = slot_current_loading_harmonics(theta, K_slot, params)
    Br_pm_gap, Btheta_pm_gap = equivalent_air_gap_field(
        theta_mech=theta,
        radius=params.mid_gap_radius,
        params=params,
        spectrum=spectrum,
    )
    Br_cs_gap, Btheta_cs_gap = slot_current_air_gap_field(
        theta_mech=theta,
        radius=params.mid_gap_radius,
        params=params,
        current_spectrum=current_spectrum,
    )
    Br_gap, Btheta_gap = total_air_gap_field(
        theta_mech=theta,
        radius=params.mid_gap_radius,
        params=params,
        pm_spectrum=spectrum,
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
        Br=Br_cs_gap,
        Btheta=Btheta_cs_gap,
        radius=params.mid_gap_radius,
        params=params,
    )
    torque_total = average_torque_maxwell(
        theta_mech=theta,
        Br=Br_gap,
        Btheta=Btheta_gap,
        radius=params.mid_gap_radius,
        params=params,
    )
    torque_interaction = torque_total - torque_pm - torque_current
    id_current, iq_current = dq_currents(params)
    lambda_pm_equiv = equivalent_pm_flux_linkage(torque_interaction, params)
    emf_phase_peak, emf_phase_rms = phase_back_emf_constants(lambda_pm_equiv, params)
    mech_power = torque_total * params.omega_mech_rad_s
    torque_angles_deg, torque_curve_nm = torque_angle_curve(theta, params, spectrum)

    current_loading_png = save_current_loading_png(
        output_dir=output_dir,
        theta_mech=theta,
        current_loading=K_slot,
        J1_slots=J1_slots,
        J2_slots=J2_slots,
    )
    air_gap_png = save_air_gap_fields_png(
        output_dir=output_dir,
        theta_mech=theta,
        Br_pm=Br_pm_gap,
        Br_current=Br_cs_gap,
        Br_total=Br_gap,
        Btheta_pm=Btheta_pm_gap,
        Btheta_current=Btheta_cs_gap,
        Btheta_total=Btheta_gap,
    )
    harmonics_png = save_harmonics_png(
        output_dir=output_dir,
        pm_spectrum=spectrum,
        current_spectrum=current_spectrum,
    )
    torque_angle_png = save_torque_angle_png(
        output_dir=output_dir,
        electrical_angles_deg=torque_angles_deg,
        torques=torque_curve_nm,
        operating_angle_deg=np.rad2deg(params.torque_angle_elec),
        operating_torque_nm=torque_total,
    )

    print("Linear PM baseline")
    print(f"pole pairs               : {params.pole_pairs}")
    print(f"stator slots             : {params.stator_slots}")
    print(f"magnet arc ratio         : {params.magnet_arc_ratio:.3f}")
    print(f"M0 [A/m]                 : {params.M0:.3f}")
    print(f"air-gap mid radius [m]   : {params.mid_gap_radius:.6f}")
    print(f"odd harmonics kept       : {len(spectrum.electrical_orders)}")
    print()

    print("Dominant PM harmonics")
    for nu, k, amp in dominant_harmonic_rows(spectrum):
        print(f"nu={nu:2d}  mech_order={k:3d}  Mr_amp={amp:12.3f} A/m")
    print()

    print("Magnetization convergence")
    for limit, err in convergence_rows(theta, params, spectrum):
        print(f"up to nu={limit:2d}  rms_error={err:12.3f} A/m")
    print()

    print("Waveform checks")
    print(f"exact Mr peak [A/m]      : {np.max(np.abs(exact_Mr)):12.3f}")
    print(f"recon Mr peak [A/m]      : {np.max(np.abs(recon_Mr)):12.3f}")
    print()

    print("Slot-current load model")
    print(f"torque angle [deg el.]   : {np.rad2deg(params.torque_angle_elec):12.3f}")
    print(f"phase currents [A]       : ia={ia:8.3f}  ib={ib:8.3f}  ic={ic:8.3f}")
    print(f"turns per coil in slot   : {params.turns_per_coil_in_slot:12d}")
    print(f"slot area S [m^2]        : {params.slot_area_region_v:12.6e}")
    print(f"coil width d [rad]       : {params.coil_width_mech:12.6f}")
    print(f"slot pitch delta [rad]   : {params.slot_pitch_mech:12.6f}")
    print(f"max |J1| [A/m^2]         : {np.max(np.abs(J1_slots)):12.6f}")
    print(f"max |J2| [A/m^2]         : {np.max(np.abs(J2_slots)):12.6f}")
    print(f"max |Kslot| [A/m]        : {np.max(np.abs(K_slot)):12.6f}")
    print()

    print("Dominant current-loading harmonics")
    for order, cosine, sine, magnitude in dominant_current_loading_rows(current_spectrum):
        print(
            f"n={order:2d}  Kc={cosine:12.3f}  Ks={sine:12.3f}  |K|={magnitude:12.3f}"
        )
    print()

    print("Mid-gap field peaks")
    print(f"PM-only Br peak [T]      : {np.max(np.abs(Br_pm_gap)):12.6f}")
    print(f"PM-only Btheta peak [T]  : {np.max(np.abs(Btheta_pm_gap)):12.6f}")
    print(f"I-only Br peak [T]       : {np.max(np.abs(Br_cs_gap)):12.6f}")
    print(f"I-only Btheta peak [T]   : {np.max(np.abs(Btheta_cs_gap)):12.6f}")
    print(f"Total Br peak [T]        : {np.max(np.abs(Br_gap)):12.6f}")
    print(f"Total Btheta peak [T]    : {np.max(np.abs(Btheta_gap)):12.6f}")
    print()

    print("Average torque from Maxwell stress")
    print(f"PM-only torque [Nm]      : {torque_pm:12.6f}")
    print(f"I-only torque [Nm]       : {torque_current:12.6f}")
    print(f"Total torque [Nm]        : {torque_total:12.6f}")
    print(f"Interaction torque [Nm]  : {torque_interaction:12.6f}")
    print()

    print("Equivalent dq and back-EMF")
    print(f"id [A peak]              : {id_current:12.6f}")
    print(f"iq [A peak]              : {iq_current:12.6f}")
    print(f"lambda_pm [Wb-turn]      : {lambda_pm_equiv:12.6f}")
    print(f"speed [rpm mech]         : {params.mechanical_speed_rpm:12.3f}")
    print(f"freq [Hz elec]           : {params.electrical_frequency_hz:12.3f}")
    print(f"phase back-EMF pk [V]    : {emf_phase_peak:12.6f}")
    print(f"phase back-EMF rms [V]   : {emf_phase_rms:12.6f}")
    print(f"mech power [W]           : {mech_power:12.6f}")
    print()

    print("PNG outputs")
    print(f"output dir               : {output_dir.resolve()}")
    print(f"current loading          : {current_loading_png.resolve()}")
    print(f"air-gap fields           : {air_gap_png.resolve()}")
    print(f"harmonics                : {harmonics_png.resolve()}")
    print(f"torque vs angle          : {torque_angle_png.resolve()}")


if __name__ == "__main__":
    main()

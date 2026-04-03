from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearPMModelParams:
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
    magnet_tangential_segments: int = 1
    magnet_radial_layers: int = 1
    initial_rotor_angle_mech: float = 0.0
    Br_60C: float = 1.247
    mu_rec: float = 1.039
    mu_0: float = 4.0 * np.pi * 1.0e-7

    core_mu_r_linear: float = 1000.0

    max_odd_harmonic: int = 45
    max_current_harmonic: int = 90
    sample_count: int = 4096
    pm_only_coupled_max_harmonic: int = 45
    pm_only_cogging_point_count: int = 19
    region_iii_subsections: int = 1
    region_iv_subsections: int = 6
    region_v_subsections: int = 6
    max_slot_opening_length: float = 10.0e-3
    slot_region_mu_r_linear: float = 1.0

    phase_current_peak: float = 50.0
    turns_per_phase: int = 100
    turns_per_coil_in_slot: int = 14
    winding_factor_fundamental: float = 0.95
    torque_angle_elec: float = 0.0
    stator_q_axis_offset_elec: float = 0.5 * np.pi
    mechanical_speed_rpm: float = 1400.0
    output_dir: str = "outputs/linear_model"

    @property
    def M0(self) -> float:
        return self.Br_60C / self.mu_0

    @property
    def pole_count(self) -> int:
        return 2 * self.pole_pairs

    @property
    def magnet_air_gap_arc_mech(self) -> float:
        return self.pole_pitch_mech - self.magnet_arc_mech

    @property
    def magnet_mu_r_linear(self) -> float:
        return self.mu_rec

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
    def slot_opening_angle_mech(self) -> float:
        return self.max_slot_opening_length / self.slot_opening_radius

    @property
    def slot_body_angle_mech(self) -> float:
        return self.coil_width_mech

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


@dataclass(frozen=True)
class RegionPermeabilityDescriptor:
    region_name: str
    theta_i: np.ndarray
    beta_sl: float
    beta_sub: float
    subsection_mu: np.ndarray
    subsection_count: int


@dataclass(frozen=True)
class SubdomainScaffold:
    region_names: tuple[str, ...]
    interface_radii: tuple[float, ...]
    unknowns_per_harmonic: int
    boundary_condition_count: int
    missing_equations: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BaselineSolution:
    params: LinearPMModelParams
    theta_mech: np.ndarray
    pm_spectrum: HarmonicSpectrum
    current_spectrum: CurrentLoadingSpectrum
    exact_Mr: np.ndarray
    recon_Mr: np.ndarray
    phase_currents: tuple[float, float, float]
    J1_slots: np.ndarray
    J2_slots: np.ndarray
    current_loading: np.ndarray
    Br_pm_gap: np.ndarray
    Btheta_pm_gap: np.ndarray
    Br_current_gap: np.ndarray
    Btheta_current_gap: np.ndarray
    Br_total_gap: np.ndarray
    Btheta_total_gap: np.ndarray
    torque_pm: float
    torque_current: float
    torque_total: float
    torque_interaction: float
    id_current: float
    iq_current: float
    lambda_pm_equiv: float
    emf_phase_peak: float
    emf_phase_rms: float
    mech_power: float


@dataclass(frozen=True)
class PMOnly5RegionSolution:
    params: LinearPMModelParams
    theta_mech: np.ndarray
    harmonic_orders: np.ndarray
    eval_radius: float
    Br_gap: np.ndarray
    Btheta_gap: np.ndarray
    rotor_angles_mech: np.ndarray
    cogging_torque_nm: np.ndarray
    system_size: int
    solve_residual: float


@dataclass(frozen=True)
class CoupledRegionOperators:
    region_name: str
    inner_radius: float
    outer_radius: float
    mu_c_r: np.ndarray
    mu_c_r_inv: np.ndarray
    mu_c_theta: np.ndarray
    mu_c_theta_inv: np.ndarray
    V: np.ndarray
    U: np.ndarray
    W: np.ndarray
    W_inv: np.ndarray
    lambda_values: np.ndarray

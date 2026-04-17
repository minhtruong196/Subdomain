from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    """Rotor PM segmentation and basic machine topology."""

    slots: int = 72
    poles: int = 12
    Nz: int = 16

    alpha_p: float = 0.9                               
    h_mm: float = 20
    hp_mm: float = 83
    edge_radius_mode: str = "profile"
    edge_pm_side_length_mm: float | None = None

    @property
    def pole_pairs(self) -> int:
        return self.poles // 2

    @property
    def zeta0_rad(self) -> float:
        return self.alpha_p * np.pi / (2.0 * self.pole_pairs * (self.Nz - 1))

    @property
    def upper_arc_half_angle_rad(self) -> float:
        return self.alpha_p * np.pi / (2.0 * self.pole_pairs)


@dataclass(frozen=True)
class StatorConfig:
    """Stator dimensions shared by boundary, air-gap, and performance code."""

    stator_inner_radius_m: float = 90 * 1e-3
    slot_depth_m: float = 33.0e-3
    slot_width_m: float = 4.2e-3
    airgap_length_m: float = 2 * 1e-3
    # None means use the middle of the physical air gap:
    # 0.5 * (stator inner radius + rotor outer radius).
    airgap_radius_m: float | None = None

    @property
    def R_s_m(self) -> float:
        return self.stator_inner_radius_m

    @property
    def R_sa_m(self) -> float:
        return self.stator_inner_radius_m + self.slot_depth_m

    @property
    def R_sc_m(self) -> float:
        return self.stator_inner_radius_m + 0.5 * self.slot_depth_m

    @property
    def b_s_rad(self) -> float:
        return self.slot_width_m / self.stator_inner_radius_m


@dataclass(frozen=True)
class MagnetConfig:
    Br_T: float = 1.065
    mu0: float = 4.0 * np.pi * 1.0e-7
    mu_r: float = 1.0
    magnetization_model: str = "radial"         #radial or parallel

    @property
    def M0_A_per_m(self) -> float:
        return self.Br_T / self.mu0


@dataclass(frozen=True)
class SolverConfig:
    max_pole_harmonic: int = 200
    slot_harmonics: int = 10
    magnetization_sample_count: int = 4000
    airgap_sample_count: int = 721


@dataclass(frozen=True)
class CurrentConfig:
    """Slot current density inputs. Defaults are the no-load case."""

    Jui_A_per_m2: float = 0.0
    Jdi_A_per_m2: float = 0.0


@dataclass(frozen=True)
class OperatingConfig:
    active_axial_length_m: float = 88.0e-3
    rated_speed_rpm: float = 900.0
    series_turns_per_phase: int = 144       #conductors per slot = series_turns_per_phase*2/(slot/3)
    coil_pitch_slots: int = 5               #short pitch
    torque_position_count: int = 121
    torque_theta_count: int = 1440
    emf_sample_count: int = 361

    @property
    def omega_mech_rad_s(self) -> float:
        return 2.0 * np.pi * self.rated_speed_rpm / 60.0


@dataclass(frozen=True)
class OutputConfig:
    geometry_dir: str = "outputs/subdomain_geometry"
    magnetization_dir: str = "outputs/subdomain_magnetization"
    boundary_dir: str = "outputs/subdomain_boundary"
    airgap_dir: str = "outputs/subdomain_airgap"
    performance_dir: str = "outputs/subdomain_performance"


@dataclass(frozen=True)
class MachineConfig:
    """Single configuration object used by the refactored subdomain modules."""

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    stator: StatorConfig = field(default_factory=StatorConfig)
    magnet: MagnetConfig = field(default_factory=MagnetConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    current: CurrentConfig = field(default_factory=CurrentConfig)
    operating: OperatingConfig = field(default_factory=OperatingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @property
    def pole_pairs(self) -> int:
        return self.geometry.pole_pairs

    @property
    def c_periods(self) -> int:
        return self.geometry.pole_pairs

    @property
    def reduced_slot_count(self) -> int:
        return self.geometry.slots // (2 * self.c_periods)

    @property
    def rotor_outer_radius_m(self) -> float:
        return self.stator.R_s_m - self.stator.airgap_length_m

    @property
    def rotor_outer_radius_mm(self) -> float:
        return self.rotor_outer_radius_m * 1.0e3

    @property
    def Rp_mm(self) -> float:
        return self.rotor_outer_radius_mm - self.geometry.h_mm

    @property
    def magnet_depth_mm(self) -> float:
        return self.rotor_outer_radius_mm - self.geometry.hp_mm

    @property
    def airgap_length_m(self) -> float:
        return self.stator.airgap_length_m

    @property
    def airgap_radius_m(self) -> float:
        if self.stator.airgap_radius_m is not None:
            return self.stator.airgap_radius_m
        return 0.5 * (self.stator.R_s_m + self.rotor_outer_radius_m)

    def validate_dimensions(self) -> None:
        if self.geometry.poles % 2:
            raise ValueError("poles must be even")
        if self.magnet_depth_mm <= 0.0:
            raise ValueError("magnet depth must be positive: Ror - hp > 0")
        if self.Rp_mm <= 0.0:
            raise ValueError("Rp must be positive: Ror - h > 0")
        if self.airgap_length_m <= 0.0:
            raise ValueError("air-gap length must be positive: stator_inner_radius > rotor_outer_radius")
        if self.geometry.edge_radius_mode not in {"profile", "side_length"}:
            raise ValueError("edge_radius_mode must be 'profile' or 'side_length'")

    def with_stator(self, **changes: float) -> MachineConfig:
        return replace(self, stator=replace(self.stator, **changes))

    def with_geometry(self, **changes: float | int | str | None) -> MachineConfig:
        return replace(self, geometry=replace(self.geometry, **changes))

    def with_magnet(self, **changes: float | str) -> MachineConfig:
        return replace(self, magnet=replace(self.magnet, **changes))

    def with_solver(self, **changes: int) -> MachineConfig:
        return replace(self, solver=replace(self.solver, **changes))

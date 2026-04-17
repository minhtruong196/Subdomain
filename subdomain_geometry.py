from __future__ import annotations

from pathlib import Path

import numpy as np

from subdomain_config import GeometryConfig, MachineConfig


def zeta_j_rad(j: int, geometry: GeometryConfig) -> float:
    if not 1 <= j <= geometry.Nz - 1:
        raise ValueError(f"regular segment j must be in [1, {geometry.Nz - 1}]")
    return (2.0 * j - 1.0) * geometry.zeta0_rad / 2.0


def upper_radius_regular_mm(zeta_rad: float, config: MachineConfig) -> float:
    radicand = config.Rp_mm**2 - (config.geometry.h_mm * np.sin(zeta_rad)) ** 2
    if radicand < 0.0:
        raise ValueError("upper radius has no real solution for this zeta")
    return float(config.geometry.h_mm * np.cos(zeta_rad) + np.sqrt(radicand))


def lower_radius_regular_mm(zeta_rad: float, config: MachineConfig) -> float:
    return float(config.geometry.hp_mm / np.cos(zeta_rad))


def upper_radius_edge_mm(edge_zeta_rad: float, config: MachineConfig) -> float:
    if config.geometry.edge_radius_mode == "profile":
        return upper_radius_regular_mm(edge_zeta_rad, config)
    if config.geometry.edge_radius_mode == "side_length":
        if config.geometry.edge_pm_side_length_mm is None:
            raise ValueError("edge_pm_side_length_mm is required when edge_radius_mode='side_length'")
        return lower_radius_regular_mm(edge_zeta_rad, config) + 0.5 * config.geometry.edge_pm_side_length_mm
    raise ValueError("edge_radius_mode must be 'profile' or 'side_length'")


def segment_radii_mm(config: MachineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = config.geometry
    orders = np.arange(1, geometry.Nz + 1, dtype=int)
    zeta = np.empty(geometry.Nz, dtype=float)
    Ru = np.empty(geometry.Nz, dtype=float)
    Rl = np.empty(geometry.Nz, dtype=float)

    for idx, j in enumerate(orders[:-1]):
        zeta[idx] = zeta_j_rad(int(j), geometry)
        Ru[idx] = upper_radius_regular_mm(zeta[idx], config)
        Rl[idx] = lower_radius_regular_mm(zeta[idx], config)

    edge_zeta = geometry.upper_arc_half_angle_rad
    zeta[-1] = edge_zeta
    Rl[-1] = lower_radius_regular_mm(edge_zeta, config)
    Ru[-1] = upper_radius_edge_mm(edge_zeta, config)

    return orders, Ru, Rl


def segment_radii_m(config: MachineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders, Ru_mm, Rl_mm = segment_radii_mm(config)
    return orders, Ru_mm * 1.0e-3, Rl_mm * 1.0e-3


def selected_segment_radii_m(config: MachineConfig, segment_j: int) -> tuple[float, float]:
    _, Ru_m, Rl_m = segment_radii_m(config)
    idx = segment_j - 1
    if idx < 0 or idx >= len(Ru_m):
        raise ValueError(f"segment_j must be in [1, {len(Ru_m)}]")
    return float(Ru_m[idx]), float(Rl_m[idx])


def validate_rotor_geometry(config: MachineConfig) -> None:
    _, Ru_m, Rl_m = segment_radii_m(config)
    if np.any(Ru_m <= Rl_m):
        bad = int(np.where(Ru_m <= Rl_m)[0][0]) + 1
        raise ValueError(
            f"invalid PM segment geometry at j={bad}: "
            f"Ru={Ru_m[bad - 1] * 1.0e3:.6f} mm <= Rl={Rl_m[bad - 1] * 1.0e3:.6f} mm"
        )

    max_ru = float(np.max(Ru_m))
    if max_ru >= config.stator.R_s_m:
        raise ValueError(
            f"rotor/PM outer radius reaches stator bore: max Ru={max_ru * 1.0e3:.6f} mm, "
            f"Ris={config.stator.R_s_m * 1.0e3:.6f} mm"
        )

    if not (max_ru < config.airgap_radius_m < config.stator.R_s_m):
        raise ValueError(
            f"air-gap sampling radius must be between max Ru and Ris: "
            f"max Ru={max_ru * 1.0e3:.6f} mm, Rg={config.airgap_radius_m * 1.0e3:.6f} mm, "
            f"Ris={config.stator.R_s_m * 1.0e3:.6f} mm"
        )


def theta_i(slot_number: int, config: MachineConfig) -> float:
    return float(np.pi / config.geometry.slots * (2 * slot_number - 1))


def save_geometry_csv(config: MachineConfig, path: Path) -> Path:
    orders, Ru_mm, Rl_mm = segment_radii_mm(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([orders, Ru_mm, Rl_mm])
    np.savetxt(path, data, delimiter=",", header="segment_j,Ru_mm,Rl_mm", comments="")
    return path

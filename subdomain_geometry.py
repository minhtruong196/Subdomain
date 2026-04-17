from __future__ import annotations

from pathlib import Path

import numpy as np

from subdomain_config import GeometryConfig, MachineConfig


def zeta_j_rad(j: int, geometry: GeometryConfig) -> float:
    if not 1 <= j <= geometry.Nz - 1:
        raise ValueError(f"regular segment j must be in [1, {geometry.Nz - 1}]")
    return (2.0 * j - 1.0) * geometry.zeta0_rad / 2.0


def upper_radius_regular_mm(zeta_rad: float, geometry: GeometryConfig) -> float:
    radicand = geometry.Rp_mm**2 - (geometry.h_mm * np.sin(zeta_rad)) ** 2
    if radicand < 0.0:
        raise ValueError("upper radius has no real solution for this zeta")
    return float(geometry.h_mm * np.cos(zeta_rad) + np.sqrt(radicand))


def lower_radius_regular_mm(zeta_rad: float, geometry: GeometryConfig) -> float:
    return float(geometry.hp_mm / np.cos(zeta_rad))


def segment_radii_mm(geometry: GeometryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders = np.arange(1, geometry.Nz + 1, dtype=int)
    zeta = np.empty(geometry.Nz, dtype=float)
    Ru = np.empty(geometry.Nz, dtype=float)
    Rl = np.empty(geometry.Nz, dtype=float)

    for idx, j in enumerate(orders[:-1]):
        zeta[idx] = zeta_j_rad(int(j), geometry)
        Ru[idx] = upper_radius_regular_mm(zeta[idx], geometry)
        Rl[idx] = lower_radius_regular_mm(zeta[idx], geometry)

    edge_zeta = geometry.upper_arc_half_angle_rad
    zeta[-1] = edge_zeta
    Rl[-1] = lower_radius_regular_mm(edge_zeta, geometry)
    Ru[-1] = Rl[-1] + 0.5 * geometry.pm_side_length_mm

    return orders, Ru, Rl


def segment_radii_m(geometry: GeometryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders, Ru_mm, Rl_mm = segment_radii_mm(geometry)
    return orders, Ru_mm * 1.0e-3, Rl_mm * 1.0e-3


def selected_segment_radii_m(config: MachineConfig, segment_j: int) -> tuple[float, float]:
    _, Ru_m, Rl_m = segment_radii_m(config.geometry)
    idx = segment_j - 1
    if idx < 0 or idx >= len(Ru_m):
        raise ValueError(f"segment_j must be in [1, {len(Ru_m)}]")
    return float(Ru_m[idx]), float(Rl_m[idx])


def theta_i(slot_number: int, config: MachineConfig) -> float:
    return float(np.pi / config.geometry.slots * (2 * slot_number - 1))


def save_geometry_csv(config: MachineConfig, path: Path) -> Path:
    orders, Ru_mm, Rl_mm = segment_radii_mm(config.geometry)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([orders, Ru_mm, Rl_mm])
    np.savetxt(path, data, delimiter=",", header="segment_j,Ru_mm,Rl_mm", comments="")
    return path

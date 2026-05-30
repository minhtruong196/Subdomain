from __future__ import annotations

import numpy as np

from subdomain_config import MachineConfig


def pole_harmonic_orders(config: MachineConfig) -> np.ndarray:
    """Return nu = mc / p = 1, 3, 5, ... up to the configured cutoff."""

    return np.arange(1, config.solver.max_pole_harmonic + 1, 2, dtype=int)


def mechanical_harmonic_orders(config: MachineConfig) -> np.ndarray:
    """Return mc from Eq. (9). For this paper c = p, so mc = nu * p."""

    nu = pole_harmonic_orders(config)
    return nu * config.geometry.pole_pairs


def ka_kb(mc: np.ndarray, j: int, zeta0: float) -> tuple[np.ndarray, np.ndarray]:
    a0 = (j - 1) * zeta0
    a1 = j * zeta0

    ka = (np.sin((mc + 1.0) * a1) - np.sin((mc + 1.0) * a0)) / (mc + 1.0)
    kb = np.empty_like(ka)

    singular = np.isclose(mc, 1.0)
    kb[~singular] = (
        np.sin((mc[~singular] - 1.0) * a1)
        - np.sin((mc[~singular] - 1.0) * a0)
    ) / (mc[~singular] - 1.0)
    kb[singular] = a1 - a0

    return ka, kb


def magnetization_coefficients(
    segment_j: int,
    config: MachineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= segment_j <= config.geometry.Nz:
        raise ValueError(f"segment_j must be in [1, {config.geometry.Nz}]")

    mc = mechanical_harmonic_orders(config).astype(float)
    zeta0 = config.geometry.zeta0_rad
    Br_T = config.magnet.Br_T
    mu0 = config.magnet.mu0

    if config.magnet.magnetization_model == "radial":
        nu = pole_harmonic_orders(config).astype(float)
        Mr_m = (
            4.0
            * Br_T
            / (nu * np.pi * mu0)
            * (np.sin(mc * segment_j * zeta0) - np.sin(mc * (segment_j - 1) * zeta0))
        )
        Mt_m = np.zeros_like(Mr_m)
        return mc, Mr_m, Mt_m

    if config.magnet.magnetization_model != "parallel":
        raise ValueError("magnetization_model must be 'parallel' or 'radial'")

    factor = 2.0 * config.geometry.pole_pairs * Br_T / (np.pi * mu0)

    Mr_m = np.empty_like(mc)
    Mt_m = np.empty_like(mc)

    singular = np.isclose(mc, 1.0)
    if np.any(~singular):
        ka, kb = ka_kb(mc[~singular], segment_j, zeta0)
        Mr_m[~singular] = factor * (ka + kb)
        Mt_m[~singular] = factor * (ka - kb)

    if np.any(singular):
        a0 = (segment_j - 1) * zeta0
        a1 = segment_j * zeta0
        value = (
            config.geometry.pole_pairs
            * Br_T
            / (np.pi * mu0)
            * (np.sin(2.0 * a1) - np.sin(2.0 * a0))
        )
        Mr_m[singular] = value
        Mt_m[singular] = value

    return mc, Mr_m, Mt_m


def theta_grid_half_pole(config: MachineConfig) -> np.ndarray:
    return np.linspace(
        0.0,
        np.pi / (2.0 * config.geometry.pole_pairs),
        config.solver.magnetization_sample_count,
        endpoint=True,
    )


def reconstruct_mr(
    theta_rad: np.ndarray,
    mc: np.ndarray,
    Mr_m: np.ndarray,
    delta_rad: float = 0.0,
) -> np.ndarray:
    phase = np.outer(mc, theta_rad - delta_rad)
    return np.sum(Mr_m[:, None] * np.cos(phase), axis=0)


def exact_mr_pair(theta_rad: np.ndarray, segment_j: int, config: MachineConfig, delta_rad: float = 0.0) -> np.ndarray:
    pole_pitch = np.pi / config.geometry.pole_pairs
    half_pole = 0.5 * pole_pitch
    theta_local = np.mod(theta_rad - delta_rad, pole_pitch)
    theta_folded = np.minimum(theta_local, pole_pitch - theta_local)

    lower = (segment_j - 1) * config.geometry.zeta0_rad
    upper = segment_j * config.geometry.zeta0_rad
    inside = (lower <= theta_folded) & (theta_folded <= upper)

    Mr = np.zeros_like(theta_rad)
    Mr[inside] = config.magnet.M0_A_per_m * np.cos(theta_folded[inside])
    Mr[(theta_folded < 0.0) | (theta_folded > half_pole)] = 0.0
    return Mr

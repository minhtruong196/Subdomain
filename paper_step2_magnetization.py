from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from paper_step1_geometry import PaperGeometryParams

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MagnetizationParams:
    geometry: PaperGeometryParams = PaperGeometryParams()
    Br_T: float = 1.0
    mu0: float = 4.0 * np.pi * 1.0e-7
    max_pole_harmonic: int = 200
    delta_rad: float = 0.0
    sample_count: int = 4000
    output_dir: str = "outputs/step2_magnetization"

    @property
    def M0_A_per_m(self) -> float:
        return self.Br_T / self.mu0


def pole_harmonic_orders(params: MagnetizationParams) -> np.ndarray:
    """Return nu = mc / p = 1, 3, 5, ... up to the configured cutoff."""

    return np.arange(1, params.max_pole_harmonic + 1, 2, dtype=int)


def mechanical_harmonic_orders(params: MagnetizationParams) -> np.ndarray:
    """Return mc from Eq. (9). For this paper c = p, so mc = nu * p."""

    nu = pole_harmonic_orders(params)
    return nu * params.geometry.pole_pairs


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
    j: int,
    params: MagnetizationParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= j <= params.geometry.Nz:
        raise ValueError(f"j must be in [1, {params.geometry.Nz}]")

    mc = mechanical_harmonic_orders(params).astype(float)
    zeta0 = params.geometry.zeta0_rad
    factor = 2.0 * params.geometry.pole_pairs * params.Br_T / (np.pi * params.mu0)

    Mr_m = np.empty_like(mc)
    Mt_m = np.empty_like(mc)

    singular = np.isclose(mc, 1.0)
    if np.any(~singular):
        ka, kb = ka_kb(mc[~singular], j, zeta0)
        Mr_m[~singular] = factor * (ka + kb)
        Mt_m[~singular] = factor * (ka - kb)

    if np.any(singular):
        a0 = (j - 1) * zeta0
        a1 = j * zeta0
        value = (
            params.geometry.pole_pairs
            * params.Br_T
            / (np.pi * params.mu0)
            * (np.sin(2.0 * a1) - np.sin(2.0 * a0))
        )
        Mr_m[singular] = value
        Mt_m[singular] = value

    return mc, Mr_m, Mt_m


def theta_grid_half_pole(params: MagnetizationParams) -> np.ndarray:
    return np.linspace(
        0.0,
        np.pi / (2.0 * params.geometry.pole_pairs),
        params.sample_count,
        endpoint=True,
    )


def reconstruct_mr(
    theta_rad: np.ndarray,
    mc: np.ndarray,
    Mr_m: np.ndarray,
    params: MagnetizationParams,
) -> np.ndarray:
    phase = np.outer(mc, theta_rad - params.delta_rad)
    return np.sum(Mr_m[:, None] * np.cos(phase), axis=0)


def exact_mr_pair(theta_rad: np.ndarray, j: int, params: MagnetizationParams) -> np.ndarray:
    pole_pitch = np.pi / params.geometry.pole_pairs
    half_pole = 0.5 * pole_pitch
    theta_local = np.mod(theta_rad - params.delta_rad, pole_pitch)
    theta_folded = np.minimum(theta_local, pole_pitch - theta_local)

    lower = (j - 1) * params.geometry.zeta0_rad
    upper = j * params.geometry.zeta0_rad
    inside = (lower <= theta_folded) & (theta_folded <= upper)

    Mr = np.zeros_like(theta_rad)
    Mr[inside] = params.M0_A_per_m * np.cos(theta_folded[inside])
    Mr[(theta_folded < 0.0) | (theta_folded > half_pole)] = 0.0
    return Mr


def save_mr_plot(
    theta_rad: np.ndarray,
    exact: np.ndarray,
    reconstructed: np.ndarray,
    j: int,
    params: MagnetizationParams,
) -> Path:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.rad2deg(theta_rad)
    exact_norm = exact / params.M0_A_per_m
    recon_norm = reconstructed / params.M0_A_per_m

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.plot(theta_deg, exact_norm, color="#111827", linewidth=2.0, label="Exact segment")
    ax.plot(theta_deg, recon_norm, color="#0072BD", linewidth=1.4, label="Fourier m <= 200")
    ax.set_xlabel("Mechanical angle theta (deg)", fontsize=14)
    ax.set_ylabel(r"$M_{r(j)} / (B_r / \mu_0)$", fontsize=14)
    ax.set_title(f"Radial magnetization waveform in one half-pole, j = {j}", fontsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(theta_deg[0], theta_deg[-1])
    ax.set_ylim(-0.25, 1.25)

    path = output_dir / f"mr_waveform_j{j}_m{params.max_pole_harmonic}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    params = MagnetizationParams()
    j = 1

    mc, Mr_m, Mt_m = magnetization_coefficients(j, params)
    theta = theta_grid_half_pole(params)
    Mr_recon = reconstruct_mr(theta, mc, Mr_m, params)
    Mr_exact = exact_mr_pair(theta, j, params)
    plot_path = save_mr_plot(theta, Mr_exact, Mr_recon, j, params)

    print("Step 2 magnetization vector")
    print(f"j                         : {j}")
    print(f"slots, poles              : {params.geometry.slots}, {params.geometry.poles}")
    print(f"p = c                     : {params.geometry.pole_pairs}")
    print(f"max pole harmonic nu      : {params.max_pole_harmonic}")
    print(f"kept harmonic count       : {len(mc)}")
    print(f"first mc values           : {mc[:8].astype(int).tolist()}")
    print(f"M0 = Br/mu0 [A/m]         : {params.M0_A_per_m:.6e}")
    print(f"max normalized Mr exact   : {np.max(Mr_exact / params.M0_A_per_m):.6f}")
    print(f"max normalized Mr Fourier : {np.max(Mr_recon / params.M0_A_per_m):.6f}")
    print()
    print("first coefficients normalized by M0")
    for order, mr, mt in zip(mc[:8].astype(int), Mr_m[:8] / params.M0_A_per_m, Mt_m[:8] / params.M0_A_per_m):
        print(f"mc={order:4d}  Mr_m/M0={mr: .8f}  Mtheta_m/M0={mt: .8f}")
    print()
    print(f"plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()

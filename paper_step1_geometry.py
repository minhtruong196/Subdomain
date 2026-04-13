from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PaperGeometryParams:
    """Geometry parameters used for Fig. 4(d) in the paper."""

    slots: int = 72
    poles: int = 12
    Nz: int = 16

    alpha_p: float = 0.922
    h_mm: float = 57.0
    Rp_mm: float = 29.8
    hp_mm: float = 77.05
    pm_side_length_mm: float = 2.42

    output_dir: str = "outputs/step1_geometry"

    @property
    def pole_pairs(self) -> int:
        return self.poles // 2

    @property
    def zeta0_rad(self) -> float:
        return self.alpha_p * np.pi / (2.0 * self.pole_pairs * (self.Nz - 1))

    @property
    def upper_arc_half_angle_rad(self) -> float:
        return self.alpha_p * np.pi / (2.0 * self.pole_pairs)


def zeta_j_rad(j: int, params: PaperGeometryParams) -> float:
    if not 1 <= j <= params.Nz - 1:
        raise ValueError(f"regular segment j must be in [1, {params.Nz - 1}]")
    return (2.0 * j - 1.0) * params.zeta0_rad / 2.0


def upper_radius_regular_mm(zeta_rad: float, params: PaperGeometryParams) -> float:
    radicand = params.Rp_mm**2 - (params.h_mm * np.sin(zeta_rad)) ** 2
    if radicand < 0.0:
        raise ValueError("upper radius has no real solution for this zeta")
    return float(params.h_mm * np.cos(zeta_rad) + np.sqrt(radicand))


def lower_radius_regular_mm(zeta_rad: float, params: PaperGeometryParams) -> float:
    return float(params.hp_mm / np.cos(zeta_rad))


def segment_radii_mm(params: PaperGeometryParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders = np.arange(1, params.Nz + 1, dtype=int)
    zeta = np.empty(params.Nz, dtype=float)
    Ru = np.empty(params.Nz, dtype=float)
    Rl = np.empty(params.Nz, dtype=float)

    for idx, j in enumerate(orders[:-1]):
        zeta[idx] = zeta_j_rad(int(j), params)
        Ru[idx] = upper_radius_regular_mm(zeta[idx], params)
        Rl[idx] = lower_radius_regular_mm(zeta[idx], params)

    edge_zeta = params.upper_arc_half_angle_rad
    zeta[-1] = edge_zeta
    Rl[-1] = lower_radius_regular_mm(edge_zeta, params)
    Ru[-1] = Rl[-1] + 0.5 * params.pm_side_length_mm

    return orders, Ru, Rl


def save_ru_rl_plot(
    orders: np.ndarray,
    Ru: np.ndarray,
    Rl: np.ndarray,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    ax.plot(orders, Ru, "o-", color="#0072BD", linewidth=2.2, markersize=8, label=r"$R_u$")
    ax.plot(orders, Rl, "o-", color="#D95319", linewidth=2.2, markersize=8, label=r"$R_l$")

    ax.set_xlim(0, 16)
    ax.set_ylim(76, 88)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_yticks([76, 78, 80, 82, 84, 86, 88])
    ax.set_xlabel("Order of segments", fontsize=16)
    ax.set_ylabel(r"$R$ (mm)", fontsize=16)
    ax.grid(True, color="#F2A6A6", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=16)
    ax.tick_params(labelsize=14)

    arrow = {"arrowstyle": "-|>", "color": "black", "linewidth": 1.1}
    ax.annotate(r"$R_{u(1)}$", xy=(orders[0], Ru[0]), xytext=(0.25, 85.3), fontsize=16, arrowprops=arrow)
    ax.annotate(r"$R_{l(1)}$", xy=(orders[0], Rl[0]), xytext=(0.2, 77.35), fontsize=16, arrowprops=arrow)
    ax.annotate(r"$R_{u(j)}$", xy=(12, Ru[11]), xytext=(9.25, 83.2), fontsize=17, arrowprops=arrow)
    ax.annotate(r"$R_{l(j)}$", xy=(12, Rl[11]), xytext=(9.35, 78.95), fontsize=17, arrowprops=arrow)
    ax.annotate(r"$R_{u(N_z)}$", xy=(orders[-1], Ru[-1]), xytext=(13.65, 80.8), fontsize=16, arrowprops=arrow)
    ax.annotate(r"$R_{l(N_z)}$", xy=(orders[-1], Rl[-1]), xytext=(13.85, 79.65), fontsize=16, arrowprops=arrow)

    path = output_dir / "fig4d_ru_rl_segments.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    params = PaperGeometryParams()
    orders, Ru, Rl = segment_radii_mm(params)
    plot_path = save_ru_rl_plot(orders, Ru, Rl, Path(params.output_dir))

    print("Step 1 geometry replication")
    print(f"slots={params.slots}, poles={params.poles}, pole_pairs={params.pole_pairs}, Nz={params.Nz}")
    print(f"zeta0 [rad] = {params.zeta0_rad:.10f}")
    print(f"upper arc half angle [rad] = {params.upper_arc_half_angle_rad:.10f}")
    print()
    print("j, zeta_deg, Ru_mm, Rl_mm")
    for j, ru, rl in zip(orders, Ru, Rl):
        angle = zeta_j_rad(int(j), params) if j < params.Nz else params.upper_arc_half_angle_rad
        print(f"{j:2d}, {np.rad2deg(angle):9.5f}, {ru:8.4f}, {rl:8.4f}")
    print()
    print(f"plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()

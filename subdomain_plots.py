from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from subdomain_boundary import UnknownLayout
from subdomain_config import MachineConfig
from subdomain_geometry import segment_radii_mm, zeta_j_rad


def save_geometry_plot(config: MachineConfig) -> Path:
    output_dir = Path(config.outputs.geometry_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orders, Ru, Rl = segment_radii_mm(config.geometry)
    fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    ax.plot(orders, Ru, "o-", color="#0072BD", linewidth=2.2, markersize=8, label=r"$R_u$")
    ax.plot(orders, Rl, "o-", color="#D95319", linewidth=2.2, markersize=8, label=r"$R_l$")

    ax.set_xlim(0, config.geometry.Nz)
    ax.set_ylim(float(np.floor(np.min(Rl))) - 1.0, float(np.ceil(np.max(Ru))) + 1.0)
    ax.set_xlabel("Order of segments", fontsize=16)
    ax.set_ylabel(r"$R$ (mm)", fontsize=16)
    ax.grid(True, color="#F2A6A6", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=16)
    ax.tick_params(labelsize=14)

    path = output_dir / "ru_rl_segments.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_geometry_table(config: MachineConfig) -> Path:
    output_dir = Path(config.outputs.geometry_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orders, Ru_mm, Rl_mm = segment_radii_mm(config.geometry)
    zeta_deg = [
        np.rad2deg(zeta_j_rad(int(j), config.geometry))
        if j < config.geometry.Nz
        else np.rad2deg(config.geometry.upper_arc_half_angle_rad)
        for j in orders
    ]
    path = output_dir / "segment_radii.csv"
    np.savetxt(
        path,
        np.column_stack([orders, zeta_deg, Ru_mm, Rl_mm]),
        delimiter=",",
        header="segment_j,zeta_deg,Ru_mm,Rl_mm",
        comments="",
    )
    return path


def save_magnetization_plot(
    theta_rad: np.ndarray,
    exact: np.ndarray,
    reconstructed: np.ndarray,
    segment_j: int,
    config: MachineConfig,
) -> Path:
    output_dir = Path(config.outputs.magnetization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.rad2deg(theta_rad)
    exact_norm = exact / config.magnet.M0_A_per_m
    recon_norm = reconstructed / config.magnet.M0_A_per_m

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.plot(theta_deg, exact_norm, color="#111827", linewidth=2.0, label="Exact segment")
    ax.plot(
        theta_deg,
        recon_norm,
        color="#0072BD",
        linewidth=1.4,
        label=f"Fourier nu <= {config.solver.max_pole_harmonic}",
    )
    ax.set_xlabel("Mechanical angle theta (deg)", fontsize=14)
    ax.set_ylabel(r"$M_{r(j)} / (B_r / \mu_0)$", fontsize=14)
    ax.set_title(f"Radial magnetization waveform in one half-pole, j = {segment_j}", fontsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(theta_deg[0], theta_deg[-1])
    ax.set_ylim(-0.25, 1.25)

    path = output_dir / f"mr_waveform_j{segment_j}_m{config.solver.max_pole_harmonic}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_boundary_solution(
    solution: np.ndarray,
    K: np.ndarray,
    Y: np.ndarray,
    layout: UnknownLayout,
    meta: dict[str, np.ndarray | float],
    residual: float,
    config: MachineConfig,
    segment_j: int,
) -> Path:
    output_dir = Path(config.outputs.boundary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"solution_j{segment_j}_m{config.solver.max_pole_harmonic}_n{config.solver.slot_harmonics}.npz"
    np.savez_compressed(
        path,
        solution=solution,
        K=K,
        Y=Y,
        residual=residual,
        total_unknowns=layout.total,
        m_count=layout.m_count,
        n_count=layout.n_count,
        slot_count=layout.slot_count,
        **meta,
    )
    return path


def save_flux_outputs(
    theta_elec_deg: np.ndarray,
    Br: np.ndarray,
    Btheta: np.ndarray,
    residuals: list[float],
    config: MachineConfig,
) -> tuple[Path, Path]:
    output_dir = Path(config.outputs.airgap_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = config.magnet.magnetization_model

    data_path = output_dir / f"no_load_flux_density_{model}_data.npz"
    np.savez_compressed(
        data_path,
        theta_elec_deg=theta_elec_deg,
        Br_T=Br,
        Btheta_T=Btheta,
        residuals=np.asarray(residuals),
        airgap_radius_m=config.stator.airgap_radius_m,
        magnetization_model=model,
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), constrained_layout=True)
    axes[0].plot(theta_elec_deg, Br, color="#E68613", linewidth=2.2, label=f"{model}: Br max={np.max(Br):.4f} T")
    axes[0].set_ylabel("Radial flux density (T)", fontsize=12)
    axes[0].set_xlim(-90, 90)
    axes[0].set_ylim(min(-0.05, float(np.min(Br)) * 1.1), max(1.0, float(np.max(Br)) * 1.1))
    axes[0].set_xticks([-90, -45, 0, 45, 90])
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=11, loc="upper right")

    axes[1].plot(
        theta_elec_deg,
        Btheta,
        color="#E68613",
        linewidth=2.2,
        label=f"{model}: Btheta max={np.max(np.abs(Btheta)):.4f} T",
    )
    axes[1].set_xlabel("Position (Elec.Deg.)", fontsize=12)
    axes[1].set_ylabel("Tangential flux density (T)", fontsize=12)
    axes[1].set_xlim(-90, 90)
    axes[1].set_ylim(min(-0.2, float(np.min(Btheta)) * 1.1), max(0.2, float(np.max(Btheta)) * 1.1))
    axes[1].set_xticks([-90, -45, 0, 45, 90])
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=11, loc="upper right")

    plot_path = output_dir / f"no_load_flux_density_{model}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path, data_path


def save_performance_plots(
    torque_deg: np.ndarray,
    torque_mNm: np.ndarray,
    emf_deg: np.ndarray,
    phase_emf_v: np.ndarray,
    line_line_emf_v: np.ndarray,
    flux_linkage: np.ndarray,
    config: MachineConfig,
) -> tuple[Path, Path, Path]:
    output_dir = Path(config.outputs.performance_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "performance_data.npz"
    np.savez_compressed(
        data_path,
        torque_rotor_pos_mech_deg=torque_deg,
        cogging_torque_mNm=torque_mNm,
        emf_position_elec_deg=emf_deg,
        no_load_phase_back_emf_v=phase_emf_v,
        no_load_line_line_back_emf_v=line_line_emf_v,
        phase_a_flux_linkage_wb=flux_linkage,
        airgap_radius_m=config.stator.airgap_radius_m,
        magnetization_model=config.magnet.magnetization_model,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(torque_deg, torque_mNm, color="#0072BD", linewidth=2.2, label="Subdomain")
    ax.set_xlabel("Rotor position (Mech.Deg.)")
    ax.set_ylabel("Cogging torque (mN.m)")
    ax.set_xlim(0, 30)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    torque_path = output_dir / "cogging_torque.png"
    fig.savefig(torque_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(emf_deg, line_line_emf_v, color="#D95319", linewidth=2.2, label="Line-line estimate")
    ax.plot(emf_deg, phase_emf_v, color="#7E7E7E", linewidth=1.5, linestyle="--", label="Phase-A checkpoint")
    ax.set_xlabel("Position (Elec.Deg.)")
    ax.set_ylabel("No-load back-EMF (V)")
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    emf_path = output_dir / "no_load_back_emf.png"
    fig.savefig(emf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return torque_path, emf_path, data_path

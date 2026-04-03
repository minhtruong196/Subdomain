from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np

from config import BaselineSolution, PMOnly5RegionSolution
from post_processing import torque_angle_curve
from solver import total_air_gap_field


def output_directory(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mechanical_degrees(theta_mech: np.ndarray) -> np.ndarray:
    return np.rad2deg(theta_mech)


def _polar_to_xy(radius: float, angle_deg: float) -> tuple[float, float]:
    angle_rad = np.deg2rad(angle_deg)
    return radius * np.cos(angle_rad), radius * np.sin(angle_rad)


def save_current_loading_png(output_dir: Path, solution: BaselineSolution) -> Path:
    theta_deg = mechanical_degrees(solution.theta_mech)
    slot_ids = np.arange(1, len(solution.J1_slots) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(theta_deg, solution.current_loading, color="#155E75", linewidth=1.5)
    axes[0].set_title("Equivalent Slot Current Loading")
    axes[0].set_xlabel("Mechanical Angle [deg]")
    axes[0].set_ylabel("Kz [A/m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.0, 360.0)

    width = 0.38
    axes[1].bar(
        slot_ids - width / 2, solution.J1_slots, width=width, label="J1", color="#DC2626"
    )
    axes[1].bar(
        slot_ids + width / 2, solution.J2_slots, width=width, label="J2", color="#2563EB"
    )
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


def save_air_gap_fields_png(output_dir: Path, solution: BaselineSolution) -> Path:
    theta_deg = mechanical_degrees(solution.theta_mech)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(theta_deg, solution.Br_pm_gap, label="PM only", color="#1D4ED8", linewidth=1.2)
    axes[0].plot(
        theta_deg, solution.Br_current_gap, label="Current only", color="#D97706", linewidth=1.2
    )
    axes[0].plot(theta_deg, solution.Br_total_gap, label="Total", color="#111827", linewidth=1.5)
    axes[0].set_title("Air-Gap Radial Flux Density")
    axes[0].set_xlabel("Mechanical Angle [deg]")
    axes[0].set_ylabel("Br [T]")
    axes[0].set_xlim(0.0, 360.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(theta_deg, solution.Btheta_pm_gap, label="PM only", color="#1D4ED8", linewidth=1.2)
    axes[1].plot(
        theta_deg,
        solution.Btheta_current_gap,
        label="Current only",
        color="#D97706",
        linewidth=1.2,
    )
    axes[1].plot(
        theta_deg, solution.Btheta_total_gap, label="Total", color="#111827", linewidth=1.5
    )
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


def save_harmonics_png(output_dir: Path, solution: BaselineSolution) -> Path:
    pm_idx = np.argsort(np.abs(solution.pm_spectrum.amplitudes))[::-1][:12]
    current_mag = np.sqrt(
        solution.current_spectrum.cosine**2 + solution.current_spectrum.sine**2
    )
    current_idx = np.argsort(current_mag)[::-1][:12]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].bar(
        solution.pm_spectrum.electrical_orders[pm_idx],
        solution.pm_spectrum.amplitudes[pm_idx],
        color="#7C3AED",
    )
    axes[0].set_title("Dominant PM Magnetization Harmonics")
    axes[0].set_xlabel("Electrical Harmonic Order")
    axes[0].set_ylabel("Mr Amplitude [A/m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(
        solution.current_spectrum.orders[current_idx],
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


def save_torque_angle_png(output_dir: Path, solution: BaselineSolution) -> Path:
    electrical_angles_deg, torques = torque_angle_curve(
        solution.theta_mech,
        solution.params,
        solution.pm_spectrum,
        total_air_gap_field,
    )
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(electrical_angles_deg, torques, color="#991B1B", linewidth=1.7)
    ax.scatter(
        [np.rad2deg(solution.params.torque_angle_elec)],
        [solution.torque_total],
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


def save_5region_pm_only_fields_png(
    output_dir: Path, solution: PMOnly5RegionSolution
) -> Path:
    theta_deg = mechanical_degrees(solution.theta_mech)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(theta_deg, solution.Br_gap, color="#1D4ED8", linewidth=1.4)
    axes[0].set_title("5-Region PM-Only Radial Flux Density")
    axes[0].set_xlabel("Mechanical Angle [deg]")
    axes[0].set_ylabel("Br [T]")
    axes[0].set_xlim(0.0, 360.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta_deg, solution.Btheta_gap, color="#0F766E", linewidth=1.4)
    axes[1].set_title("5-Region PM-Only Tangential Flux Density")
    axes[1].set_xlabel("Mechanical Angle [deg]")
    axes[1].set_ylabel("Btheta [T]")
    axes[1].set_xlim(0.0, 360.0)
    axes[1].grid(True, alpha=0.3)

    path = output_dir / "pm_only_5region_fields.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_5region_pm_only_cogging_png(
    output_dir: Path, solution: PMOnly5RegionSolution
) -> Path:
    period_deg = 360.0 / np.lcm(
        solution.params.stator_slots,
        solution.params.pole_count,
    )
    rotor_deg = mechanical_degrees(solution.rotor_angles_mech)
    full_rev_deg = np.linspace(0.0, 360.0, 721)
    wrapped_deg = np.mod(full_rev_deg, period_deg)
    tiled_torque = np.interp(
        wrapped_deg,
        rotor_deg,
        solution.cogging_torque_nm,
    )

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(full_rev_deg, tiled_torque, color="#991B1B", linewidth=1.6)
    for x in np.arange(0.0, 360.0 + 0.5 * period_deg, period_deg):
        ax.axvline(x, color="#9CA3AF", linewidth=0.6, alpha=0.35)
    ax.set_title("5-Region PM-Only Cogging Torque Preview Over 360 Mechanical Deg")
    ax.set_xlabel("Rotor Mechanical Angle [deg]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_xlim(0.0, 360.0)
    ax.grid(True, alpha=0.3)

    path = output_dir / "pm_only_5region_cogging.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_motor_schematic_png(output_dir: Path, solution: BaselineSolution) -> Path:
    params = solution.params
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"}, constrained_layout=True)

    slot_pitch_deg = 360.0 / params.stator_slots
    slot_air_deg = min(np.rad2deg(params.slot_opening_angle_mech), 0.72 * slot_pitch_deg)
    slot_outer_radius = params.region_iv_boundary_radius + 0.55 * (
        params.stator_outer_radius - params.region_iv_boundary_radius
    )
    slot_centers_deg = -15.0 - np.arange(params.stator_slots) * slot_pitch_deg

    # Visual winding pattern is illustrative only and is kept separate from the
    # solver convention so image debugging does not perturb the baseline.
    phase_colors = {"A": "#F59E0B", "B": "#FDE047", "C": "#16A34A"}
    slot_phase_labels = {
        1: ("A-", "C+"),
        2: ("B-", "A+"),
        3: ("C-", "B+"),
        4: ("A-", "C+"),
        5: ("B-", "A+"),
        6: ("C-", "B+"),
        7: ("A-", "C+"),
        8: ("B-", "A+"),
        9: ("C-", "B+"),
        10: ("A-", "C+"),
        11: ("B-", "A+"),
        12: ("C-", "B+"),
    }

    # Stator steel base ring
    ax.add_patch(
        Wedge(
            (0.0, 0.0),
            params.stator_outer_radius,
            0.0,
            360.0,
            width=params.stator_outer_radius - params.air_gap_outer_radius,
            facecolor="#D8DADF",
            edgecolor="#AEB5BF",
            linewidth=1.0,
        )
    )

    # Air gap
    ax.add_patch(
        Wedge(
            (0.0, 0.0),
            params.air_gap_outer_radius,
            0.0,
            360.0,
            width=params.air_gap_outer_radius - params.air_gap_inner_radius,
            facecolor="#F6F7FB",
            edgecolor="white",
            linewidth=1.0,
        )
    )

    # Rotor core and rotor yoke
    ax.add_patch(
        Wedge(
            (0.0, 0.0),
            params.air_gap_inner_radius,
            0.0,
            360.0,
            width=params.air_gap_inner_radius - params.rotor_outer_radius,
            facecolor="#22D3EE",
            edgecolor="white",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            params.rotor_outer_radius,
            facecolor="#BFC3C7",
            edgecolor="white",
            linewidth=1.0,
        )
    )

    # Shaft
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            0.55 * params.rotor_outer_radius,
            facecolor="#A8A8A8",
            edgecolor="white",
            linewidth=1.0,
        )
    )

    # Surface magnets
    pole_pitch_deg = 180.0 / params.pole_pairs
    magnet_arc_deg = np.rad2deg(params.magnet_arc_mech)
    magnet_colors = ["#DC2626", "#1D4ED8"]
    for pole_idx in range(2 * params.pole_pairs):
        center_deg = pole_idx * pole_pitch_deg
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                params.air_gap_inner_radius,
                center_deg - magnet_arc_deg / 2.0,
                center_deg + magnet_arc_deg / 2.0,
                width=params.air_gap_inner_radius - params.rotor_outer_radius,
                facecolor=magnet_colors[pole_idx % 2],
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
            )
        )

    # Radial-wall slots: tooth / air / tooth is controlled directly by angle.
    for slot_idx, center_deg in enumerate(slot_centers_deg):
        left_label, right_label = slot_phase_labels[slot_idx + 1]
        left_phase, right_phase = left_label[0], right_label[0]

        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                slot_outer_radius,
                center_deg - 0.5 * slot_air_deg,
                center_deg + 0.5 * slot_air_deg,
                width=slot_outer_radius - params.air_gap_outer_radius,
                facecolor="white",
                edgecolor="#B7BDC7",
                linewidth=1.0,
                zorder=5,
            )
        )

        coil_gap_deg = 0.12 * slot_air_deg
        coil_block_deg = max(0.18 * slot_air_deg, 0.42 * (slot_air_deg - coil_gap_deg))
        coil_inner_radius = params.slot_opening_radius + 0.28 * (
            slot_outer_radius - params.slot_opening_radius
        )
        coil_outer_radius = slot_outer_radius - 0.0025
        coil_specs = [
            (
                center_deg - 0.5 * coil_gap_deg - coil_block_deg,
                center_deg - 0.5 * coil_gap_deg,
                left_phase,
            ),
            (
                center_deg + 0.5 * coil_gap_deg,
                center_deg + 0.5 * coil_gap_deg + coil_block_deg,
                right_phase,
            ),
        ]
        for start_deg, end_deg, phase_name in coil_specs:
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    coil_outer_radius,
                    start_deg,
                    end_deg,
                    width=coil_outer_radius - coil_inner_radius,
                    facecolor=phase_colors[phase_name],
                    edgecolor="#6B7280",
                    linewidth=0.7,
                    zorder=6,
                )
            )

    # Region labels for the paper subdomains.
    region_font = dict(
        ha="center",
        va="center",
        fontsize=12,
        color="#111827",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.82),
        zorder=9,
    )
    label_positions = [
        ("I", 0.5 * (params.rotor_outer_radius + params.air_gap_inner_radius), 18.0),
        ("II", 0.5 * (params.air_gap_inner_radius + params.air_gap_outer_radius), 18.0),
        ("III", 0.5 * (params.air_gap_outer_radius + params.slot_opening_radius), 45.0),
        ("IV", 0.5 * (params.slot_opening_radius + params.region_iv_boundary_radius), 18.0),
        ("V", 0.5 * (params.region_iv_boundary_radius + params.stator_outer_radius), 18.0),
    ]
    for region_name, radius, angle_deg in label_positions:
        x, y = _polar_to_xy(radius, angle_deg)
        ax.text(x, y, region_name, **region_font)

    # Legend
    legend_x = 0.73 * params.stator_outer_radius
    legend_y = 0.76 * params.stator_outer_radius
    for idx, phase_name in enumerate(["A", "B", "C"]):
        y = legend_y - idx * 0.10 * params.stator_outer_radius
        ax.add_patch(
            Rectangle(
                (legend_x, y),
                0.11 * params.stator_outer_radius,
                0.045 * params.stator_outer_radius,
                facecolor=phase_colors[phase_name],
                edgecolor="#6B7280",
                linewidth=0.6,
                zorder=10,
            )
        )
        ax.text(
            legend_x + 0.14 * params.stator_outer_radius,
            y + 0.022 * params.stator_outer_radius,
            phase_name,
            ha="left",
            va="center",
            fontsize=11,
            weight="bold",
            color="#111827",
        )

    ax.text(
        0.0,
        -1.08 * params.stator_outer_radius,
        "PMSM Cross-Section With Paper Regions",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.0,
        -1.19 * params.stator_outer_radius,
        (
            f"Illustrative geometry only, 12 slots, 8 poles, "
            f"{1e3 * params.max_slot_opening_length:.1f} mm slot opening"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    lim = 1.1 * params.stator_outer_radius
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")

    path = output_dir / "motor_schematic.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

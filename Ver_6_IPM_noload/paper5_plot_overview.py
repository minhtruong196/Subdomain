import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
from paper5_plot_femm_geometry import plot_geometry as plot_actual_femm_geometry


DEFAULT_RESULT_DIR = "results/paper5_plots"
DEFAULT_SLOTLESS_FEMM = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_STRUCTURE1 = "results/paper5_structure1_mu_user_outeravg_corrected/structure1_br_bt.csv"
DEFAULT_STRUCTURE2 = "results/paper5_structure2_offset30_h5/structure2_br_bt.csv"
DEFAULT_COMBINED = "results/paper5_combined_slotless_user_mu/combined_br_bt.csv"
DEFAULT_DIAGNOSTIC_COMBINED = "results/paper5_combined_slotless_user_mu_kdiag/combined_br_bt.csv"
DEFAULT_BRIDGE_MU = "results/paper5_equivalent_femm_mu_user_bridge_points.csv"
DEFAULT_GEOMETRY_FEM = "paper_ipm_vshape_equivalent_1over6.FEM"


def read_field_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "angle_deg": np.array([float(row["angle_deg"]) for row in rows]),
        "Br_T": np.array([float(row["Br_T"]) for row in rows]),
        "Bt_T": np.array([float(row["Bt_T"]) for row in rows]),
    }


def read_mu_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def rms(values):
    return float(np.sqrt(np.mean(values * values)))


def plot_field_comparison(out_dir, datasets):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    styles = {
        "slotless FEMM": {"color": "black", "linewidth": 2.2},
        "slotted FEMM": {"color": "0.55", "linewidth": 1.4, "linestyle": "--"},
        "Structure 1": {"color": "#2f6fbb", "linewidth": 1.4},
        "Structure 2": {"color": "#d95f02", "linewidth": 1.7},
        "Structure 1+2": {"color": "#1b9e77", "linewidth": 1.7},
        "S2 + Kdiag*S1": {"color": "#984ea3", "linewidth": 1.4, "linestyle": "-."},
    }

    for name, data in datasets.items():
        if data is None:
            continue
        style = styles.get(name, {})
        label_br = f"{name} Br rms={rms(data['Br_T']):.3f}T"
        label_bt = f"{name} Bt rms={rms(data['Bt_T']):.3f}T"
        axes[0].plot(data["angle_deg"], data["Br_T"], label=label_br, **style)
        axes[1].plot(data["angle_deg"], data["Bt_T"], label=label_bt, **style)

    axes[0].set_title("Paper [5] Slotless Analytical Structures vs FEMM")
    axes[0].set_ylabel("Br (T)")
    axes[1].set_ylabel("Bt (T)")
    axes[1].set_xlabel("Mechanical angle (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out / "field_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def polar_xy(radius_mm, angle_rad):
    return radius_mm * math.cos(angle_rad), radius_mm * math.sin(angle_rad)


def draw_arc(ax, radius, start, end, **kwargs):
    theta = np.linspace(start, end, 300)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), **kwargs)


def draw_radial(ax, r0, r1, angle, **kwargs):
    x0, y0 = polar_xy(r0, angle)
    x1, y1 = polar_xy(r1, angle)
    ax.plot([x0, x1], [y0, y1], **kwargs)


def plot_geometry(out_dir, bridge_rows):
    """Legacy schematic plot of analytical spans. Kept for manual debugging only."""
    spec = base.PAPER_SPECS["vshape"]
    dims = equivalent.equivalent_pm_dimensions(spec)
    sector_start = math.radians(base.SECTOR_START_DEG_DEFAULT)
    sector_end = math.radians(base.SECTOR_START_DEG_DEFAULT + base.SECTOR_SPAN_DEG_DEFAULT)
    center = math.radians(30.0)
    alpha = dims["alpha_rad"]
    alpha1 = dims["alpha1_rad"]
    rf_inner = dims["rf"]
    rf_outer = dims["rm"]
    rl_minus_w1 = dims["rl"] - dims["w1"]
    rl = dims["rl"]
    rr = spec.rotor_outer_radius
    rs = spec.stator_inner_radius

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    ring_style = {"color": "0.25", "linewidth": 1.0}
    for radius, label in [
        (rf_inner, "PM inner"),
        (rf_outer, "PM outer / Rf paper"),
        (rl_minus_w1, "radial PM outer"),
        (rl, "Rl"),
        (rr, "Rr"),
        (rs, "Rs"),
    ]:
        draw_arc(ax, radius, sector_start, sector_end, **ring_style)
        x, y = polar_xy(radius, sector_end + math.radians(1.0))
        ax.text(x, y, label, fontsize=8, va="center")

    for angle in [sector_start, sector_end, center]:
        draw_radial(ax, 0, rs, angle, color="0.75", linewidth=0.8, linestyle="--")

    windows = [
        (center - alpha / 2.0, center - alpha1 / 2.0, "#d95f02", "Structure 1 radial PM span"),
        (center + alpha1 / 2.0, center + alpha / 2.0, "#d95f02", None),
        (center - alpha1 / 2.0, center + alpha1 / 2.0, "#7570b3", "Structure 2 tangential PM span"),
    ]
    for start, end, color, label in windows:
        draw_arc(ax, 0.5 * (rf_outer + rl_minus_w1), start, end, color=color, linewidth=5, alpha=0.75, label=label)
        draw_radial(ax, rf_outer, rl_minus_w1, start, color=color, linewidth=0.8, alpha=0.7)
        draw_radial(ax, rf_outer, rl_minus_w1, end, color=color, linewidth=0.8, alpha=0.7)

    for row in bridge_rows:
        x = float(row["x_mm"])
        y = float(row["y_mm"])
        mu = float(row["mu_r_abs_from_B_over_mu0H"])
        name = row["name"]
        ax.scatter([x], [y], s=48, color="#e7298a", zorder=5)
        ax.text(x + 0.8, y + 0.8, f"{name}\nmu={mu:.2f}", fontsize=8, color="#7a0048")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Equivalent Rotor Geometry And FEMM Bridge Sample Points")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out / "geometry_bridge_points.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot paper [5] current analytical/FEMM overview.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--slotless-femm", default=DEFAULT_SLOTLESS_FEMM)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--structure1", default=DEFAULT_STRUCTURE1)
    parser.add_argument("--structure2", default=DEFAULT_STRUCTURE2)
    parser.add_argument("--combined", default=DEFAULT_COMBINED)
    parser.add_argument("--diagnostic-combined", default=DEFAULT_DIAGNOSTIC_COMBINED)
    parser.add_argument("--bridge-mu", default=DEFAULT_BRIDGE_MU)
    parser.add_argument("--geometry-fem", default=DEFAULT_GEOMETRY_FEM)
    parser.add_argument(
        "--legacy-schematic-geometry",
        action="store_true",
        help="Plot the old analytical schematic instead of actual FEMM geometry.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = {
        "slotless FEMM": read_field_csv(args.slotless_femm),
        "slotted FEMM": read_field_csv(args.slotted_femm),
        "Structure 1": read_field_csv(args.structure1),
        "Structure 2": read_field_csv(args.structure2),
        "Structure 1+2": read_field_csv(args.combined),
        "S2 + Kdiag*S1": read_field_csv(args.diagnostic_combined),
    }

    field_path = plot_field_comparison(args.result_dir, datasets)
    if args.legacy_schematic_geometry:
        bridge_rows = read_mu_csv(args.bridge_mu)
        geometry_path = plot_geometry(args.result_dir, bridge_rows)
    else:
        geometry_path = plot_actual_femm_geometry(
            args.geometry_fem,
            Path(args.result_dir) / "actual_femm_geometry_bridge_points.png",
            args.bridge_mu,
        )

    print(f"Field comparison: {field_path}")
    print(f"Actual FEMM geometry/bridge points: {geometry_path}")


if __name__ == "__main__":
    main()

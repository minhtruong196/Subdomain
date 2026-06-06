import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper5_plot_femm_geometry import arc_points, read_fem_geometry


DEFAULT_FEM = "paper_ipm_vshape_equivalent_1over6.FEM"
DEFAULT_MU_CSV = "results/paper5_equivalent_femm_mu_user_bridge_points.csv"
DEFAULT_OUTPUT = "results/paper5_geometry_audit/equivalent_geometry_user_bridge_zoom.png"
DEFAULT_SUMMARY = "results/paper5_geometry_audit/equivalent_geometry_user_bridge_summary.csv"


GROUP_NAMES = {
    1: "stator_yoke_or_analysis_cut",
    2: "stator_slot_lower",
    3: "stator_slot_upper",
    4: "stator_tooth_tip",
    5: "airgap",
    6: "rotor_outer_core",
    7: "rotor_side_bridge",
    8: "rotor_radial_pm",
    9: "rotor_inner_pm_band_or_center_core",
    10: "rotor_inner_core",
    11: "shaft_air",
}


GROUP_COLORS = {
    1: "#8c8c8c",
    2: "#bdbdbd",
    3: "#9ecae1",
    4: "#6baed6",
    5: "#31a354",
    6: "#636363",
    7: "#9467bd",
    8: "#d62728",
    9: "#ff7f0e",
    10: "#252525",
    11: "#17becf",
}


def read_mu_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def segment_distance(px, py, p0, p1):
    p = np.array([px, py], dtype=float)
    a = np.array(p0, dtype=float)
    b = np.array(p1, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def nearest_geometry_distance(point, nodes, segments, arcs):
    px = float(point["x_mm"])
    py = float(point["y_mm"])
    nearest = {"distance_mm": float("inf"), "kind": "", "group": None}

    for n0, n1, group in segments:
        distance = segment_distance(px, py, nodes[n0], nodes[n1])
        if distance < nearest["distance_mm"]:
            nearest = {"distance_mm": distance, "kind": "segment", "group": group}

    for n0, n1, angle, group in arcs:
        x, y = arc_points(nodes[n0], nodes[n1], angle, samples=360)
        distances = np.hypot(x - px, y - py)
        distance = float(np.min(distances))
        if distance < nearest["distance_mm"]:
            nearest = {"distance_mm": distance, "kind": "arc", "group": group}

    return nearest


def nearest_label(point, labels):
    px = float(point["x_mm"])
    py = float(point["y_mm"])
    best = None
    for label in labels:
        distance = math.hypot(px - label["x"], py - label["y"])
        if best is None or distance < best["distance_mm"]:
            best = {
                "distance_mm": distance,
                "group": label["group"],
                "group_name": GROUP_NAMES.get(label["group"], "unknown"),
                "x_mm": label["x"],
                "y_mm": label["y"],
            }
    return best or {
        "distance_mm": float("nan"),
        "group": None,
        "group_name": "none",
        "x_mm": float("nan"),
        "y_mm": float("nan"),
    }


def draw_geometry(ax, nodes, segments, arcs, labels, show_labels=False):
    for n0, n1, group in segments:
        x0, y0 = nodes[n0]
        x1, y1 = nodes[n1]
        ax.plot([x0, x1], [y0, y1], color=GROUP_COLORS.get(group, "#333333"), linewidth=1.0)
    for n0, n1, angle, group in arcs:
        x, y = arc_points(nodes[n0], nodes[n1], angle)
        ax.plot(x, y, color=GROUP_COLORS.get(group, "#333333"), linewidth=1.0)

    for label in labels:
        group = label["group"]
        if group not in GROUP_COLORS:
            continue
        ax.scatter(label["x"], label["y"], s=16, color=GROUP_COLORS[group], zorder=3)
        if show_labels:
            ax.text(label["x"] + 0.25, label["y"] + 0.25, f"g{group}", fontsize=6)


def plot_audit(fem_path, mu_csv, output_path):
    nodes, segments, arcs, labels = read_fem_geometry(fem_path)
    rows = read_mu_rows(mu_csv)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    titles = ["full equivalent FEMM sector"] + [row["name"] for row in rows]

    for ax, title in zip(axes, titles):
        draw_geometry(ax, nodes, segments, arcs, labels, show_labels=(title == titles[0]))
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)

    for row in rows:
        x = float(row["x_mm"])
        y = float(row["y_mm"])
        mu = float(row["mu_r_abs_from_B_over_mu0H"])
        b_abs = float(row["B_abs_T"])
        axes[0].scatter([x], [y], s=80, facecolors="none", edgecolors="#e7298a", linewidths=1.8, zorder=5)
        axes[0].text(x + 0.4, y + 0.4, row["name"], fontsize=7, color="#a00058")
        idx = rows.index(row) + 1
        axes[idx].scatter([x], [y], s=110, facecolors="none", edgecolors="#e7298a", linewidths=2.0, zorder=5)
        axes[idx].text(x + 0.15, y + 0.15, f"mu={mu:.2f}\n|B|={b_abs:.3f} T", fontsize=8, color="#a00058")
        axes[idx].set_xlim(x - 4.0, x + 4.0)
        axes[idx].set_ylim(y - 4.0, y + 4.0)

    for ax in axes:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle(f"Actual Equivalent FEMM Geometry Audit: {Path(fem_path).name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def write_summary(fem_path, mu_csv, summary_path):
    nodes, segments, arcs, labels = read_fem_geometry(fem_path)
    rows = read_mu_rows(mu_csv)
    output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "x_mm",
        "y_mm",
        "B_abs_T",
        "H_abs_A_per_m",
        "mu_r_abs_from_B_over_mu0H",
        "nearest_label_group",
        "nearest_label_group_name",
        "nearest_label_distance_mm",
        "nearest_boundary_kind",
        "nearest_boundary_group",
        "nearest_boundary_group_name",
        "nearest_boundary_distance_mm",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            nearest = nearest_geometry_distance(row, nodes, segments, arcs)
            label = nearest_label(row, labels)
            writer.writerow(
                {
                    "name": row["name"],
                    "x_mm": row["x_mm"],
                    "y_mm": row["y_mm"],
                    "B_abs_T": row["B_abs_T"],
                    "H_abs_A_per_m": row["H_abs_A_per_m"],
                    "mu_r_abs_from_B_over_mu0H": row["mu_r_abs_from_B_over_mu0H"],
                    "nearest_label_group": label["group"],
                    "nearest_label_group_name": label["group_name"],
                    "nearest_label_distance_mm": label["distance_mm"],
                    "nearest_boundary_kind": nearest["kind"],
                    "nearest_boundary_group": nearest["group"],
                    "nearest_boundary_group_name": GROUP_NAMES.get(nearest["group"], "unknown"),
                    "nearest_boundary_distance_mm": nearest["distance_mm"],
                }
            )
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit user-selected saturated bridge points against actual equivalent FEMM geometry."
    )
    parser.add_argument("--fem", default=DEFAULT_FEM)
    parser.add_argument("--mu-csv", default=DEFAULT_MU_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_path = plot_audit(args.fem, args.mu_csv, args.output)
    summary_path = write_summary(args.fem, args.mu_csv, args.summary)
    print(f"Geometry audit plot: {plot_path}")
    print(f"Geometry audit summary: {summary_path}")


if __name__ == "__main__":
    main()

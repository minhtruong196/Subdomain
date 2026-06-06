import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FEM = "paper5_slotless_equivalent_1over6.FEM"
DEFAULT_OUTPUT = "results/paper5_plots/femm_geometry_actual.png"


def read_fem_geometry(path):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    nodes = []
    segments = []
    arcs = []
    labels = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("[NumPoints]"):
            count = int(line.split("=")[1])
            idx += 1
            for _ in range(count):
                parts = lines[idx].split()
                nodes.append((float(parts[0]), float(parts[1])))
                idx += 1
            continue
        if line.startswith("[NumSegments]"):
            count = int(line.split("=")[1])
            idx += 1
            for _ in range(count):
                parts = lines[idx].split()
                segments.append((int(parts[0]), int(parts[1]), int(parts[5]) if len(parts) > 5 else 0))
                idx += 1
            continue
        if line.startswith("[NumArcSegments]"):
            count = int(line.split("=")[1])
            idx += 1
            for _ in range(count):
                parts = lines[idx].split()
                arcs.append(
                    (
                        int(parts[0]),
                        int(parts[1]),
                        float(parts[2]),
                        int(parts[6]) if len(parts) > 6 else 0,
                    )
                )
                idx += 1
            continue
        if line.startswith("[NumBlockLabels]"):
            count = int(line.split("=")[1])
            idx += 1
            for _ in range(count):
                parts = lines[idx].split()
                labels.append(
                    {
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "block_type": int(parts[2]),
                        "magdir": float(parts[5]),
                        "group": int(parts[6]),
                    }
                )
                idx += 1
            continue
        idx += 1
    return nodes, segments, arcs, labels


def read_mu_rows(path):
    if not path:
        return []
    import csv

    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def arc_points(p0, p1, angle_deg, samples=120):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len == 0.0 or abs(angle_deg) < 1e-12:
        return np.array([p0[0], p1[0]]), np.array([p0[1], p1[1]])
    phi = math.radians(angle_deg)
    radius = chord_len / (2.0 * math.sin(abs(phi) / 2.0))
    mid = 0.5 * (p0 + p1)
    unit = chord / chord_len
    normal = np.array([-unit[1], unit[0]])
    h = math.sqrt(max(radius * radius - (0.5 * chord_len) ** 2, 0.0))
    center = mid + normal * h
    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    theta = np.linspace(a0, a0 + phi, samples)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    # FEMM arcs can be clockwise depending on node order; pick the curve whose
    # endpoint is closer to the stored second node.
    if np.hypot(x[-1] - p1[0], y[-1] - p1[1]) > 1e-3:
        center = mid - normal * h
        a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
        theta = np.linspace(a0, a0 + phi, samples)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
    return x, y


def plot_geometry(fem_path, output_path, mu_csv=None):
    nodes, segments, arcs, labels = read_fem_geometry(fem_path)
    mu_rows = read_mu_rows(mu_csv)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for n0, n1, group in segments:
        x0, y0 = nodes[n0]
        x1, y1 = nodes[n1]
        color = "#777777" if group == 1 else "#333333"
        lw = 0.9 if group == 1 else 1.2
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw)
    for n0, n1, angle, group in arcs:
        x, y = arc_points(nodes[n0], nodes[n1], angle)
        color = "#777777" if group == 1 else "#333333"
        lw = 0.9 if group == 1 else 1.2
        ax.plot(x, y, color=color, linewidth=lw)

    group_colors = {
        5: "#1f77b4",
        6: "#555555",
        7: "#9467bd",
        8: "#d62728",
        9: "#ff7f0e",
        10: "#555555",
        11: "#17becf",
    }
    for label in labels:
        group = label["group"]
        if group not in group_colors:
            continue
        ax.scatter(label["x"], label["y"], s=28, color=group_colors[group], zorder=4)
        ax.text(label["x"] + 0.35, label["y"] + 0.35, f"g{group}", fontsize=7)

    for row in mu_rows:
        x = float(row["x_mm"])
        y = float(row["y_mm"])
        mu = float(row["mu_r_abs_from_B_over_mu0H"])
        name = row["name"]
        ax.scatter([x], [y], s=82, facecolors="none", edgecolors="#e7298a", linewidths=1.8, zorder=6)
        ax.text(x + 0.5, y - 0.8, f"{name}\nmu={mu:.1f}", fontsize=7, color="#a00058")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actual FEMM Geometry From {Path(fem_path).name}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Plot actual geometry stored in a FEMM .FEM file.")
    parser.add_argument("--fem", default=DEFAULT_FEM)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mu-csv", help="Optional FEMM mu sample CSV to overlay.")
    return parser.parse_args()


def main():
    args = parse_args()
    output = plot_geometry(args.fem, args.output, args.mu_csv)
    print(f"Actual FEMM geometry plot: {output}")


if __name__ == "__main__":
    main()

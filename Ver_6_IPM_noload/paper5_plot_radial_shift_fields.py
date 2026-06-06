import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper5_slotting_correction import read_field_csv


DEFAULT_RESULT_DIR = "results/paper5_radial_shift_validation_gap_p1"


def read_summary(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: float(row["shift_mm"]))
    return rows


def plot_grid(result_dir, rows):
    out = Path(result_dir)
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])

    for row_idx, row in enumerate(rows):
        shift = float(row["shift_mm"])
        femm = read_field_csv(row["femm_csv"])
        analytical = read_field_csv(row["analytical_csv"])

        ax_br = axes[row_idx, 0]
        ax_bt = axes[row_idx, 1]
        ax_br.plot(femm["angle_deg"], femm["Br_T"], color="black", linewidth=2.0, label="FEMM equivalent")
        ax_br.plot(
            analytical["angle_deg"],
            analytical["Br_T"],
            color="#2ca02c",
            linewidth=1.6,
            label="analytical",
        )
        ax_bt.plot(femm["angle_deg"], femm["Bt_T"], color="black", linewidth=2.0, label="FEMM equivalent")
        ax_bt.plot(
            analytical["angle_deg"],
            analytical["Bt_T"],
            color="#2ca02c",
            linewidth=1.6,
            label="analytical",
        )

        ax_br.set_ylabel(f"shift {shift:g}\nBr (T)")
        ax_bt.set_ylabel("Bt (T)")
        ax_br.grid(True, alpha=0.28)
        ax_bt.grid(True, alpha=0.28)
        ax_br.text(
            0.02,
            0.92,
            f"Br L2={float(row['Br_L2_vs_femm_equivalent']):.3f}",
            transform=ax_br.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        ax_bt.text(
            0.02,
            0.92,
            f"Bt L2={float(row['Bt_L2_vs_femm_equivalent']):.3f}",
            transform=ax_bt.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        if row_idx == 0:
            ax_br.set_title("Br")
            ax_bt.set_title("Bt")
            ax_br.legend(fontsize=8)
            ax_bt.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Mechanical angle (deg)")
    axes[-1, 1].set_xlabel("Mechanical angle (deg)")
    fig.tight_layout()
    path = out / "radial_shift_field_grid.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_individual(result_dir, rows):
    out = Path(result_dir)
    paths = []
    for row in rows:
        shift = float(row["shift_mm"])
        safe = f"{shift:.6g}".replace("-", "m").replace(".", "p")
        femm = read_field_csv(row["femm_csv"])
        analytical = read_field_csv(row["analytical_csv"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for ax, comp, ylabel in (
            (axes[0], "Br_T", "Br (T)"),
            (axes[1], "Bt_T", "Bt (T)"),
        ):
            ax.plot(femm["angle_deg"], femm[comp], color="black", linewidth=2.0, label="FEMM equivalent")
            ax.plot(analytical["angle_deg"], analytical[comp], color="#2ca02c", linewidth=1.7, label="analytical")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.28)
            ax.legend(fontsize=8)
        axes[0].set_title(
            f"radial_shift={shift:g} mm, "
            f"Br L2={float(row['Br_L2_vs_femm_equivalent']):.4f}, "
            f"Bt L2={float(row['Bt_L2_vs_femm_equivalent']):.4f}"
        )
        axes[1].set_xlabel("Mechanical angle (deg)")
        fig.tight_layout()
        path = out / f"radial_shift_{safe}_field_comparison.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Plot FEMM vs analytical Br/Bt for radial shift sweep results.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = read_summary(result_dir / "radial_shift_validation_summary.csv")
    grid = plot_grid(result_dir, rows)
    individual = plot_individual(result_dir, rows)
    print("=== Radial shift field plots ===")
    print(f"Grid: {grid}")
    for path in individual:
        print(f"Individual: {path}")


if __name__ == "__main__":
    main()

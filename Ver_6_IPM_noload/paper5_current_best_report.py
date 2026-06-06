import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper5_slotting_correction import read_field_csv, interp_to
from paper5_vshape_stage1 import relative_l2, stats


DEFAULT_SLOTLESS_FEMM = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_SUBDOMAIN = "results/paper5_structure2_offset30_h7/structure2_br_bt.csv"
DEFAULT_CORRECTED = "results/paper5_slotting_structure2_h7_empirical_lh6/slotting_corrected_br_bt.csv"
DEFAULT_RESULT_DIR = "results/paper5_current_best"


def on_grid(data, angle):
    return {
        "angle_deg": angle,
        "Br_T": interp_to(data["angle_deg"], data["Br_T"], angle),
        "Bt_T": interp_to(data["angle_deg"], data["Bt_T"], angle),
    }


def region_masks(angle):
    return {
        "left_edge_0_12deg": (0.0 <= angle) & (angle <= 12.0),
        "middle_12_48deg": (12.0 < angle) & (angle < 48.0),
        "right_edge_48_60deg": (48.0 <= angle) & (angle <= 60.0),
        "all_0_60deg": np.ones_like(angle, dtype=bool),
    }


def write_region_summary(path, angle, datasets, reference_name):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    masks = region_masks(angle)
    ref = datasets[reference_name]
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "component", "region", "relative_l2", "rms_T", "ref_rms_T"])
        for name, data in datasets.items():
            if name == reference_name:
                continue
            for component in ("Br_T", "Bt_T"):
                for region, mask in masks.items():
                    writer.writerow(
                        [
                            name,
                            component,
                            region,
                            relative_l2(data[component][mask], ref[component][mask]),
                            stats(data[component][mask])["rms"],
                            stats(ref[component][mask])["rms"],
                        ]
                    )
    return output


def plot_report(out_dir, angle, slotless, slotted, subdomain, corrected, corrected_label):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "current_best_focus_report.png"

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
    components = [("Br_T", "Br (T)"), ("Bt_T", "Bt (T)")]

    for col, (component, ylabel) in enumerate(components):
        ax = axes[0, col]
        ax.plot(angle, slotted[component], color="black", linewidth=2.0, label="FEMM slotted benchmark")
        ax.plot(angle, slotless[component], color="0.55", linestyle="--", linewidth=1.2, label="FEMM slotless benchmark")
        ax.plot(angle, subdomain[component], color="#ff7f0e", linewidth=1.4, label="subdomain slotless")
        ax.plot(angle, corrected[component], color="#2ca02c", linewidth=1.7, label=corrected_label)
        ax.set_title(f"{ylabel} full sector")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)

        for row, (start, end, title) in enumerate(((0.0, 12.0, "left edge"), (48.0, 60.0, "right edge")), start=1):
            ax = axes[row, col]
            mask = (start <= angle) & (angle <= end)
            ax.plot(angle[mask], slotted[component][mask], color="black", linewidth=2.0, label="FEMM slotted")
            ax.plot(angle[mask], corrected[component][mask], color="#2ca02c", linewidth=1.8, label="corrected")
            ax.plot(angle[mask], subdomain[component][mask], color="#ff7f0e", linewidth=1.2, label="subdomain")
            ax.fill_between(
                angle[mask],
                corrected[component][mask],
                slotted[component][mask],
                color="#2ca02c",
                alpha=0.15,
                linewidth=0,
            )
            ax.set_title(f"{ylabel} {title} zoom")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.28)
            ax.legend(fontsize=8)

    axes[2, 0].set_xlabel("Mechanical angle (deg)")
    axes[2, 1].set_xlabel("Mechanical angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a compact current-best paper [5] comparison report.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--slotless-femm", default=DEFAULT_SLOTLESS_FEMM)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--subdomain", default=DEFAULT_SUBDOMAIN)
    parser.add_argument("--corrected", default=DEFAULT_CORRECTED)
    parser.add_argument("--corrected-label", default="subdomain + FEMM-derived slotting")
    return parser.parse_args()


def main():
    args = parse_args()
    slotted = read_field_csv(args.slotted_femm)
    angle = slotted["angle_deg"]
    datasets = {
        "FEMM slotless benchmark": on_grid(read_field_csv(args.slotless_femm), angle),
        "FEMM slotted benchmark": slotted,
        "subdomain slotless": on_grid(read_field_csv(args.subdomain), angle),
        args.corrected_label: on_grid(read_field_csv(args.corrected), angle),
    }
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = write_region_summary(
        out / "current_best_region_errors.csv",
        angle,
        datasets,
        "FEMM slotted benchmark",
    )
    plot = plot_report(
        out,
        angle,
        datasets["FEMM slotless benchmark"],
        datasets["FEMM slotted benchmark"],
        datasets["subdomain slotless"],
        datasets[args.corrected_label],
        args.corrected_label,
    )
    print(f"Current-best focus plot: {plot}")
    print(f"Region error summary: {summary}")


if __name__ == "__main__":
    main()

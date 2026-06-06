import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper5_vshape_stage1 import read_benchmark, relative_l2, best_scalar, stats


DEFAULT_STRUCTURE1_CSV = "results/paper5_structure1_mub1723_offset30_h3/structure1_br_bt.csv"
DEFAULT_STRUCTURE2_CSV = "results/paper5_structure2_offset30_h5/structure2_br_bt.csv"
DEFAULT_BENCHMARK_CSV = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"
DEFAULT_RESULT_DIR = "results/paper5_combined_slotless"


def read_component(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    angle_deg = np.array([float(row["angle_deg"]) for row in rows])
    br = np.array([float(row["Br_T"]) for row in rows])
    bt = np.array([float(row["Bt_T"]) for row in rows])
    return angle_deg, br, bt


def write_outputs(out_dir, angle_deg, br, bt, benchmark, scale1, scale2):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "combined_br_bt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for a, br_v, bt_v in zip(angle_deg, br, bt):
            writer.writerow([a, br_v, bt_v])

    summary_path = out / "combined_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        writer.writerow(["structure1_scale", scale1])
        writer.writerow(["structure2_scale", scale2])
        for prefix, values in (("model_Br", stats(br)), ("model_Bt", stats(bt))):
            for key, value in values.items():
                writer.writerow([f"{prefix}_{key}", value])
        if benchmark is not None:
            _, br_ref, bt_ref = benchmark
            for prefix, values in (("femm_Br", stats(br_ref)), ("femm_Bt", stats(bt_ref))):
                for key, value in values.items():
                    writer.writerow([f"{prefix}_{key}", value])
            br_best = best_scalar(br, br_ref)
            bt_best = best_scalar(bt, bt_ref)
            writer.writerow(["Br_relative_l2", relative_l2(br, br_ref)])
            writer.writerow(["Bt_relative_l2", relative_l2(bt, bt_ref)])
            writer.writerow(["Br_best_scalar_to_FEMM", br_best])
            writer.writerow(["Bt_best_scalar_to_FEMM", bt_best])
            writer.writerow(["Br_relative_l2_after_best_scalar", relative_l2(br * br_best, br_ref)])
            writer.writerow(["Bt_relative_l2_after_best_scalar", relative_l2(bt * bt_best, bt_ref)])

    png_path = out / "combined_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(angle_deg, br, label="Structure 1 + 2 Br")
    axes[1].plot(angle_deg, bt, label="Structure 1 + 2 Bt")
    if benchmark is not None:
        _, br_ref, bt_ref = benchmark
        axes[0].plot(angle_deg, br_ref, "--", label="slotless FEMM Br")
        axes[1].plot(angle_deg, bt_ref, "--", label="slotless FEMM Bt")
    axes[0].set_ylabel("Br (T)")
    axes[1].set_ylabel("Bt (T)")
    axes[1].set_xlabel("Mechanical angle (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return csv_path, summary_path, png_path


def parse_args():
    parser = argparse.ArgumentParser(description="Combine paper [5] Structure 1 and Structure 2 slotless fields.")
    parser.add_argument("--structure1-csv", default=DEFAULT_STRUCTURE1_CSV)
    parser.add_argument("--structure2-csv", default=DEFAULT_STRUCTURE2_CSV)
    parser.add_argument("--benchmark-csv", default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--structure1-scale", type=float, default=1.0)
    parser.add_argument("--structure2-scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    angle1, br1, bt1 = read_component(args.structure1_csv)
    angle2, br2, bt2 = read_component(args.structure2_csv)
    if len(angle1) != len(angle2) or np.max(np.abs(angle1 - angle2)) > 1e-9:
        raise ValueError("Structure CSV angle grids do not match.")

    br = args.structure1_scale * br1 + args.structure2_scale * br2
    bt = args.structure1_scale * bt1 + args.structure2_scale * bt2
    benchmark = read_benchmark(args.benchmark_csv)
    csv_path, summary_path, png_path = write_outputs(
        args.result_dir,
        angle1,
        br,
        bt,
        benchmark,
        args.structure1_scale,
        args.structure2_scale,
    )

    print("=== Paper [5] combined slotless Structure 1 + 2 ===")
    print(f"structure1_scale = {args.structure1_scale:g}; structure2_scale = {args.structure2_scale:g}")
    print(f"model Br rms = {stats(br)['rms']:.6g} T; Bt rms = {stats(bt)['rms']:.6g} T")
    if benchmark is not None:
        _, br_ref, bt_ref = benchmark
        print(f"FEMM Br rms = {stats(br_ref)['rms']:.6g} T; Bt rms = {stats(bt_ref)['rms']:.6g} T")
        print(f"relative L2: Br = {relative_l2(br, br_ref):.6g}; Bt = {relative_l2(bt, bt_ref):.6g}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()

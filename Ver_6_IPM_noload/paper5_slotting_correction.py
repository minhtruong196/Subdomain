import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper5_vshape_stage1 import best_scalar, relative_l2, stats


DEFAULT_SLOTLESS_FEMM = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_MODEL = "results/paper5_structure2_offset30_h5/structure2_br_bt.csv"
DEFAULT_RESULT_DIR = "results/paper5_slotting_correction"
DEFAULT_SLOTS = 36
DEFAULT_PERIODIC_MULTIPLIER = 6


def read_field_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "angle_deg": np.array([float(row["angle_deg"]) for row in rows]),
        "Br_T": np.array([float(row["Br_T"]) for row in rows]),
        "Bt_T": np.array([float(row["Bt_T"]) for row in rows]),
    }


def write_field_csv(path, angle_deg, br, bt):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for angle, br_v, bt_v in zip(angle_deg, br, bt):
            writer.writerow([angle, br_v, bt_v])
    return output


def write_lambda_csv(path, angle_deg, lambda_a_raw, lambda_b_raw, lambda_a, lambda_b):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "lambda_a_raw", "lambda_b_raw", "lambda_a", "lambda_b"])
        for values in zip(angle_deg, lambda_a_raw, lambda_b_raw, lambda_a, lambda_b):
            writer.writerow(values)
    return output


def interp_to(angle_src, values, angle_dst):
    return np.interp(angle_dst, angle_src, values)


def solve_lambda(slotless, slotted, eps=1e-8):
    br0 = slotless["Br_T"]
    bt0 = slotless["Bt_T"]
    brs = interp_to(slotted["angle_deg"], slotted["Br_T"], slotless["angle_deg"])
    bts = interp_to(slotted["angle_deg"], slotted["Bt_T"], slotless["angle_deg"])

    denom = br0 * br0 + bt0 * bt0
    valid = denom > eps
    lambda_a = np.ones_like(br0)
    lambda_b = np.zeros_like(br0)
    lambda_a[valid] = (br0[valid] * brs[valid] + bt0[valid] * bts[valid]) / denom[valid]
    lambda_b[valid] = (bt0[valid] * brs[valid] - br0[valid] * bts[valid]) / denom[valid]

    # Keep endpoints benign because flux density is near zero there and the ratio is ill-conditioned.
    if np.any(valid):
        first = int(np.argmax(valid))
        last = len(valid) - int(np.argmax(valid[::-1])) - 1
        lambda_a[:first] = lambda_a[first]
        lambda_b[:first] = lambda_b[first]
        lambda_a[last + 1 :] = lambda_a[last]
        lambda_b[last + 1 :] = lambda_b[last]
    return lambda_a, lambda_b


def periodic_slot_average(angle_deg, values, slots, periodic_multiplier, phase_bins=721):
    sector_span = 360.0 / periodic_multiplier
    slot_pitch = 360.0 / slots
    phase = np.mod(angle_deg, slot_pitch)
    phase_grid = np.linspace(0.0, slot_pitch, phase_bins)
    averaged = np.zeros_like(phase_grid)

    for idx, phase_value in enumerate(phase_grid):
        shifted = np.mod(phase - phase_value + 0.5 * slot_pitch, slot_pitch) - 0.5 * slot_pitch
        sigma = slot_pitch / 80.0
        weights = np.exp(-0.5 * (shifted / sigma) ** 2)
        if np.sum(weights) < 1e-12:
            averaged[idx] = np.interp(phase_value, phase, values)
        else:
            averaged[idx] = float(np.sum(weights * values) / np.sum(weights))

    return phase_grid, averaged, sector_span, slot_pitch


def fourier_fit_periodic(angle_deg, raw_values, slots, periodic_multiplier, harmonics, preserve_mean=None):
    phase_grid, averaged, _, slot_pitch = periodic_slot_average(
        angle_deg, raw_values, slots, periodic_multiplier
    )
    x = 2.0 * np.pi * phase_grid / slot_pitch
    coeffs = [np.ones_like(x)]
    for h in range(1, harmonics + 1):
        coeffs.append(np.cos(h * x))
        coeffs.append(np.sin(h * x))
    basis = np.vstack(coeffs).T
    fit, *_ = np.linalg.lstsq(basis, averaged, rcond=None)

    phase = np.mod(angle_deg, slot_pitch)
    xp = 2.0 * np.pi * phase / slot_pitch
    out = fit[0] * np.ones_like(angle_deg)
    cursor = 1
    for h in range(1, harmonics + 1):
        out += fit[cursor] * np.cos(h * xp)
        out += fit[cursor + 1] * np.sin(h * xp)
        cursor += 2
    if preserve_mean is not None:
        out += preserve_mean - float(np.mean(out))
    return out


def smooth_lambda(angle_deg, lambda_a_raw, lambda_b_raw, slots, periodic_multiplier, harmonics):
    lambda_a = fourier_fit_periodic(
        angle_deg,
        lambda_a_raw,
        slots,
        periodic_multiplier,
        harmonics,
        preserve_mean=float(np.mean(lambda_a_raw)),
    )
    lambda_b = fourier_fit_periodic(
        angle_deg,
        lambda_b_raw,
        slots,
        periodic_multiplier,
        harmonics,
        preserve_mean=float(np.mean(lambda_b_raw)),
    )
    return lambda_a, lambda_b


def apply_slotting(model, lambda_a, lambda_b):
    br = model["Br_T"]
    bt = model["Bt_T"]
    return br * lambda_a + bt * lambda_b, bt * lambda_a - br * lambda_b


def comparison_summary(br, bt, reference):
    br_ref = reference["Br_T"]
    bt_ref = reference["Bt_T"]
    return {
        "Br_relative_l2": relative_l2(br, br_ref),
        "Bt_relative_l2": relative_l2(bt, bt_ref),
        "Br_best_scalar": best_scalar(br, br_ref),
        "Bt_best_scalar": best_scalar(bt, bt_ref),
        "model_Br_rms": stats(br)["rms"],
        "model_Bt_rms": stats(bt)["rms"],
        "femm_Br_rms": stats(br_ref)["rms"],
        "femm_Bt_rms": stats(bt_ref)["rms"],
    }


def write_summary(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        for key, value in rows.items():
            writer.writerow([key, value])
    return output


def plot_outputs(out_dir, angle_deg, lambda_a_raw, lambda_b_raw, lambda_a, lambda_b, model, slotted_model, slotless_femm, slotted_femm):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lambda_png = out / "lambda_empirical_fit.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(angle_deg, lambda_a_raw, color="0.7", linewidth=1.0, label="raw lambda_a")
    axes[0].plot(angle_deg, lambda_a, color="#1f77b4", linewidth=1.8, label="periodic fit lambda_a")
    axes[1].plot(angle_deg, lambda_b_raw, color="0.7", linewidth=1.0, label="raw lambda_b")
    axes[1].plot(angle_deg, lambda_b, color="#d62728", linewidth=1.8, label="periodic fit lambda_b")
    axes[0].set_ylabel("lambda_a")
    axes[1].set_ylabel("lambda_b")
    axes[1].set_xlabel("Mechanical angle (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.legend()
    fig.tight_layout()
    fig.savefig(lambda_png, dpi=220)
    plt.close(fig)

    field_png = out / "slotting_corrected_field_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(angle_deg, slotted_femm["Br_T"], color="black", linewidth=2.0, label="slotted FEMM Br")
    axes[0].plot(angle_deg, slotless_femm["Br_T"], color="0.5", linestyle="--", linewidth=1.2, label="slotless FEMM Br")
    axes[0].plot(angle_deg, model["Br_T"], color="#ff7f0e", linewidth=1.3, label="input model Br")
    axes[0].plot(angle_deg, slotted_model["Br_T"], color="#2ca02c", linewidth=1.8, label="slotting-corrected model Br")

    axes[1].plot(angle_deg, slotted_femm["Bt_T"], color="black", linewidth=2.0, label="slotted FEMM Bt")
    axes[1].plot(angle_deg, slotless_femm["Bt_T"], color="0.5", linestyle="--", linewidth=1.2, label="slotless FEMM Bt")
    axes[1].plot(angle_deg, model["Bt_T"], color="#ff7f0e", linewidth=1.3, label="input model Bt")
    axes[1].plot(angle_deg, slotted_model["Bt_T"], color="#2ca02c", linewidth=1.8, label="slotting-corrected model Bt")
    axes[0].set_ylabel("Br (T)")
    axes[1].set_ylabel("Bt (T)")
    axes[1].set_xlabel("Mechanical angle (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(field_png, dpi=220)
    plt.close(fig)
    return lambda_png, field_png


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply paper [5] Eq. (42)-(43) slotting correction using an empirical "
            "complex relative permeance inferred from FEMM slotless/slotted benchmarks."
        )
    )
    parser.add_argument("--slotless-femm", default=DEFAULT_SLOTLESS_FEMM)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--model-csv", default=DEFAULT_MODEL)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--periodic-multiplier", type=int, default=DEFAULT_PERIODIC_MULTIPLIER)
    parser.add_argument("--lambda-harmonics", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    slotless_femm = read_field_csv(args.slotless_femm)
    slotted_femm = read_field_csv(args.slotted_femm)
    model = read_field_csv(args.model_csv)

    angle = slotless_femm["angle_deg"]
    if len(model["angle_deg"]) != len(angle) or np.max(np.abs(model["angle_deg"] - angle)) > 1e-9:
        model = {
            "angle_deg": angle,
            "Br_T": interp_to(model["angle_deg"], model["Br_T"], angle),
            "Bt_T": interp_to(model["angle_deg"], model["Bt_T"], angle),
        }
    slotted_on_grid = {
        "angle_deg": angle,
        "Br_T": interp_to(slotted_femm["angle_deg"], slotted_femm["Br_T"], angle),
        "Bt_T": interp_to(slotted_femm["angle_deg"], slotted_femm["Bt_T"], angle),
    }

    lambda_a_raw, lambda_b_raw = solve_lambda(slotless_femm, slotted_on_grid)
    lambda_a, lambda_b = smooth_lambda(
        angle,
        lambda_a_raw,
        lambda_b_raw,
        args.slots,
        args.periodic_multiplier,
        args.lambda_harmonics,
    )

    br_slot, bt_slot = apply_slotting(model, lambda_a, lambda_b)
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    field_csv = write_field_csv(out / "slotting_corrected_br_bt.csv", angle, br_slot, bt_slot)
    lambda_csv = write_lambda_csv(out / "empirical_lambda.csv", angle, lambda_a_raw, lambda_b_raw, lambda_a, lambda_b)
    summary = comparison_summary(br_slot, bt_slot, slotted_on_grid)
    summary.update(
        {
            "input_model_csv": args.model_csv,
            "slotless_femm_csv": args.slotless_femm,
            "slotted_femm_csv": args.slotted_femm,
            "lambda_harmonics": args.lambda_harmonics,
            "slots": args.slots,
        }
    )
    summary_csv = write_summary(out / "slotting_summary.csv", summary)
    lambda_png, field_png = plot_outputs(
        out,
        angle,
        lambda_a_raw,
        lambda_b_raw,
        lambda_a,
        lambda_b,
        model,
        {"angle_deg": angle, "Br_T": br_slot, "Bt_T": bt_slot},
        slotless_femm,
        slotted_on_grid,
    )

    print("=== Paper [5] Eq. (42)-(43) empirical slotting correction ===")
    print(f"model CSV: {args.model_csv}")
    print(f"corrected Br rms = {stats(br_slot)['rms']:.6g} T; Bt rms = {stats(bt_slot)['rms']:.6g} T")
    print(f"slotted FEMM Br rms = {stats(slotted_on_grid['Br_T'])['rms']:.6g} T; Bt rms = {stats(slotted_on_grid['Bt_T'])['rms']:.6g} T")
    print(f"relative L2: Br = {summary['Br_relative_l2']:.6g}; Bt = {summary['Bt_relative_l2']:.6g}")
    print(f"Field CSV: {field_csv}")
    print(f"Lambda CSV: {lambda_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Lambda plot: {lambda_png}")
    print(f"Field plot: {field_png}")


if __name__ == "__main__":
    main()

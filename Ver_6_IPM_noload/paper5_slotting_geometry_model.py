import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
from paper5_slotting_correction import apply_slotting, interp_to, read_field_csv
from paper5_vshape_stage1 import relative_l2, stats


DEFAULT_MODEL = "results/paper5_structure2_alpha52_offset30_h7/structure2_br_bt.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_SLOTLESS_FEMM = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"
DEFAULT_RESULT_DIR = "results/paper5_slotting_geometry_alpha52"


def angular_distance_deg(angle, center, period):
    return (angle - center + 0.5 * period) % period - 0.5 * period


def slot_centers_deg(spec, stator_rotation_deg, sector_start_deg, sector_end_deg):
    pitch = 360.0 / spec.slots
    centers = []
    for idx in range(spec.slots):
        theta = stator_rotation_deg + idx * pitch
        while theta < sector_start_deg:
            theta += 360.0
        while theta > sector_end_deg:
            theta -= 360.0
        if sector_start_deg <= theta <= sector_end_deg:
            centers.append(theta)
    return np.array(sorted(centers), dtype=float)


def raised_cosine_notch(distance, half_width):
    x = np.abs(distance) / half_width
    values = np.zeros_like(distance)
    mask = x < 1.0
    values[mask] = 0.5 * (1.0 + np.cos(np.pi * x[mask]))
    return values


def gaussian_notch(distance, sigma):
    return np.exp(-0.5 * (distance / sigma) ** 2)


def geometry_lambda(
    angle_deg,
    slots,
    slot_opening_deg,
    stator_rotation_deg,
    sector_start_deg,
    sector_end_deg,
    lambda_drop,
    lambda_b_gain,
    width_scale,
    notch_kind,
    normalize_mean,
):
    spec = base.PAPER_SPECS["vshape"]
    pitch = 360.0 / slots
    centers = slot_centers_deg(spec, stator_rotation_deg, sector_start_deg, sector_end_deg)
    half_width = 0.5 * slot_opening_deg * width_scale
    sigma = half_width / 1.35 if half_width > 0.0 else 1.0

    slot_sum = np.zeros_like(angle_deg)
    derivative_sum = np.zeros_like(angle_deg)
    for center in centers:
        distance = angular_distance_deg(angle_deg, center, pitch)
        if notch_kind == "raised-cosine":
            notch = raised_cosine_notch(distance, half_width)
            deriv = np.zeros_like(angle_deg)
            mask = np.abs(distance) < half_width
            deriv[mask] = -0.5 * np.pi / half_width * np.sin(np.pi * np.abs(distance[mask]) / half_width)
            deriv[mask] *= np.sign(distance[mask])
        else:
            notch = gaussian_notch(distance, sigma)
            deriv = -(distance / (sigma * sigma)) * notch
        slot_sum += notch
        derivative_sum += deriv

    if np.max(slot_sum) > 0.0:
        slot_sum /= np.max(slot_sum)
    max_abs_deriv = np.max(np.abs(derivative_sum))
    if max_abs_deriv > 0.0:
        derivative_sum /= max_abs_deriv

    lambda_a = 1.0 - lambda_drop * slot_sum
    if normalize_mean:
        lambda_a /= np.mean(lambda_a)
    lambda_b = lambda_b_gain * derivative_sum
    return lambda_a, lambda_b, centers


def closed_form_slot_parameters():
    spec = base.PAPER_SPECS["vshape"]
    airgap_mm = spec.stator_inner_radius - spec.rotor_outer_radius
    slot_opening_mm = spec.stator_inner_radius * spec.slot_opening_span
    slot_top_depth_mm = spec.slot_top_radius - spec.stator_inner_radius
    slot_pitch_mm = 2.0 * math.pi * spec.stator_inner_radius / spec.slots

    # Geometry-only fringing estimate. The slot mouth does not act like the full
    # slot-top depth; scale it by opening/(opening + airgap) to account for
    # fringing at the mouth without using any FEMM field data.
    fringing_factor = slot_opening_mm / (slot_opening_mm + airgap_mm)
    effective_slot_depth_mm = slot_top_depth_mm * fringing_factor
    lambda_min = airgap_mm / (airgap_mm + effective_slot_depth_mm)
    lambda_drop = max(0.0, min(0.75, 1.0 - lambda_min))

    # The quadrature component is a fringing/skew term. Use a bounded fraction of
    # the radial notch depth based only on the same slot opening/airgap geometry.
    lambda_b_gain = 0.5 * lambda_drop * slot_opening_mm / (slot_opening_mm + 0.5 * airgap_mm)

    width_scale = max(0.7, min(1.8, slot_pitch_mm / (slot_pitch_mm + slot_opening_mm)))
    return {
        "lambda_drop": lambda_drop,
        "lambda_b_gain": lambda_b_gain,
        "width_scale": width_scale,
        "airgap_mm": airgap_mm,
        "slot_opening_mm": slot_opening_mm,
        "slot_top_depth_mm": slot_top_depth_mm,
        "slot_pitch_mm": slot_pitch_mm,
        "fringing_factor": fringing_factor,
        "effective_slot_depth_mm": effective_slot_depth_mm,
    }


def write_field(path, angle, br, bt):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for row in zip(angle, br, bt):
            writer.writerow(row)
    return output


def write_lambda(path, angle, lambda_a, lambda_b):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "lambda_a", "lambda_b"])
        for row in zip(angle, lambda_a, lambda_b):
            writer.writerow(row)
    return output


def write_summary(path, values):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        for key, value in values.items():
            writer.writerow([key, value])
    return output


def plot_results(out_dir, angle, model, corrected, slotted, slotless, lambda_a, lambda_b, centers):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    field_path = out / "geometry_slotting_field_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, component, ylabel in (
        (axes[0], "Br_T", "Br (T)"),
        (axes[1], "Bt_T", "Bt (T)"),
    ):
        if slotted is not None:
            ax.plot(angle, slotted[component], color="black", linewidth=2.0, label="slotted FEMM")
        if slotless is not None:
            ax.plot(angle, slotless[component], color="0.55", linestyle="--", linewidth=1.2, label="slotless FEMM")
        ax.plot(angle, model[component], color="#ff7f0e", linewidth=1.2, label="subdomain slotless")
        ax.plot(angle, corrected[component], color="#2ca02c", linewidth=1.7, label="subdomain + geometry slotting")
        for center in centers:
            ax.axvline(center, color="0.85", linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("Mechanical angle (deg)")
    fig.tight_layout()
    fig.savefig(field_path, dpi=220)
    plt.close(fig)

    lambda_path = out / "geometry_lambda.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(angle, lambda_a, color="#1f77b4", label="lambda_a geometry")
    axes[1].plot(angle, lambda_b, color="#d62728", label="lambda_b geometry")
    for ax in axes:
        for center in centers:
            ax.axvline(center, color="0.85", linewidth=0.7)
        ax.grid(True, alpha=0.28)
        ax.legend()
    axes[1].set_xlabel("Mechanical angle (deg)")
    fig.tight_layout()
    fig.savefig(lambda_path, dpi=220)
    plt.close(fig)
    return field_path, lambda_path


def read_on_grid(path, angle):
    data = read_field_csv(path)
    return {
        "angle_deg": angle,
        "Br_T": interp_to(data["angle_deg"], data["Br_T"], angle),
        "Bt_T": interp_to(data["angle_deg"], data["Bt_T"], angle),
    }


def evaluate_case(args, lambda_drop, lambda_b_gain, width_scale, result_dir=None):
    model0 = read_field_csv(args.model_csv)
    angle = model0["angle_deg"]
    model = {"angle_deg": angle, "Br_T": model0["Br_T"], "Bt_T": model0["Bt_T"]}
    slotted = None if args.no_femm else read_on_grid(args.slotted_femm, angle)
    slotless = None if args.no_femm else read_on_grid(args.slotless_femm, angle)
    closed_form = {}
    if args.slot_params == "closed-form":
        closed_form = closed_form_slot_parameters()
        lambda_drop = closed_form["lambda_drop"]
        lambda_b_gain = closed_form["lambda_b_gain"]
        width_scale = closed_form["width_scale"]
    slot_opening_deg = math.degrees(base.PAPER_SPECS["vshape"].slot_opening_span)
    lambda_a, lambda_b, centers = geometry_lambda(
        angle,
        base.PAPER_SPECS["vshape"].slots,
        slot_opening_deg,
        args.stator_rotation_deg,
        args.sector_start_deg,
        args.sector_end_deg,
        lambda_drop,
        lambda_b_gain,
        width_scale,
        args.notch_kind,
        args.normalize_mean,
    )
    br, bt = apply_slotting(model, lambda_a, lambda_b)
    corrected = {"angle_deg": angle, "Br_T": br, "Bt_T": bt}
    summary = {
        "Br_relative_l2": "" if slotted is None else relative_l2(br, slotted["Br_T"]),
        "Bt_relative_l2": "" if slotted is None else relative_l2(bt, slotted["Bt_T"]),
        "Br_rms": stats(br)["rms"],
        "Bt_rms": stats(bt)["rms"],
        "slotted_FEMM_Br_rms": "" if slotted is None else stats(slotted["Br_T"])["rms"],
        "slotted_FEMM_Bt_rms": "" if slotted is None else stats(slotted["Bt_T"])["rms"],
        "lambda_drop": lambda_drop,
        "lambda_b_gain": lambda_b_gain,
        "width_scale": width_scale,
        "slot_params": args.slot_params,
        "notch_kind": args.notch_kind,
        "normalize_mean": args.normalize_mean,
        "slot_opening_deg": slot_opening_deg,
        "slot_centers_deg": ";".join(f"{value:.6g}" for value in centers),
        "model_csv": args.model_csv,
    }
    summary.update({f"closed_form_{key}": value for key, value in closed_form.items()})
    if result_dir is not None:
        out = Path(result_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_field(out / "geometry_slotting_br_bt.csv", angle, br, bt)
        write_lambda(out / "geometry_lambda.csv", angle, lambda_a, lambda_b)
        write_summary(out / "geometry_slotting_summary.csv", summary)
        field_path, lambda_path = plot_results(out, angle, model, corrected, slotted, slotless, lambda_a, lambda_b, centers)
        summary["field_plot"] = field_path
        summary["lambda_plot"] = lambda_path
    return summary


def parse_range(text):
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        return parts
    if len(parts) == 3:
        start, stop, step = parts
        values = []
        value = start
        while value <= stop + 1e-12:
            values.append(value)
            value += step
        return values
    raise argparse.ArgumentTypeError("range must be one value or start,stop,step")


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry-only slotting permeance model for paper [5].")
    parser.add_argument("--model-csv", default=DEFAULT_MODEL)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--slotless-femm", default=DEFAULT_SLOTLESS_FEMM)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--stator-rotation-deg", type=float, default=base.STATOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--sector-start-deg", type=float, default=0.0)
    parser.add_argument("--sector-end-deg", type=float, default=60.0)
    parser.add_argument("--lambda-drop", type=float, default=0.32)
    parser.add_argument("--lambda-b-gain", type=float, default=0.14)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--slot-params", choices=("manual", "closed-form"), default="manual")
    parser.add_argument("--notch-kind", choices=("gaussian", "raised-cosine"), default="gaussian")
    parser.add_argument("--normalize-mean", dest="normalize_mean", action="store_true")
    parser.add_argument("--no-normalize-mean", dest="normalize_mean", action="store_false")
    parser.set_defaults(normalize_mean=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--no-femm", action="store_true")
    parser.add_argument("--drop-range", type=parse_range, default=parse_range("0.08,0.36,0.02"))
    parser.add_argument("--b-gain-range", type=parse_range, default=parse_range("-0.22,0.22,0.02"))
    parser.add_argument("--width-range", type=parse_range, default=parse_range("0.8,3.0,0.2"))
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        if args.no_femm:
            raise ValueError("--sweep needs FEMM comparison; omit --no-femm or run a single closed-form case.")
        if args.slot_params == "closed-form":
            raise ValueError("--sweep is for manual parameters; closed-form has no sweep variables.")
        rows = []
        for width in args.width_range:
            for drop in args.drop_range:
                for gain in args.b_gain_range:
                    rows.append(evaluate_case(args, drop, gain, width))
        rows = sorted(rows, key=lambda row: row["Br_relative_l2"] + row["Bt_relative_l2"])
        with open(out / "geometry_slotting_sweep.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        best = rows[0]
        best_dir = out / "best"
        evaluate_case(
            args,
            best["lambda_drop"],
            best["lambda_b_gain"],
            best["width_scale"],
            best_dir,
        )
        print("=== Geometry-only slotting sweep ===")
        print(
            f"best: drop={best['lambda_drop']:.6g}, b_gain={best['lambda_b_gain']:.6g}, "
            f"width={best['width_scale']:.6g}"
        )
        print(f"relative L2: Br={best['Br_relative_l2']:.6g}, Bt={best['Bt_relative_l2']:.6g}")
        print(f"Sweep: {out / 'geometry_slotting_sweep.csv'}")
        print(f"Best dir: {best_dir}")
    else:
        summary = evaluate_case(args, args.lambda_drop, args.lambda_b_gain, args.width_scale, out)
        print("=== Geometry-only slotting ===")
        print(
            f"drop={summary['lambda_drop']:.6g}, b_gain={summary['lambda_b_gain']:.6g}, "
            f"width={summary['width_scale']:.6g}"
        )
        if summary["Br_relative_l2"] == "":
            print("relative L2: skipped (--no-femm)")
        else:
            print(f"relative L2: Br={summary['Br_relative_l2']:.6g}, Bt={summary['Bt_relative_l2']:.6g}")
        print(f"Field plot: {summary['field_plot']}")
        print(f"Lambda plot: {summary['lambda_plot']}")


if __name__ == "__main__":
    main()

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
import V_shape_equavalent_br_bt_export as exporter
import paper5_structure2_solver as s2
from paper5_loaded_torque_from_br import torque_from_br_case
from paper5_pole_edge_correction import apply_edge_correction, closed_form_params as edge_closed_form
from paper5_slotting_correction import apply_slotting, interp_to, read_field_csv
from paper5_slotting_geometry_model import closed_form_slot_parameters, geometry_lambda
from paper5_vshape_stage1 import relative_l2, stats


DEFAULT_RESULT_DIR = "results/paper5_radial_shift_validation"
DEFAULT_SHIFTS = "0.7,1.2,1.7,2.2,2.7"
DEFAULT_CURRENT_RMS_A = 17.6
DEFAULT_TORQUE_ANGLE_RANGE = "0,360,2.5"


def parse_list(text):
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_range(text):
    parts = parse_list(text)
    if len(parts) == 1:
        return parts
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("range must be one value or start,stop,step")
    start, stop, step = parts
    values = []
    value = start
    while value <= stop + 0.5 * step:
        values.append(value)
        value += step
    return values


def safe_shift_name(shift):
    text = f"{shift:.6g}".replace("-", "m").replace(".", "p")
    return f"shift_{text}"


def write_field(path, angle, br, bt):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for row in zip(angle, br, bt):
            writer.writerow(row)
    return output


def write_rows(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return output
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_and_export_femm(shift, case_dir, args):
    fem_path = case_dir / f"paper_ipm_vshape_equivalent_{safe_shift_name(shift)}.FEM"
    equivalent.build_model(
        output_path=fem_path,
        current_rms_a=0.0,
        equivalent_radial_shift_mm=shift,
        turns_per_layer=base.TURNS_PER_LAYER_DEFAULT,
    )
    _, _, csv_path = exporter.solve_and_export_br_bt(
        fem_path=fem_path,
        result_dir=case_dir / "femm_equivalent",
        current_rms_a=0.0,
        current_angle_deg=0.0,
        airgap_radius_mm=args.airgap_radius_mm,
        arc_start_deg=0.0,
        arc_end_deg=60.0,
        arc_sample_margin_deg=0.01,
        num_field_points=301,
        periodic_multiplier=6,
        torque_groups=exporter.DEFAULT_TORQUE_GROUPS,
        use_existing_solution=False,
    )
    return fem_path, Path(csv_path)


def gap_permeance_gain(params, args):
    if args.gap_permeance_power == 0.0:
        return 1.0
    spec = base.PAPER_SPECS["vshape"]
    ref_ns = SimpleNamespace(
        alpha_deg=args.alpha_deg,
        w1_mm=equivalent.EQUIVALENT_W1_MM_DEFAULT,
        w2_mm=equivalent.EQUIVALENT_W2_MM_DEFAULT,
        wb1_mm=None,
        hb1_mm=None,
        wb2_mm=None,
        equivalent_radial_shift_mm=args.gap_reference_shift_mm,
        brem_t=spec.magnet_remanence_t,
        mu_pm=1.05,
    )
    ref_params = s2.build_params(ref_ns)
    gap = s2.mm_to_m(args.airgap_radius_mm) - params.rf_outer
    gap_ref = s2.mm_to_m(args.airgap_radius_mm) - ref_params.rf_outer
    if gap <= 0.0 or gap_ref <= 0.0:
        raise ValueError(f"Invalid PM-to-evaluation gap: gap={gap}, gap_ref={gap_ref}")
    return (gap_ref / gap) ** args.gap_permeance_power


def slot_parameters(args):
    if args.slot_params == "closed-form":
        params = closed_form_slot_parameters()
    elif args.slot_params == "manual":
        params = {
            "lambda_drop": args.lambda_drop,
            "lambda_b_gain": args.lambda_b_gain,
            "width_scale": args.width_scale,
        }
    else:
        raise ValueError(f"Unsupported slot parameter mode: {args.slot_params}")
    return params


def edge_parameters(args):
    if args.edge_params == "closed-form":
        params = edge_closed_form()
    elif args.edge_params == "manual":
        params = {
            "edge_width_deg": args.edge_width_deg,
            "window_power": args.edge_window_power,
            "edge_bt_gain": args.edge_bt_gain,
        }
    else:
        raise ValueError(f"Unsupported edge parameter mode: {args.edge_params}")
    return params


def analytical_filename(args):
    label = args.analytical_label.strip()
    if label:
        safe = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    elif args.slot_params == "manual" or args.edge_params == "manual" or args.alpha_deg is not None:
        safe = "analytical_balanced"
    else:
        safe = "analytical_br_best"
    return f"{safe}.csv"


def solve_analytical(shift, femm_field, case_dir, args):
    spec = base.PAPER_SPECS["vshape"]
    ns = SimpleNamespace(
        alpha_deg=args.alpha_deg,
        w1_mm=equivalent.EQUIVALENT_W1_MM_DEFAULT,
        w2_mm=equivalent.EQUIVALENT_W2_MM_DEFAULT,
        wb1_mm=None,
        hb1_mm=None,
        wb2_mm=None,
        equivalent_radial_shift_mm=shift,
        brem_t=spec.magnet_remanence_t,
        mu_pm=1.05,
    )
    params = s2.build_params(ns)
    sol, idx, cond = s2.solve_structure2(params, args.n_harmonics, args.m_harmonics, args.k_harmonics)
    angle = femm_field["angle_deg"]
    theta = np.radians(angle + args.theta_offset_deg)
    br0, bt0 = s2.evaluate_airgap(params, sol, idx, theta, s2.mm_to_m(args.airgap_radius_mm))
    slotless = {"angle_deg": angle, "Br_T": br0, "Bt_T": bt0}

    slot_params = slot_parameters(args)
    slot_opening_deg = math.degrees(spec.slot_opening_span)
    lambda_a, lambda_b, _ = geometry_lambda(
        angle,
        spec.slots,
        slot_opening_deg,
        base.STATOR_ROTATION_DEG_DEFAULT,
        0.0,
        60.0,
        slot_params["lambda_drop"],
        slot_params["lambda_b_gain"],
        slot_params["width_scale"],
        "gaussian",
        True,
    )
    br_slot, bt_slot = apply_slotting(slotless, lambda_a, lambda_b)
    slotted = {"angle_deg": angle, "Br_T": br_slot, "Bt_T": bt_slot}

    edge_params = edge_parameters(args)
    br_edge, bt_edge, _, _ = apply_edge_correction(
        slotted,
        edge_params["edge_width_deg"],
        edge_params["window_power"],
        edge_params["edge_bt_gain"],
        args.bt_mode,
    )
    gain = gap_permeance_gain(params, args)
    br_edge *= gain
    bt_edge *= gain
    analytical_csv = write_field(case_dir / analytical_filename(args), angle, br_edge, bt_edge)
    return {
        "angle_deg": angle,
        "Br_T": br_edge,
        "Bt_T": bt_edge,
        "csv": analytical_csv,
        "condition": cond,
        "rf_outer_mm": params.rf_outer * 1e3,
        "rm_inner_mm": params.rm_inner * 1e3,
        "span_deg": math.degrees(params.span),
        "alpha_deg": math.degrees(params.alpha),
        "alpha1_deg": math.degrees(params.alpha1),
        "gap_permeance_gain": gain,
        "gap_permeance_power": args.gap_permeance_power,
        "gap_reference_shift_mm": args.gap_reference_shift_mm,
        "slot_params": slot_params,
        "edge_params": edge_params,
        "bt_mode": args.bt_mode,
    }


def best_torque(field, args):
    best = None
    for angle in args.torque_angle_range:
        _, torque_total, _ = torque_from_br_case(
            field,
            angle,
            args.current_rms_a,
            base.TURNS_PER_LAYER_DEFAULT,
            6,
            0.0,
            60.0,
            base.STATOR_ROTATION_DEG_DEFAULT,
        )
        if best is None or abs(torque_total) > abs(best["torque_total_Nm"]):
            best = {"angle_deg": angle, "torque_total_Nm": torque_total}
    return best


def plot_trends(out_dir, rows):
    out = Path(out_dir)
    shift = np.array([row["shift_mm"] for row in rows])
    femm_t = np.array([abs(row["femm_torque_estimate_Nm"]) for row in rows])
    code_t = np.array([abs(row["analytical_torque_estimate_Nm"]) for row in rows])
    br_l2 = np.array([row["Br_L2_vs_femm_equivalent"] for row in rows])
    br_rms_femm = np.array([row["femm_Br_rms"] for row in rows])
    br_rms_code = np.array([row["analytical_Br_rms"] for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(shift, femm_t, marker="o", label="FEMM equivalent Br torque estimate")
    axes[0].plot(shift, code_t, marker="o", label="analytical Br torque estimate")
    axes[0].set_ylabel("Torque estimate (Nm)")
    axes[0].legend()

    axes[1].plot(shift, br_l2, marker="o", color="#d62728")
    axes[1].set_ylabel("Br L2 code vs FEMM")

    axes[2].plot(shift, br_rms_femm, marker="o", label="FEMM Br RMS")
    axes[2].plot(shift, br_rms_code, marker="o", label="analytical Br RMS")
    axes[2].set_ylabel("Br RMS (T)")
    axes[2].set_xlabel("equivalent_radial_shift_mm")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "radial_shift_validation_trends.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate analytical trend when equivalent_radial_shift_mm changes in both FEMM and subdomain code."
    )
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--shifts", type=parse_list, default=parse_list(DEFAULT_SHIFTS))
    parser.add_argument("--current-rms-a", type=float, default=DEFAULT_CURRENT_RMS_A)
    parser.add_argument("--torque-angle-range", type=parse_range, default=parse_range(DEFAULT_TORQUE_ANGLE_RANGE))
    parser.add_argument("--airgap-radius-mm", type=float, default=39.4)
    parser.add_argument("--theta-offset-deg", type=float, default=30.0)
    parser.add_argument("--n-harmonics", type=int, default=7)
    parser.add_argument("--m-harmonics", type=int, default=7)
    parser.add_argument("--k-harmonics", type=int, default=7)
    parser.add_argument(
        "--alpha-deg",
        type=float,
        default=None,
        help="Override equivalent Structure 2 alpha angle. Leave unset for the equivalent geometry default.",
    )
    parser.add_argument("--analytical-label", default="", help="Filename label for analytical CSVs.")
    parser.add_argument("--slot-params", choices=("closed-form", "manual"), default="closed-form")
    parser.add_argument("--lambda-drop", type=float, default=0.32)
    parser.add_argument("--lambda-b-gain", type=float, default=0.14)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--edge-params", choices=("closed-form", "manual"), default="closed-form")
    parser.add_argument("--edge-width-deg", type=float, default=11.5)
    parser.add_argument("--edge-window-power", type=float, default=1.0)
    parser.add_argument("--edge-bt-gain", type=float, default=0.08)
    parser.add_argument(
        "--bt-mode",
        choices=("untouched", "window", "drive", "window-drive"),
        default="drive",
    )
    parser.add_argument(
        "--gap-permeance-power",
        type=float,
        default=0.0,
        help=(
            "Geometry-only permeance scaling based on PM outer radius to field-evaluation radius. "
            "0 disables it; 1 is first-order inverse-gap scaling."
        ),
    )
    parser.add_argument("--gap-reference-shift-mm", type=float, default=1.7)
    parser.add_argument("--skip-femm", action="store_true", help="Reuse existing FEMM CSVs in result folders.")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for shift in args.shifts:
        case_dir = out / safe_shift_name(shift)
        case_dir.mkdir(parents=True, exist_ok=True)
        femm_csv = case_dir / "femm_equivalent" / "br_bt_arc.csv"
        if args.skip_femm and femm_csv.exists():
            fem_path = case_dir / f"paper_ipm_vshape_equivalent_{safe_shift_name(shift)}.FEM"
        else:
            fem_path, femm_csv = build_and_export_femm(shift, case_dir, args)

        femm_field = read_field_csv(femm_csv)
        analytical = solve_analytical(shift, femm_field, case_dir, args)
        femm_torque = best_torque(femm_field, args)
        analytical_torque = best_torque(analytical, args)

        row = {
            "shift_mm": shift,
            "fem_path": fem_path,
            "femm_csv": femm_csv,
            "analytical_csv": analytical["csv"],
            "condition": analytical["condition"],
            "rf_outer_mm": analytical["rf_outer_mm"],
            "rm_inner_mm": analytical["rm_inner_mm"],
            "span_deg": analytical["span_deg"],
            "alpha_deg": analytical["alpha_deg"],
            "alpha1_deg": analytical["alpha1_deg"],
            "requested_alpha_deg": args.alpha_deg if args.alpha_deg is not None else "",
            "slot_params": args.slot_params,
            "lambda_drop": analytical["slot_params"]["lambda_drop"],
            "lambda_b_gain": analytical["slot_params"]["lambda_b_gain"],
            "width_scale": analytical["slot_params"]["width_scale"],
            "edge_params": args.edge_params,
            "edge_width_deg": analytical["edge_params"]["edge_width_deg"],
            "edge_window_power": analytical["edge_params"]["window_power"],
            "edge_bt_gain": analytical["edge_params"]["edge_bt_gain"],
            "bt_mode": analytical["bt_mode"],
            "gap_permeance_gain": analytical["gap_permeance_gain"],
            "gap_permeance_power": analytical["gap_permeance_power"],
            "gap_reference_shift_mm": analytical["gap_reference_shift_mm"],
            "Br_L2_vs_femm_equivalent": relative_l2(analytical["Br_T"], femm_field["Br_T"]),
            "Bt_L2_vs_femm_equivalent": relative_l2(analytical["Bt_T"], femm_field["Bt_T"]),
            "femm_Br_rms": stats(femm_field["Br_T"])["rms"],
            "analytical_Br_rms": stats(analytical["Br_T"])["rms"],
            "femm_Bt_rms": stats(femm_field["Bt_T"])["rms"],
            "analytical_Bt_rms": stats(analytical["Bt_T"])["rms"],
            "femm_torque_estimate_Nm": femm_torque["torque_total_Nm"],
            "femm_torque_angle_deg": femm_torque["angle_deg"],
            "analytical_torque_estimate_Nm": analytical_torque["torque_total_Nm"],
            "analytical_torque_angle_deg": analytical_torque["angle_deg"],
            "torque_estimate_relative_error": (
                abs(analytical_torque["torque_total_Nm"]) - abs(femm_torque["torque_total_Nm"])
            )
            / abs(femm_torque["torque_total_Nm"]),
        }
        rows.append(row)
        print(
            f"shift={shift:.6g} mm: Br L2={row['Br_L2_vs_femm_equivalent']:.6g}, "
            f"FEMM torque est={row['femm_torque_estimate_Nm']:.6g}, "
            f"code torque est={row['analytical_torque_estimate_Nm']:.6g}, "
            f"torque err={100.0 * row['torque_estimate_relative_error']:.3g}%"
        )

    summary_csv = write_rows(out / "radial_shift_validation_summary.csv", rows)
    trend_plot = plot_trends(out, rows)
    print("=== Radial shift validation ===")
    print(f"Summary: {summary_csv}")
    print(f"Trend plot: {trend_plot}")


if __name__ == "__main__":
    main()

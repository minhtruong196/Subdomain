import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
import paper5_structure1_solver as s1
from paper5_pole_edge_correction import apply_edge_correction
from paper5_slotting_correction import apply_slotting, interp_to, read_field_csv
from paper5_slotting_geometry_model import geometry_lambda
from paper5_vshape_stage1 import relative_l2, stats


DEFAULT_S2 = "results/paper5_structure2_alpha52p25_offset30_h7/structure2_br_bt.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_RESULT_DIR = "results/paper5_structure1_integration_alpha52p25"
DEFAULT_MU_BRIDGE = 14.47815100961945


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


def read_on_grid(path, angle):
    data = read_field_csv(path)
    return {
        "angle_deg": angle,
        "Br_T": interp_to(data["angle_deg"], data["Br_T"], angle),
        "Bt_T": interp_to(data["angle_deg"], data["Bt_T"], angle),
    }


def solve_s1(args):
    ns = SimpleNamespace(
        alpha_deg=args.alpha_deg,
        w1_mm=equivalent.EQUIVALENT_W1_MM_DEFAULT,
        w2_mm=equivalent.EQUIVALENT_W2_MM_DEFAULT,
        wb1_mm=None,
        hb1_mm=None,
        wb2_mm=None,
        equivalent_radial_shift_mm=equivalent.EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
        brem_t=base.PAPER_SPECS["vshape"].magnet_remanence_t,
        mu_pm=args.mu_pm,
        mu_bridge=args.mu_bridge,
    )
    params = s1.build_params(ns)
    mat, vec, idx = s1.assemble_system(params, args.n_harmonics, args.k_harmonics, args.g_harmonics)
    if args.solver == "direct":
        sol = np.linalg.solve(mat, vec)
        cond = np.linalg.cond(mat)
        return params, sol, idx, cond

    row_scale = np.maximum(np.max(np.abs(mat), axis=1), np.abs(vec))
    row_scale[row_scale == 0.0] = 1.0
    mat_r = mat / row_scale[:, None]
    vec_r = vec / row_scale
    col_scale = np.max(np.abs(mat_r), axis=0)
    col_scale[col_scale == 0.0] = 1.0
    mat_rc = mat_r / col_scale[None, :]
    z, *_ = np.linalg.lstsq(mat_rc, vec_r, rcond=args.lstsq_rcond)
    sol = z / col_scale
    cond = np.linalg.cond(mat_rc)
    return params, sol, idx, cond


def evaluate_s1(params, sol, idx, angle, theta_offset_deg, apply_s1_correction, correction_center_deg):
    theta = np.radians(angle + theta_offset_deg)
    br, bt = s1.evaluate_airgap(params, sol, idx, theta, s1.mm_to_m(39.4))
    if apply_s1_correction:
        br, bt = s1.apply_structure1_correction(
            params,
            np.radians(angle),
            br,
            bt,
            math.radians(correction_center_deg),
            1.0,
        )
    return br, bt


def parse_range(text):
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
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


def metrics(angle, br, bt, ref):
    masks = {
        "all": np.ones_like(angle, dtype=bool),
        "edge": (angle <= 12.0) | (angle >= 48.0),
        "middle": (angle > 12.0) & (angle < 48.0),
    }
    out = {
        "Br_rms": stats(br)["rms"],
        "Bt_rms": stats(bt)["rms"],
    }
    for name, mask in masks.items():
        out[f"Br_L2_{name}"] = relative_l2(br[mask], ref["Br_T"][mask])
        out[f"Bt_L2_{name}"] = relative_l2(bt[mask], ref["Bt_T"][mask])
    return out


def plot_case(path, angle, ref, base_model, best_model, s1_field, title):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, comp, ylabel in (
        (axes[0], "Br_T", "Br (T)"),
        (axes[1], "Bt_T", "Bt (T)"),
    ):
        ax.plot(angle, ref[comp], color="black", linewidth=2.0, label="slotted FEMM validation")
        ax.plot(angle, base_model[comp], color="#1f77b4", linewidth=1.4, label="baseline without S1")
        ax.plot(angle, best_model[comp], color="#2ca02c", linewidth=1.8, label="with Structure 1 diagnostic")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    axes[2].plot(angle, s1_field["Br_T"], label="S1 Br before scale")
    axes[2].plot(angle, s1_field["Bt_T"], label="S1 Bt before scale")
    axes[2].set_ylabel("S1 field (T)")
    axes[2].set_xlabel("Mechanical angle (deg)")
    axes[2].grid(True, alpha=0.28)
    axes[2].legend(fontsize=8)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def make_pipeline(angle, slotless, args):
    slot_opening_deg = math.degrees(base.PAPER_SPECS["vshape"].slot_opening_span)
    lambda_a, lambda_b, _ = geometry_lambda(
        angle,
        base.PAPER_SPECS["vshape"].slots,
        slot_opening_deg,
        base.STATOR_ROTATION_DEG_DEFAULT,
        0.0,
        60.0,
        args.lambda_drop,
        args.lambda_b_gain,
        args.width_scale,
        "gaussian",
        True,
    )
    br_slot, bt_slot = apply_slotting(slotless, lambda_a, lambda_b)
    slotted = {"angle_deg": angle, "Br_T": br_slot, "Bt_T": bt_slot}
    br_edge, bt_edge, _, _ = apply_edge_correction(
        slotted,
        args.edge_width_deg,
        1.0,
        args.edge_bt_gain,
        "drive",
    )
    return {"angle_deg": angle, "Br_T": br_edge, "Bt_T": bt_edge}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostic integration of Structure 1 into the current S2/slot/edge pipeline.")
    parser.add_argument("--structure2-csv", default=DEFAULT_S2)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--alpha-deg", type=float, default=52.25)
    parser.add_argument("--mu-bridge", type=float, default=DEFAULT_MU_BRIDGE)
    parser.add_argument("--mu-pm", type=float, default=1.05)
    parser.add_argument("--n-harmonics", type=int, default=3)
    parser.add_argument("--k-harmonics", type=int, default=3)
    parser.add_argument("--g-harmonics", type=int, default=3)
    parser.add_argument("--solver", choices=("direct", "rowcol"), default="rowcol")
    parser.add_argument("--lstsq-rcond", type=float, default=1e-12)
    parser.add_argument("--offset-range", type=parse_range, default=parse_range("-90,90,5"))
    parser.add_argument("--scale-range", type=parse_range, default=parse_range("-5,5,0.25"))
    parser.add_argument("--lambda-drop", type=float, default=0.32)
    parser.add_argument("--lambda-b-gain", type=float, default=0.14)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--edge-width-deg", type=float, default=11.5)
    parser.add_argument("--edge-bt-gain", type=float, default=0.08)
    parser.add_argument("--correction-center-deg", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    s2_data = read_field_csv(args.structure2_csv)
    angle = s2_data["angle_deg"]
    ref = read_on_grid(args.slotted_femm, angle)
    params, sol, idx, cond = solve_s1(args)

    baseline = make_pipeline(angle, s2_data, args)
    baseline_metrics = metrics(angle, baseline["Br_T"], baseline["Bt_T"], ref)

    rows = []
    best = None
    for apply_corr in (False, True):
        for offset in args.offset_range:
            s1_br, s1_bt = evaluate_s1(
                params,
                sol,
                idx,
                angle,
                offset,
                apply_corr,
                args.correction_center_deg,
            )
            for scale in args.scale_range:
                slotless = {
                    "angle_deg": angle,
                    "Br_T": s2_data["Br_T"] + scale * s1_br,
                    "Bt_T": s2_data["Bt_T"] + scale * s1_bt,
                }
                final = make_pipeline(angle, slotless, args)
                row = {
                    "apply_s1_correction": apply_corr,
                    "s1_theta_offset_deg": offset,
                    "s1_scale": scale,
                    "s1_condition": cond,
                    "solver": args.solver,
                    **metrics(angle, final["Br_T"], final["Bt_T"], ref),
                }
                row["objective"] = row["Br_L2_all"] + row["Bt_L2_all"] + max(0.0, row["Br_L2_all"] - 0.10) * 2.0
                rows.append(row)
                if best is None or row["objective"] < best["row"]["objective"]:
                    best = {
                        "row": row,
                        "field": final,
                        "s1": {"angle_deg": angle, "Br_T": s1_br, "Bt_T": s1_bt},
                    }

    rows.sort(key=lambda row: row["objective"])
    write_rows(out / "structure1_integration_sweep.csv", rows)
    write_rows(out / "structure1_integration_best_summary.csv", [best["row"]])
    write_field(out / "baseline_without_s1.csv", angle, baseline["Br_T"], baseline["Bt_T"])
    write_field(out / "best_with_s1.csv", angle, best["field"]["Br_T"], best["field"]["Bt_T"])
    plot = plot_case(
        out / "structure1_integration_best.png",
        angle,
        ref,
        baseline,
        best["field"],
        best["s1"],
        "Structure 1 integration diagnostic",
    )

    print("=== Structure 1 integration diagnostic ===")
    print(f"S1 matrix condition ({args.solver}) = {cond:.6g}")
    print(
        "baseline: "
        f"Br={baseline_metrics['Br_L2_all']:.6g}, Bt={baseline_metrics['Bt_L2_all']:.6g}"
    )
    print(
        "best with S1: "
        f"Br={best['row']['Br_L2_all']:.6g}, Bt={best['row']['Bt_L2_all']:.6g}, "
        f"scale={best['row']['s1_scale']:.6g}, offset={best['row']['s1_theta_offset_deg']:.6g}, "
        f"correction={best['row']['apply_s1_correction']}"
    )
    print(f"Sweep: {out / 'structure1_integration_sweep.csv'}")
    print(f"Best summary: {out / 'structure1_integration_best_summary.csv'}")
    print(f"Plot: {plot}")


if __name__ == "__main__":
    main()

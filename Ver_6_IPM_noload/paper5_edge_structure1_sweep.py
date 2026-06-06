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
from paper5_slotting_correction import (
    apply_slotting,
    interp_to,
    read_field_csv,
    smooth_lambda,
    solve_lambda,
)
from paper5_vshape_stage1 import relative_l2, stats


DEFAULT_RESULT_DIR = "results/paper5_edge_structure1_sweep"
DEFAULT_SLOTLESS_FEMM = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_S2 = "results/paper5_structure2_offset30_h7/structure2_br_bt.csv"
DEFAULT_MU_BRIDGE = 14.47815100961945


def read_model(path, angle):
    data = read_field_csv(path)
    return {
        "angle_deg": angle,
        "Br_T": interp_to(data["angle_deg"], data["Br_T"], angle),
        "Bt_T": interp_to(data["angle_deg"], data["Bt_T"], angle),
    }


def region_masks(angle):
    left = (0.0 <= angle) & (angle <= 12.0)
    middle = (12.0 < angle) & (angle < 48.0)
    right = (48.0 <= angle) & (angle <= 60.0)
    edge = left | right
    return {
        "left": left,
        "middle": middle,
        "right": right,
        "edge": edge,
        "all": np.ones_like(angle, dtype=bool),
    }


def vectorize(br, bt, mask, include_bt):
    if include_bt:
        return np.concatenate([br[mask], bt[mask]])
    return br[mask]


def solve_s1_once(args):
    ns = SimpleNamespace(
        alpha_deg=args.alpha_deg,
        w1_mm=args.w1_mm,
        w2_mm=args.w2_mm,
        wb1_mm=args.wb1_mm,
        hb1_mm=args.hb1_mm,
        wb2_mm=args.wb2_mm,
        equivalent_radial_shift_mm=args.equivalent_radial_shift_mm,
        brem_t=args.brem_t,
        mu_pm=args.mu_pm,
        mu_bridge=args.mu_bridge,
    )
    params = s1.build_params(ns)
    mat, vec, idx = s1.assemble_system(params, args.n_harmonics, args.k_harmonics, args.g_harmonics)
    if args.solver_scaling == "none":
        sol = np.linalg.solve(mat, vec)
        cond = np.linalg.cond(mat)
    else:
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


def evaluate_s1(params, sol, idx, angle_deg, theta_offset_deg, correction_center_deg, apply_correction):
    theta_eval = np.radians(angle_deg + theta_offset_deg)
    br, bt = s1.evaluate_airgap(params, sol, idx, theta_eval, s1.mm_to_m(39.4))
    if apply_correction:
        br, bt = s1.apply_structure1_correction(
            params,
            np.radians(angle_deg),
            br,
            bt,
            math.radians(correction_center_deg),
            1.0,
        )
    return br, bt


def fit_one_scale(base_br, base_bt, comp_br, comp_bt, ref_br, ref_bt, mask, include_bt):
    residual = vectorize(ref_br - base_br, ref_bt - base_bt, mask, include_bt)
    comp = vectorize(comp_br, comp_bt, mask, include_bt)
    denom = float(np.dot(comp, comp))
    if denom == 0.0:
        return 0.0
    return float(np.dot(comp, residual) / denom)


def fit_two_scales(base1_br, base1_bt, base2_br, base2_bt, ref_br, ref_bt, mask, include_bt):
    y = vectorize(ref_br, ref_bt, mask, include_bt)
    x1 = vectorize(base1_br, base1_bt, mask, include_bt)
    x2 = vectorize(base2_br, base2_bt, mask, include_bt)
    mat = np.vstack([x1, x2]).T
    coeff, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coeff[0]), float(coeff[1])


def score(angle, br, bt, ref_br, ref_bt):
    masks = region_masks(angle)
    out = {}
    for name, mask in masks.items():
        out[f"Br_L2_{name}"] = relative_l2(br[mask], ref_br[mask])
        out[f"Bt_L2_{name}"] = relative_l2(bt[mask], ref_bt[mask])
    out["Br_rms"] = stats(br)["rms"]
    out["Bt_rms"] = stats(bt)["rms"]
    return out


def write_summary(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return output
    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_field(path, angle, br, bt):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for row in zip(angle, br, bt):
            writer.writerow(row)
    return output


def plot_best(path, angle, ref, s2_data, best_slotless, best_slotted, slotted_ref, title):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, component, label in (
        (axes[0], "Br_T", "Br (T)"),
        (axes[1], "Bt_T", "Bt (T)"),
    ):
        ax.plot(angle, ref[component], color="0.45", linestyle="--", linewidth=1.2, label="slotless FEMM")
        ax.plot(angle, s2_data[component], color="#ff7f0e", linewidth=1.2, label="S2 slotless")
        ax.plot(angle, best_slotless[component], color="#1f77b4", linewidth=1.7, label="best S2/S1 slotless diagnostic")
        ax.plot(angle, slotted_ref[component], color="black", linewidth=2.0, label="slotted FEMM")
        ax.plot(angle, best_slotted[component], color="#2ca02c", linewidth=1.7, label="best diagnostic + slotting")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    axes[0].set_title(title)
    axes[1].set_xlabel("Mechanical angle (deg)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic sweep: can Structure 1 fix edge errors when added to Structure 2?"
    )
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--slotless-femm", default=DEFAULT_SLOTLESS_FEMM)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--structure2", default=DEFAULT_S2)
    parser.add_argument("--mu-bridge", type=float, default=DEFAULT_MU_BRIDGE)
    parser.add_argument("--n-harmonics", type=int, default=3)
    parser.add_argument("--k-harmonics", type=int, default=3)
    parser.add_argument("--g-harmonics", type=int, default=3)
    parser.add_argument("--offset-start", type=float, default=-90.0)
    parser.add_argument("--offset-end", type=float, default=90.0)
    parser.add_argument("--offset-step", type=float, default=1.0)
    parser.add_argument("--correction-center-deg", type=float, default=30.0)
    parser.add_argument("--brem-t", type=float, default=base.PAPER_SPECS["vshape"].magnet_remanence_t)
    parser.add_argument("--mu-pm", type=float, default=1.05)
    parser.add_argument("--w1-mm", type=float, default=equivalent.EQUIVALENT_W1_MM_DEFAULT)
    parser.add_argument("--w2-mm", type=float, default=equivalent.EQUIVALENT_W2_MM_DEFAULT)
    parser.add_argument("--equivalent-radial-shift-mm", type=float, default=equivalent.EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT)
    parser.add_argument("--wb1-mm", type=float)
    parser.add_argument("--hb1-mm", type=float)
    parser.add_argument("--wb2-mm", type=float)
    parser.add_argument("--alpha-deg", type=float)
    parser.add_argument("--lambda-harmonics", type=int, default=6)
    parser.add_argument("--solver-scaling", choices=("none", "rowcol"), default="rowcol")
    parser.add_argument("--lstsq-rcond", type=float, default=1e-12)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    slotless = read_field_csv(args.slotless_femm)
    slotted = read_model(args.slotted_femm, slotless["angle_deg"])
    angle = slotless["angle_deg"]
    s2_data = read_model(args.structure2, angle)
    masks = region_masks(angle)

    params, sol, idx, cond = solve_s1_once(args)
    rows = []
    best = None
    offsets = np.arange(args.offset_start, args.offset_end + 0.5 * args.offset_step, args.offset_step)

    for apply_correction in (False, True):
        for offset in offsets:
            br1, bt1 = evaluate_s1(
                params,
                sol,
                idx,
                angle,
                offset,
                args.correction_center_deg,
                apply_correction,
            )
            for fit_kind, include_bt, fit_region in (
                ("edge_Br_only_kS1", False, "edge"),
                ("edge_BrBt_kS1", True, "edge"),
                ("all_BrBt_kS1", True, "all"),
                ("edge_BrBt_cS2_kS1", True, "edge"),
                ("all_BrBt_cS2_kS1", True, "all"),
            ):
                mask = masks[fit_region]
                if fit_kind.endswith("cS2_kS1"):
                    c2, k1 = fit_two_scales(
                        s2_data["Br_T"],
                        s2_data["Bt_T"],
                        br1,
                        bt1,
                        slotless["Br_T"],
                        slotless["Bt_T"],
                        mask,
                        include_bt,
                    )
                    br = c2 * s2_data["Br_T"] + k1 * br1
                    bt = c2 * s2_data["Bt_T"] + k1 * bt1
                else:
                    c2 = 1.0
                    k1 = fit_one_scale(
                        s2_data["Br_T"],
                        s2_data["Bt_T"],
                        br1,
                        bt1,
                        slotless["Br_T"],
                        slotless["Bt_T"],
                        mask,
                        include_bt,
                    )
                    br = s2_data["Br_T"] + k1 * br1
                    bt = s2_data["Bt_T"] + k1 * bt1
                metrics = score(angle, br, bt, slotless["Br_T"], slotless["Bt_T"])
                row = {
                    "fit_kind": fit_kind,
                    "apply_s1_correction": apply_correction,
                    "theta_offset_deg": offset,
                    "structure2_scale": c2,
                    "structure1_scale": k1,
                    "structure1_condition": cond,
                    "solver_scaling": args.solver_scaling,
                    **metrics,
                }
                rows.append(row)
                objective = metrics["Br_L2_edge"] + metrics["Bt_L2_edge"] + 0.35 * (
                    metrics["Br_L2_middle"] + metrics["Bt_L2_middle"]
                )
                if best is None or objective < best["objective"]:
                    best = {
                        "objective": objective,
                        "row": row,
                        "Br_T": br,
                        "Bt_T": bt,
                        "S1_Br_T": br1,
                        "S1_Bt_T": bt1,
                    }

    summary = write_summary(out / "edge_structure1_sweep_summary.csv", rows)
    best_slotless = {"angle_deg": angle, "Br_T": best["Br_T"], "Bt_T": best["Bt_T"]}
    slotless_csv = write_field(out / "best_slotless_s2_s1_diagnostic.csv", angle, best["Br_T"], best["Bt_T"])

    lambda_a_raw, lambda_b_raw = solve_lambda(slotless, slotted)
    lambda_a, lambda_b = smooth_lambda(
        angle,
        lambda_a_raw,
        lambda_b_raw,
        base.PAPER_SPECS["vshape"].slots,
        6,
        args.lambda_harmonics,
    )
    br_slot, bt_slot = apply_slotting(best_slotless, lambda_a, lambda_b)
    slotted_diag = {"angle_deg": angle, "Br_T": br_slot, "Bt_T": bt_slot}
    slotted_csv = write_field(out / "best_slotted_s2_s1_diagnostic.csv", angle, br_slot, bt_slot)
    plot = plot_best(
        out / "best_edge_structure1_diagnostic.png",
        angle,
        slotless,
        s2_data,
        best_slotless,
        slotted_diag,
        slotted,
        "Structure 1 edge-fit diagnostic (not physical correction)",
    )
    slotted_metrics = score(angle, br_slot, bt_slot, slotted["Br_T"], slotted["Bt_T"])
    best_report = {
        **best["row"],
        **{f"slotted_{key}": value for key, value in slotted_metrics.items()},
        "slotless_csv": slotless_csv,
        "slotted_csv": slotted_csv,
        "plot": plot,
        "summary_csv": summary,
    }
    write_summary(out / "best_edge_structure1_diagnostic_summary.csv", [best_report])

    print("=== Structure 1 edge-fit diagnostic ===")
    print(f"Best fit: {best['row']['fit_kind']}")
    print(f"S1 correction: {best['row']['apply_s1_correction']}")
    print(f"theta_offset_deg = {best['row']['theta_offset_deg']:.6g}")
    print(f"structure2_scale = {best['row']['structure2_scale']:.6g}")
    print(f"structure1_scale = {best['row']['structure1_scale']:.6g}")
    print(
        "slotless L2: "
        f"Br edge={best['row']['Br_L2_edge']:.6g}, Br middle={best['row']['Br_L2_middle']:.6g}, "
        f"Bt edge={best['row']['Bt_L2_edge']:.6g}, Bt middle={best['row']['Bt_L2_middle']:.6g}"
    )
    print(
        "slotted L2 after empirical slotting: "
        f"Br edge={slotted_metrics['Br_L2_edge']:.6g}, Br middle={slotted_metrics['Br_L2_middle']:.6g}, "
        f"Bt edge={slotted_metrics['Bt_L2_edge']:.6g}, Bt middle={slotted_metrics['Bt_L2_middle']:.6g}"
    )
    print(f"Summary: {summary}")
    print(f"Best summary: {out / 'best_edge_structure1_diagnostic_summary.csv'}")
    print(f"Plot: {plot}")


if __name__ == "__main__":
    main()

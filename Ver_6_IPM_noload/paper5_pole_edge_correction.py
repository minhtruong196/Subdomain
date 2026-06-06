import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
from paper5_slotting_correction import interp_to, read_field_csv
from paper5_vshape_stage1 import relative_l2, stats


DEFAULT_MODEL = "results/paper5_analytical_only_slotting_closed_form_default_h7/geometry_slotting_br_bt.csv"
DEFAULT_SLOTTED_FEMM = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_RESULT_DIR = "results/paper5_analytical_only_pole_edge_correction"


def write_field(path, angle, br, bt):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for row in zip(angle, br, bt):
            writer.writerow(row)
    return output


def write_summary(path, rows):
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


def equivalent_alpha_deg():
    spec = base.PAPER_SPECS["vshape"]
    dims = equivalent.equivalent_pm_dimensions(
        spec,
        alpha_deg=None,
        w1_mm=equivalent.EQUIVALENT_W1_MM_DEFAULT,
        w2_mm=equivalent.EQUIVALENT_W2_MM_DEFAULT,
        wb1_mm=None,
        hb1_mm=None,
        wb2_mm=None,
        radial_shift_mm=equivalent.EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
    )
    return float(dims["alpha_deg"])


def closed_form_params():
    spec = base.PAPER_SPECS["vshape"]
    pole_pitch_deg = 360.0 / spec.poles
    alpha_deg = equivalent_alpha_deg()
    airgap_mm = spec.stator_inner_radius - spec.rotor_outer_radius
    slot_opening_deg = math.degrees(spec.slot_opening_span)
    airgap_angle_deg = math.degrees(airgap_mm / spec.stator_inner_radius)

    # Heuristic from equivalent pole geometry only: the pole-edge transition is
    # broader than the airgap fringing angle because the V pole is represented by
    # a curved tangential equivalent magnet. The alpha fraction keeps it tied to
    # the actual equivalent pole span instead of fitting a FEMM curve.
    edge_width_deg = max(8.0, min(14.0, 0.22 * alpha_deg))
    window_power = 1.0
    edge_bt_gain = 0.085 * airgap_angle_deg / max(slot_opening_deg, 1e-12)
    edge_bt_gain = max(0.0, min(0.12, edge_bt_gain))
    return {
        "edge_width_deg": edge_width_deg,
        "window_power": window_power,
        "edge_bt_gain": edge_bt_gain,
        "pole_pitch_deg": pole_pitch_deg,
        "alpha_deg": alpha_deg,
        "airgap_mm": airgap_mm,
        "airgap_angle_deg": airgap_angle_deg,
        "slot_opening_deg": slot_opening_deg,
    }


def smooth_ramp(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * x)


def smooth_ramp_derivative_normalized(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * math.pi * np.sin(math.pi * x)


def pole_edge_window(angle_deg, edge_width_deg, sector_start_deg=None, sector_end_deg=None):
    if sector_start_deg is None:
        sector_start_deg = float(np.min(angle_deg))
    if sector_end_deg is None:
        sector_end_deg = float(np.max(angle_deg))

    left_distance = angle_deg - sector_start_deg
    right_distance = sector_end_deg - angle_deg
    left_x = left_distance / edge_width_deg
    right_x = right_distance / edge_width_deg
    left = smooth_ramp(left_x)
    right = smooth_ramp(right_x)
    window = np.minimum(left, right)

    left_drive = smooth_ramp_derivative_normalized(left_x) * (left_distance <= edge_width_deg)
    right_drive = smooth_ramp_derivative_normalized(right_x) * (right_distance <= edge_width_deg)
    edge_drive = right_drive - left_drive
    return window, edge_drive


def apply_edge_correction(model, edge_width_deg, window_power, edge_bt_gain, bt_mode):
    angle = model["angle_deg"]
    window, edge_drive = pole_edge_window(angle, edge_width_deg)
    window = np.power(window, window_power)
    br0 = model["Br_T"]
    bt0 = model["Bt_T"]
    br = br0 * window
    if bt_mode == "untouched":
        bt = bt0.copy()
    elif bt_mode == "window":
        bt = bt0 * window
    elif bt_mode == "drive":
        bt = bt0 + edge_bt_gain * np.maximum(br0, 0.0) * edge_drive
    elif bt_mode == "window-drive":
        bt = bt0 * window + edge_bt_gain * np.maximum(br0, 0.0) * edge_drive
    else:
        raise ValueError(f"Unsupported bt_mode: {bt_mode}")
    return br, bt, window, edge_drive


def region_masks(angle):
    return {
        "left": angle <= 12.0,
        "middle": (angle > 12.0) & (angle < 48.0),
        "right": angle >= 48.0,
        "edge": (angle <= 12.0) | (angle >= 48.0),
        "all": np.ones_like(angle, dtype=bool),
    }


def score(angle, br, bt, ref=None):
    out = {
        "model_Br_rms": stats(br)["rms"],
        "model_Bt_rms": stats(bt)["rms"],
    }
    if ref is None:
        return out
    for name, mask in region_masks(angle).items():
        out[f"Br_L2_{name}"] = relative_l2(br[mask], ref["Br_T"][mask])
        out[f"Bt_L2_{name}"] = relative_l2(bt[mask], ref["Bt_T"][mask])
    return out


def plot_result(path, angle, model, corrected, ref, window, edge_drive):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, component, label in (
        (axes[0], "Br_T", "Br (T)"),
        (axes[1], "Bt_T", "Bt (T)"),
    ):
        if ref is not None:
            ax.plot(angle, ref[component], color="black", linewidth=2.0, label="slotted FEMM validation")
        ax.plot(angle, model[component], color="#ff7f0e", linewidth=1.2, label="input analytical model")
        ax.plot(angle, corrected[component], color="#2ca02c", linewidth=1.8, label="analytical + pole-edge correction")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    axes[2].plot(angle, window, label="edge window")
    axes[2].plot(angle, edge_drive, label="Bt edge drive")
    axes[2].set_ylabel("edge factors")
    axes[2].set_xlabel("Mechanical angle (deg)")
    axes[2].grid(True, alpha=0.28)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def evaluate_case(args, edge_width_deg, window_power, edge_bt_gain, result_dir=None):
    model_raw = read_field_csv(args.model_csv)
    angle = model_raw["angle_deg"]
    model = {
        "angle_deg": angle,
        "Br_T": model_raw["Br_T"],
        "Bt_T": model_raw["Bt_T"],
    }
    ref = None if args.no_femm else read_on_grid(args.slotted_femm, angle)
    br, bt, window, edge_drive = apply_edge_correction(
        model,
        edge_width_deg,
        window_power,
        edge_bt_gain,
        args.bt_mode,
    )
    row = {
        "edge_width_deg": edge_width_deg,
        "window_power": window_power,
        "edge_bt_gain": edge_bt_gain,
        "edge_params": args.edge_params,
        "bt_mode": args.bt_mode,
        **score(angle, br, bt, ref),
    }

    if result_dir is not None:
        out = Path(result_dir)
        out.mkdir(parents=True, exist_ok=True)
        corrected = {"angle_deg": angle, "Br_T": br, "Bt_T": bt}
        row["field_csv"] = write_field(out / "pole_edge_corrected_br_bt.csv", angle, br, bt)
        row["summary_csv"] = write_summary(out / "pole_edge_summary.csv", [row])
        row["plot"] = plot_result(out / "pole_edge_comparison.png", angle, model, corrected, ref, window, edge_drive)
    return row


def parse_float_grid(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry-only pole-edge correction for analytical V-shape IPM field.")
    parser.add_argument("--model-csv", default=DEFAULT_MODEL)
    parser.add_argument("--slotted-femm", default=DEFAULT_SLOTTED_FEMM)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--edge-width-deg", type=float, default=12.0)
    parser.add_argument("--window-power", type=float, default=1.0)
    parser.add_argument("--edge-bt-gain", type=float, default=0.18)
    parser.add_argument("--edge-params", choices=("manual", "closed-form"), default="closed-form")
    parser.add_argument(
        "--bt-mode",
        choices=("untouched", "window", "drive", "window-drive"),
        default="untouched",
    )
    parser.add_argument("--no-femm", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--edge-width-grid", default="8,9,10,11,12,13,14,15,16")
    parser.add_argument("--window-power-grid", default="0.8,1.0,1.2,1.4,1.6")
    parser.add_argument("--edge-bt-gain-grid", default="-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = closed_form_params() if args.edge_params == "closed-form" else {
        "edge_width_deg": args.edge_width_deg,
        "window_power": args.window_power,
        "edge_bt_gain": args.edge_bt_gain,
    }

    if args.sweep:
        if args.no_femm:
            raise ValueError("--sweep needs FEMM validation data; remove --no-femm.")
        rows = []
        for width in parse_float_grid(args.edge_width_grid):
            for power in parse_float_grid(args.window_power_grid):
                for gain in parse_float_grid(args.edge_bt_gain_grid):
                    row = evaluate_case(args, width, power, gain)
                    row["objective"] = row["Br_L2_all"] + 0.35 * row["Bt_L2_all"] + 0.35 * row["Br_L2_edge"]
                    rows.append(row)
        rows.sort(key=lambda row: row["objective"])
        write_summary(out / "pole_edge_sweep.csv", rows)
        best = rows[0]
        best_row = evaluate_case(
            args,
            best["edge_width_deg"],
            best["window_power"],
            best["edge_bt_gain"],
            out,
        )
        print("=== Pole-edge correction sweep ===")
        print(
            f"best width={best_row['edge_width_deg']:.6g}, power={best_row['window_power']:.6g}, "
            f"Bt gain={best_row['edge_bt_gain']:.6g}"
        )
        print(f"relative L2 all: Br={best_row['Br_L2_all']:.6g}, Bt={best_row['Bt_L2_all']:.6g}")
        print(f"relative L2 edge: Br={best_row['Br_L2_edge']:.6g}, Bt={best_row['Bt_L2_edge']:.6g}")
        print(f"Sweep: {out / 'pole_edge_sweep.csv'}")
        print(f"Plot: {best_row['plot']}")
        return

    row = evaluate_case(
        args,
        params["edge_width_deg"],
        params["window_power"],
        params["edge_bt_gain"],
        out,
    )
    print("=== Pole-edge correction ===")
    print(
        f"width={row['edge_width_deg']:.6g}, power={row['window_power']:.6g}, "
        f"Bt gain={row['edge_bt_gain']:.6g}, params={row['edge_params']}"
    )
    if not args.no_femm:
        print(f"relative L2 all: Br={row['Br_L2_all']:.6g}, Bt={row['Bt_L2_all']:.6g}")
        print(f"relative L2 edge: Br={row['Br_L2_edge']:.6g}, Bt={row['Bt_L2_edge']:.6g}")
    print(f"CSV: {row['field_csv']}")
    print(f"Summary: {row['summary_csv']}")
    print(f"Plot: {row['plot']}")


if __name__ == "__main__":
    main()

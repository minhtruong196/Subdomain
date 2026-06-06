import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
from paper5_slotting_correction import interp_to, read_field_csv
from paper_vshape_torque_test import phase_currents_from_rms


MU0 = 4.0 * math.pi * 1e-7
DEFAULT_RESULT_DIR = "results/paper5_loaded_torque_from_br"
DEFAULT_CURRENT_RMS_A = 17.6
DEFAULT_PERIODIC_MULTIPLIER = 6

DEFAULT_CASES = (
    ("FEMM base no-load Br", "results/paper_vshape_1over6/br_bt_arc.csv"),
    ("FEMM equivalent no-load Br", "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"),
    ("Br-best analytical", "results/paper5_analytical_only_pole_edge_drive_closed_form_compare/pole_edge_corrected_br_bt.csv"),
    ("balanced analytical", "results/paper5_alpha52p25_slot_edge_best_compare/pole_edge_corrected_br_bt.csv"),
    ("S1 diagnostic analytical", "results/paper5_structure1_integration_alpha52p25_h5/best_with_s1.csv"),
)


def parse_float_range(text):
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


def parse_case(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("case must be name=csv_path")
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("case must be name=csv_path")
    return name, path


def slot_centers_in_sector(spec, stator_rotation_deg, sector_start_deg, sector_end_deg):
    pitch = 360.0 / spec.slots
    centers = []
    for slot_index in range(spec.slots):
        theta = stator_rotation_deg + slot_index * pitch
        while theta < sector_start_deg:
            theta += 360.0
        while theta > sector_end_deg:
            theta -= 360.0
        if sector_start_deg <= theta <= sector_end_deg:
            centers.append(theta)
    return np.array(sorted(centers), dtype=float)


def slot_winding_index(theta_deg, slots):
    pitch = 360.0 / slots
    return round((theta_deg - 5.0) / pitch) % slots


def layer_currents_for_slot(theta_deg, turns_per_layer, phase_currents):
    slot_idx = slot_winding_index(theta_deg, base.PAPER_SPECS["vshape"].slots)
    (upper_phase, upper_sign), (lower_phase, lower_sign) = base.slot_layer_phases(slot_idx)
    return (
        upper_sign * turns_per_layer * phase_currents[upper_phase],
        lower_sign * turns_per_layer * phase_currents[lower_phase],
        slot_idx,
        upper_phase,
        lower_phase,
    )


def torque_from_br_case(
    field,
    current_angle_deg,
    current_rms_a,
    turns_per_layer,
    periodic_multiplier,
    sector_start_deg,
    sector_end_deg,
    stator_rotation_deg,
):
    spec = base.PAPER_SPECS["vshape"]
    phase_currents = phase_currents_from_rms(current_rms_a, current_angle_deg)
    centers = slot_centers_in_sector(spec, stator_rotation_deg, sector_start_deg, sector_end_deg)

    rsm = 0.5 * (spec.slot_top_radius + spec.slot_bottom_radius)
    upper_r_m = 1e-3 * 0.5 * (spec.slot_top_radius + rsm)
    lower_r_m = 1e-3 * 0.5 * (rsm + spec.slot_bottom_radius)
    stack_m = 1e-3 * spec.stack_length
    br_at_slots = interp_to(field["angle_deg"], field["Br_T"], centers)

    torque_sector = 0.0
    slot_rows = []
    for theta_deg, br, in zip(centers, br_at_slots):
        upper_at, lower_at, slot_idx, upper_phase, lower_phase = layer_currents_for_slot(
            theta_deg,
            turns_per_layer,
            phase_currents,
        )
        slot_torque = stack_m * br * (upper_r_m * upper_at + lower_r_m * lower_at)
        torque_sector += slot_torque
        slot_rows.append(
            {
                "theta_deg": theta_deg,
                "slot_index_zero_based": slot_idx,
                "Br_T": br,
                "upper_phase": upper_phase,
                "lower_phase": lower_phase,
                "upper_ampere_turn": upper_at,
                "lower_ampere_turn": lower_at,
                "slot_torque_sector_Nm": slot_torque,
            }
        )

    return torque_sector, torque_sector * periodic_multiplier, slot_rows


def run_case(name, csv_path, args):
    field = read_field_csv(csv_path)
    rows = []
    best_abs = None
    best_positive = None
    best_negative = None
    for angle in args.angle_range:
        torque_sector, torque_total, _ = torque_from_br_case(
            field,
            angle,
            args.current_rms_a,
            args.turns_per_layer,
            args.periodic_multiplier,
            args.sector_start_deg,
            args.sector_end_deg,
            args.stator_rotation_deg,
        )
        row = {
            "case": name,
            "field_csv": csv_path,
            "current_angle_deg": angle,
            "current_rms_a": args.current_rms_a,
            "torque_sector_Nm": torque_sector,
            "torque_total_Nm": torque_total,
        }
        rows.append(row)
        if best_abs is None or abs(torque_total) > abs(best_abs["torque_total_Nm"]):
            best_abs = row
        if best_positive is None or torque_total > best_positive["torque_total_Nm"]:
            best_positive = row
        if best_negative is None or torque_total < best_negative["torque_total_Nm"]:
            best_negative = row
    summary = {
        "case": name,
        "field_csv": csv_path,
        "best_abs_angle_deg": best_abs["current_angle_deg"],
        "best_abs_torque_total_Nm": best_abs["torque_total_Nm"],
        "best_positive_angle_deg": best_positive["current_angle_deg"],
        "best_positive_torque_total_Nm": best_positive["torque_total_Nm"],
        "best_negative_angle_deg": best_negative["current_angle_deg"],
        "best_negative_torque_total_Nm": best_negative["torque_total_Nm"],
    }
    return rows, summary


def write_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return output
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_sweep(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    cases = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        plt.plot(
            [row["current_angle_deg"] for row in case_rows],
            [row["torque_total_Nm"] for row in case_rows],
            marker="o",
            label=case,
        )
    plt.axhline(0.0, color="0.5", linewidth=0.8)
    plt.xlabel("Current angle (deg)")
    plt.ylabel("Torque total from Br*slot-current (N.m)")
    plt.title("Loaded torque estimate from no-load Br and winding currents")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate loaded torque from a no-load Br waveform and the stator slot currents. "
            "This is a Lorentz/winding-function check, not a full armature-reaction subdomain solve."
        )
    )
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--current-rms-a", type=float, default=DEFAULT_CURRENT_RMS_A)
    parser.add_argument("--angle-range", type=parse_float_range, default=parse_float_range("120,170,2.5"))
    parser.add_argument("--turns-per-layer", type=int, default=base.TURNS_PER_LAYER_DEFAULT)
    parser.add_argument("--periodic-multiplier", type=float, default=DEFAULT_PERIODIC_MULTIPLIER)
    parser.add_argument("--sector-start-deg", type=float, default=0.0)
    parser.add_argument("--sector-end-deg", type=float, default=60.0)
    parser.add_argument("--stator-rotation-deg", type=float, default=base.STATOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--case", dest="cases", type=parse_case, action="append")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = args.cases if args.cases else list(DEFAULT_CASES)
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for name, path in cases:
        rows, summary = run_case(name, path, args)
        all_rows.extend(rows)
        summaries.append(summary)

    sweep_csv = write_csv(out / "loaded_torque_from_br_sweep.csv", all_rows)
    summary_csv = write_csv(out / "loaded_torque_from_br_summary.csv", summaries)
    plot_path = plot_sweep(out / "loaded_torque_from_br_sweep.png", all_rows)

    print("=== Loaded torque estimate from Br ===")
    print(f"Sweep CSV: {sweep_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Plot: {plot_path}")
    for row in summaries:
        print(
            f"{row['case']}: best abs {row['best_abs_torque_total_Nm']:.6g} N.m "
            f"at {row['best_abs_angle_deg']:.6g} deg"
        )


if __name__ == "__main__":
    main()

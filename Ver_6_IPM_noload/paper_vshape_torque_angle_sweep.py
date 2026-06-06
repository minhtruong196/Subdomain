import argparse
import csv
from pathlib import Path

import femm
import matplotlib.pyplot as plt

from paper_vshape_torque_test import phase_currents_from_rms
from V_shape_equavalent_br_bt_export import parse_torque_groups, format_torque_groups


DEFAULT_CURRENT_RMS_A = 17.6
DEFAULT_PERIODIC_MULTIPLIER = 6


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


def torque_for_angle(current_rms_a, current_angle_deg, torque_groups, periodic_multiplier):
    currents = phase_currents_from_rms(current_rms_a, current_angle_deg)
    for circuit_name, current in currents.items():
        femm.mi_modifycircprop(circuit_name, 1, current)
    femm.mi_analyze(1)
    femm.mi_loadsolution()
    femm.mo_clearblock()
    for group in torque_groups:
        femm.mo_groupselectblock(group)
    torque_one_sector = femm.mo_blockintegral(22)
    femm.mo_clearblock()
    femm.mo_close()
    return torque_one_sector, torque_one_sector * periodic_multiplier, currents


def run_sweep(fem_path, result_dir, current_rms_a, angle_values, torque_groups, periodic_multiplier):
    model_path = Path(fem_path).absolute()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    femm.openfemm(1)
    try:
        femm.opendocument(model_path.as_posix())
        for angle in angle_values:
            torque_one_sector, torque_total, currents = torque_for_angle(
                current_rms_a,
                angle,
                torque_groups,
                periodic_multiplier,
            )
            row = {
                "current_angle_deg": angle,
                "current_rms_a": current_rms_a,
                "Ia_A": currents["Ia"],
                "Ib_A": currents["Ib"],
                "Ic_A": currents["Ic"],
                "torque_one_sector_Nm": torque_one_sector,
                "torque_total_Nm": torque_total,
            }
            rows.append(row)
            print(f"{angle:8.3f} deg -> {torque_total: .9g} N.m")
    finally:
        try:
            femm.mi_close()
        except Exception:
            pass
        femm.closefemm()

    csv_path = out / "torque_angle_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_abs = max(rows, key=lambda row: abs(row["torque_total_Nm"]))
    best_positive = max(rows, key=lambda row: row["torque_total_Nm"])
    best_negative = min(rows, key=lambda row: row["torque_total_Nm"])
    summary = {
        "fem": model_path.name,
        "current_rms_a": current_rms_a,
        "periodic_multiplier": periodic_multiplier,
        "torque_groups": format_torque_groups(torque_groups),
        "num_angles": len(rows),
        "best_abs_angle_deg": best_abs["current_angle_deg"],
        "best_abs_torque_total_Nm": best_abs["torque_total_Nm"],
        "best_positive_angle_deg": best_positive["current_angle_deg"],
        "best_positive_torque_total_Nm": best_positive["torque_total_Nm"],
        "best_negative_angle_deg": best_negative["current_angle_deg"],
        "best_negative_torque_total_Nm": best_negative["torque_total_Nm"],
    }
    summary_path = out / "torque_angle_sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    png_path = out / "torque_angle_sweep.png"
    plt.figure(figsize=(9, 5))
    plt.plot(
        [row["current_angle_deg"] for row in rows],
        [row["torque_total_Nm"] for row in rows],
        marker="o",
    )
    plt.axhline(0.0, color="0.5", linewidth=0.8)
    plt.xlabel("Current angle (deg)")
    plt.ylabel("Torque total (N.m)")
    plt.title(f"Torque angle sweep: {model_path.name}, {current_rms_a:g} Arms")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return csv_path, summary_path, png_path, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep current angle and calculate FEMM torque.")
    parser.add_argument("--fem", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--current-rms-a", type=float, default=DEFAULT_CURRENT_RMS_A)
    parser.add_argument("--angle-range", type=parse_float_range, default=parse_float_range("0,180,15"))
    parser.add_argument("--torque-groups", type=parse_torque_groups, required=True)
    parser.add_argument("--periodic-multiplier", type=float, default=DEFAULT_PERIODIC_MULTIPLIER)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path, summary_path, png_path, summary = run_sweep(
        args.fem,
        args.result_dir,
        args.current_rms_a,
        args.angle_range,
        args.torque_groups,
        args.periodic_multiplier,
    )
    print("=== Torque angle sweep ===")
    print(f"FEM: {args.fem}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {png_path}")
    print(
        "Best abs: "
        f"{summary['best_abs_torque_total_Nm']:.9g} N.m at "
        f"{summary['best_abs_angle_deg']:.6g} deg"
    )


if __name__ == "__main__":
    main()

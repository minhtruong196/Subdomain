import argparse
import csv
import math
from pathlib import Path

import femm
import matplotlib.pyplot as plt


DEFAULT_FEM = "paper_ipm_vshape_1over6.FEM"
DEFAULT_RESULT_DIR = "results/paper_vshape_1over6"

# The generated paper V-shape FEM is a 60 mechanical degree sector.
DEFAULT_PERIODIC_MULTIPLIER = 6
DEFAULT_TORQUE_GROUP = 1

# Paper model dimensions: Rr = 38.8 mm, Rs = 40.0 mm.
DEFAULT_AIRGAP_RADIUS_MM = 39.4
DEFAULT_ARC_START_DEG = 0.0
DEFAULT_ARC_END_DEG = 60.0
DEFAULT_ARC_SAMPLE_MARGIN_DEG = 0.01
DEFAULT_NUM_FIELD_POINTS = 301

# Table I rated current is 17.6 Arms. Convert to phase peak before setting FEMM.
DEFAULT_CURRENT_RMS_A = 0
DEFAULT_CURRENT_ANGLE_DEG = 0.0


def phase_currents_from_rms(current_rms_a, current_angle_deg):
    ipeak = current_rms_a * math.sqrt(2.0)
    angle = math.radians(current_angle_deg)
    return {
        "Ia": ipeak * math.sin(angle),
        "Ib": ipeak * math.sin(angle - 2.0 * math.pi / 3.0),
        "Ic": ipeak * math.sin(angle + 2.0 * math.pi / 3.0),
    }


def iter_arc_points(radius_mm, start_deg, end_deg, num_points):
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)

    for idx in range(num_points):
        fraction = idx / (num_points - 1)
        theta = start_rad + (end_rad - start_rad) * fraction
        x = radius_mm * math.cos(theta)
        y = radius_mm * math.sin(theta)
        yield math.degrees(theta), x, y


def cartesian_b_to_polar_b(x, y, bx, by):
    theta = math.atan2(y, x)
    br = bx * math.cos(theta) + by * math.sin(theta)
    bt = -bx * math.sin(theta) + by * math.cos(theta)
    return br, bt


def plot_field_component(angles, values, label, title, output_path):
    min_idx = min(range(len(values)), key=values.__getitem__)
    max_idx = max(range(len(values)), key=values.__getitem__)
    min_angle = angles[min_idx]
    max_angle = angles[max_idx]
    min_value = values[min_idx]
    max_value = values[max_idx]

    plt.figure(figsize=(9, 5))
    plt.plot(angles, values, label=label)
    plt.scatter([min_angle, max_angle], [min_value, max_value], zorder=3)
    plt.text(
        0.98,
        0.96,
        (
            f"min = {min_value:.6g} T at {min_angle:.3g} deg\n"
            f"max = {max_value:.6g} T at {max_angle:.3g} deg"
        ),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    plt.xlabel("Mechanical angle (deg)")
    plt.ylabel("Flux density (T)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def export_br_bt_arc(
    result_dir,
    radius_mm,
    start_deg,
    end_deg,
    sample_margin_deg,
    num_points,
):
    result_path = Path(result_dir).absolute()
    result_path.mkdir(parents=True, exist_ok=True)
    sample_start_deg = start_deg + sample_margin_deg
    sample_end_deg = end_deg - sample_margin_deg

    rows = []
    for angle_deg, x, y in iter_arc_points(radius_mm, sample_start_deg, sample_end_deg, num_points):
        bx, by = femm.mo_getb(x, y)
        br, bt = cartesian_b_to_polar_b(x, y, bx, by)
        rows.append(
            {
                "angle_deg": angle_deg,
                "x_mm": x,
                "y_mm": y,
                "Bx_T": bx,
                "By_T": by,
                "Br_T": br,
                "Bt_T": bt,
            }
        )

    csv_path = result_path / "br_bt_arc.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    angles = [row["angle_deg"] for row in rows]
    br_values = [row["Br_T"] for row in rows]
    bt_values = [row["Bt_T"] for row in rows]

    br_png_path = result_path / "br_arc.png"
    bt_png_path = result_path / "bt_arc.png"
    plot_field_component(
        angles,
        br_values,
        "Br",
        f"Br on r = {radius_mm:g} mm ({sample_start_deg:g} to {sample_end_deg:g} deg)",
        br_png_path,
    )
    plot_field_component(
        angles,
        bt_values,
        "Bt",
        f"Bt on r = {radius_mm:g} mm ({sample_start_deg:g} to {sample_end_deg:g} deg)",
        bt_png_path,
    )

    return br_png_path, bt_png_path, csv_path


def write_torque_summary(result_dir, values):
    result_path = Path(result_dir).absolute()
    result_path.mkdir(parents=True, exist_ok=True)
    csv_path = result_path / "torque_summary.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "value"])
        for key, value in values.items():
            writer.writerow([key, value])
    return csv_path


def run_torque_test(
    fem_path,
    result_dir,
    current_rms_a,
    current_angle_deg,
    periodic_multiplier,
    torque_group,
    airgap_radius_mm,
    arc_start_deg,
    arc_end_deg,
    arc_sample_margin_deg,
    num_field_points,
    export_br_bt=True,
    use_existing_solution=False,
):
    model_path = Path(fem_path).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"FEM file not found: {model_path}")

    phase_currents = phase_currents_from_rms(current_rms_a, current_angle_deg)

    print("=== Model setup ===")
    print(f"FEM = {model_path}")
    print(f"current_rms_a = {current_rms_a:.6g} A")
    print(f"current_angle_deg = {current_angle_deg:.6g} deg")
    for circuit_name, current in phase_currents.items():
        print(f"{circuit_name} = {current:.6f} A")

    femm.openfemm(1)
    try:
        femm.opendocument(model_path.as_posix())

        if use_existing_solution:
            femm.mi_loadsolution()
        else:
            for circuit_name, current in phase_currents.items():
                femm.mi_modifycircprop(circuit_name, 1, current)
            femm.mi_analyze(1)
            femm.mi_loadsolution()

        femm.mo_groupselectblock(torque_group)
        torque_one_sector = femm.mo_blockintegral(22)
        femm.mo_clearblock()
        torque_total = torque_one_sector * periodic_multiplier

        summary = {
            "fem": model_path.name,
            "current_rms_a": current_rms_a,
            "current_angle_deg": current_angle_deg,
            "Ia_A": phase_currents["Ia"],
            "Ib_A": phase_currents["Ib"],
            "Ic_A": phase_currents["Ic"],
            "torque_group": torque_group,
            "periodic_multiplier": periodic_multiplier,
            "torque_one_sector_Nm": torque_one_sector,
            "torque_total_Nm": torque_total,
            "airgap_radius_mm": airgap_radius_mm,
            "arc_start_deg": arc_start_deg,
            "arc_end_deg": arc_end_deg,
            "arc_sample_margin_deg": arc_sample_margin_deg,
            "num_field_points": num_field_points,
            "export_br_bt": export_br_bt,
        }
        summary_csv = write_torque_summary(result_dir, summary)

        print("\n=== Torque result ===")
        print(f"group = {torque_group}")
        print(f"periodic_multiplier = {periodic_multiplier}")
        print(f"torque_one_sector = {torque_one_sector:.9g} N.m")
        print(f"torque_total = {torque_total:.9g} N.m")
        print(f"Torque CSV: {summary_csv}")

        if export_br_bt:
            br_png_path, bt_png_path, csv_path = export_br_bt_arc(
                result_dir,
                airgap_radius_mm,
                arc_start_deg,
                arc_end_deg,
                arc_sample_margin_deg,
                num_field_points,
            )
            print("\n=== Br/Bt arc export ===")
            print(f"radius = {airgap_radius_mm:g} mm")
            print(f"angle = {arc_start_deg:g} deg to {arc_end_deg:g} deg")
            print(f"sample margin = {arc_sample_margin_deg:g} deg")
            print(f"Br PNG: {br_png_path}")
            print(f"Bt PNG: {bt_png_path}")
            print(f"CSV: {csv_path}")

        return torque_total
    finally:
        try:
            femm.mo_close()
        except Exception:
            pass
        try:
            femm.mi_close()
        except Exception:
            pass
        femm.closefemm()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate torque and export airgap Br/Bt for paper_ipm_vshape_1over6.FEM."
    )
    parser.add_argument("--fem", default=DEFAULT_FEM)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--current-rms-a", type=float, default=DEFAULT_CURRENT_RMS_A)
    parser.add_argument("--current-angle-deg", type=float, default=DEFAULT_CURRENT_ANGLE_DEG)
    parser.add_argument("--periodic-multiplier", type=float, default=DEFAULT_PERIODIC_MULTIPLIER)
    parser.add_argument("--torque-group", type=int, default=DEFAULT_TORQUE_GROUP)
    parser.add_argument("--airgap-radius-mm", type=float, default=DEFAULT_AIRGAP_RADIUS_MM)
    parser.add_argument("--arc-start-deg", type=float, default=DEFAULT_ARC_START_DEG)
    parser.add_argument("--arc-end-deg", type=float, default=DEFAULT_ARC_END_DEG)
    parser.add_argument("--arc-sample-margin-deg", type=float, default=DEFAULT_ARC_SAMPLE_MARGIN_DEG)
    parser.add_argument("--num-field-points", type=int, default=DEFAULT_NUM_FIELD_POINTS)
    parser.add_argument("--no-export-br-bt", action="store_true")
    parser.add_argument(
        "--use-existing-solution",
        action="store_true",
        help="Read the current .ans solution without changing circuit currents or solving again.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_torque_test(
        fem_path=args.fem,
        result_dir=args.result_dir,
        current_rms_a=args.current_rms_a,
        current_angle_deg=args.current_angle_deg,
        periodic_multiplier=args.periodic_multiplier,
        torque_group=args.torque_group,
        airgap_radius_mm=args.airgap_radius_mm,
        arc_start_deg=args.arc_start_deg,
        arc_end_deg=args.arc_end_deg,
        arc_sample_margin_deg=args.arc_sample_margin_deg,
        num_field_points=args.num_field_points,
        export_br_bt=not args.no_export_br_bt,
        use_existing_solution=args.use_existing_solution,
    )


if __name__ == "__main__":
    main()

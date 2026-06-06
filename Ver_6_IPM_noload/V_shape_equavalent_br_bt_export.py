import argparse
import csv
from pathlib import Path

import femm

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
from paper_vshape_torque_test import export_br_bt_arc, phase_currents_from_rms


DEFAULT_RESULT_DIR = "results/paper_vshape_equivalent_1over6"
DEFAULT_AIRGAP_RADIUS_MM = 39.4
DEFAULT_ARC_START_DEG = 0.0
DEFAULT_ARC_END_DEG = 60.0
DEFAULT_ARC_SAMPLE_MARGIN_DEG = 0.01
DEFAULT_NUM_FIELD_POINTS = 301
DEFAULT_CURRENT_ANGLE_DEG = 0.0
DEFAULT_PERIODIC_MULTIPLIER = 6
DEFAULT_TORQUE_GROUPS = tuple(range(equivalent.GROUP_ROTOR_OUTER, equivalent.GROUP_SHAFT_AIR + 1))

# Edit these values when running this file directly from the IDE.
RUN_USE_EXISTING_FEM = True
RUN_USE_EXISTING_SOLUTION = False
RUN_CURRENT_RMS_A = 0.0
RUN_CURRENT_ANGLE_DEG = DEFAULT_CURRENT_ANGLE_DEG
RUN_AIRGAP_RADIUS_MM = DEFAULT_AIRGAP_RADIUS_MM
RUN_ARC_START_DEG = DEFAULT_ARC_START_DEG
RUN_ARC_END_DEG = DEFAULT_ARC_END_DEG
RUN_ARC_SAMPLE_MARGIN_DEG = DEFAULT_ARC_SAMPLE_MARGIN_DEG
RUN_NUM_FIELD_POINTS = DEFAULT_NUM_FIELD_POINTS
RUN_PERIODIC_MULTIPLIER = DEFAULT_PERIODIC_MULTIPLIER
RUN_TORQUE_GROUPS = DEFAULT_TORQUE_GROUPS


def parse_torque_groups(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(group) for group in value)

    groups = []
    for chunk in str(value).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            groups.extend(range(start, end + step, step))
        else:
            groups.append(int(chunk))

    if not groups:
        raise argparse.ArgumentTypeError("torque groups must not be empty")

    return tuple(dict.fromkeys(groups))


def format_torque_groups(groups):
    return ",".join(str(group) for group in groups)


def write_export_summary(result_dir, values):
    result_path = Path(result_dir).absolute()
    result_path.mkdir(parents=True, exist_ok=True)
    csv_path = result_path / "br_bt_export_summary.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "value"])
        for key, value in values.items():
            writer.writerow([key, value])
    return csv_path


def solve_and_export_br_bt(
    fem_path,
    result_dir,
    current_rms_a,
    current_angle_deg,
    airgap_radius_mm,
    arc_start_deg,
    arc_end_deg,
    arc_sample_margin_deg,
    num_field_points,
    periodic_multiplier,
    torque_groups,
    use_existing_solution=False,
):
    if num_field_points < 2:
        raise ValueError("--num-field-points must be at least 2.")

    model_path = Path(fem_path).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"FEM file not found: {model_path}")

    phase_currents = phase_currents_from_rms(current_rms_a, current_angle_deg)

    print("=== Equivalent Br/Bt export setup ===")
    print(f"FEM = {model_path}")
    print(f"current_rms_a = {current_rms_a:.6g} A")
    print(f"current_angle_deg = {current_angle_deg:.6g} deg")
    for circuit_name, current in phase_currents.items():
        print(f"{circuit_name} = {current:.6f} A")

    femm.openfemm(1)
    try:
        femm.opendocument(model_path.as_posix())

        if use_existing_solution:
            print("using existing solution: current inputs are not applied to the .ans file")
            femm.mi_loadsolution()
        else:
            print("solving with requested current inputs")
            for circuit_name, current in phase_currents.items():
                femm.mi_modifycircprop(circuit_name, 1, current)
            femm.mi_analyze(1)
            femm.mi_loadsolution()

        femm.mo_clearblock()
        for torque_group in torque_groups:
            femm.mo_groupselectblock(torque_group)
        torque_one_sector = femm.mo_blockintegral(22)
        femm.mo_clearblock()
        torque_total = torque_one_sector * periodic_multiplier

        br_png_path, bt_png_path, csv_path = export_br_bt_arc(
            result_dir,
            airgap_radius_mm,
            arc_start_deg,
            arc_end_deg,
            arc_sample_margin_deg,
            num_field_points,
        )
        summary_csv = write_export_summary(
            result_dir,
            {
                "fem": model_path.name,
                "use_existing_solution": use_existing_solution,
                "current_rms_a": current_rms_a,
                "current_angle_deg": current_angle_deg,
                "Ia_A": phase_currents["Ia"],
                "Ib_A": phase_currents["Ib"],
                "Ic_A": phase_currents["Ic"],
                "airgap_radius_mm": airgap_radius_mm,
                "arc_start_deg": arc_start_deg,
                "arc_end_deg": arc_end_deg,
                "arc_sample_margin_deg": arc_sample_margin_deg,
                "num_field_points": num_field_points,
                "torque_groups": format_torque_groups(torque_groups),
                "periodic_multiplier": periodic_multiplier,
                "torque_one_sector_Nm": torque_one_sector,
                "torque_total_Nm": torque_total,
                "br_png": br_png_path,
                "bt_png": bt_png_path,
                "br_bt_csv": csv_path,
            },
        )

        print("\n=== Equivalent torque result ===")
        print(f"groups = {format_torque_groups(torque_groups)}")
        print(f"periodic_multiplier = {periodic_multiplier}")
        print(f"torque_one_sector = {torque_one_sector:.9g} N.m")
        print(f"torque_total = {torque_total:.9g} N.m")

        print("\n=== Equivalent Br/Bt arc export ===")
        print(f"radius = {airgap_radius_mm:g} mm")
        print(f"angle = {arc_start_deg:g} deg to {arc_end_deg:g} deg")
        print(f"sample margin = {arc_sample_margin_deg:g} deg")
        print(f"Br PNG: {br_png_path}")
        print(f"Bt PNG: {bt_png_path}")
        print(f"CSV: {csv_path}")
        print(f"Summary CSV: {summary_csv}")
        return br_png_path, bt_png_path, csv_path
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
        description="Build/solve the equivalent V-shape model and export airgap Br/Bt."
    )
    parser.add_argument("--output", default=equivalent.EQUIVALENT_OUTPUT_DEFAULT)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--use-existing-fem",
        action="store_true",
        default=RUN_USE_EXISTING_FEM,
        help="Do not rebuild the FEM file; open --output directly.",
    )
    parser.add_argument(
        "--use-existing-solution",
        action="store_true",
        default=RUN_USE_EXISTING_SOLUTION,
        help="Read the existing .ans solution without changing currents or solving again; current inputs are ignored.",
    )

    parser.add_argument("--current-rms-a", type=float, default=RUN_CURRENT_RMS_A)
    parser.add_argument("--current-angle-deg", type=float, default=RUN_CURRENT_ANGLE_DEG)
    parser.add_argument("--airgap-radius-mm", type=float, default=RUN_AIRGAP_RADIUS_MM)
    parser.add_argument("--arc-start-deg", type=float, default=RUN_ARC_START_DEG)
    parser.add_argument("--arc-end-deg", type=float, default=RUN_ARC_END_DEG)
    parser.add_argument("--arc-sample-margin-deg", type=float, default=RUN_ARC_SAMPLE_MARGIN_DEG)
    parser.add_argument("--num-field-points", type=int, default=RUN_NUM_FIELD_POINTS)
    parser.add_argument("--periodic-multiplier", type=float, default=RUN_PERIODIC_MULTIPLIER)
    parser.add_argument(
        "--torque-groups",
        "--torque-group",
        dest="torque_groups",
        type=parse_torque_groups,
        default=RUN_TORQUE_GROUPS,
        help="Rotor block groups for torque. Accepts comma lists and ranges, e.g. 6-11.",
    )

    parser.add_argument("--turns-per-layer", type=int, default=base.TURNS_PER_LAYER_DEFAULT)
    parser.add_argument("--stator-outer-radius", type=float, default=base.STATOR_OUTER_RADIUS_MM_DEFAULT)
    parser.add_argument("--shaft-radius", type=float, default=base.SHAFT_RADIUS_MM_DEFAULT)
    parser.add_argument("--w1-mm", type=float, default=equivalent.EQUIVALENT_W1_MM_DEFAULT)
    parser.add_argument("--w2-mm", type=float, default=equivalent.EQUIVALENT_W2_MM_DEFAULT)
    parser.add_argument(
        "--equivalent-radial-shift-mm",
        "--magnet-radial-shift-mm",
        dest="equivalent_radial_shift_mm",
        type=float,
        default=equivalent.EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
        help="Move the whole equivalent PM + Air pocket radially; positive is outward.",
    )
    parser.add_argument("--wb1-mm", type=float)
    parser.add_argument("--hb1-mm", type=float)
    parser.add_argument("--wb2-mm", type=float)
    parser.add_argument("--alpha-deg", type=float)
    parser.add_argument("--stator-rotation-deg", type=float, default=base.STATOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--rotor-rotation-deg", type=float, default=base.ROTOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--sector-start-deg", type=float, default=base.SECTOR_START_DEG_DEFAULT)
    parser.add_argument("--sector-span-deg", type=float, default=base.SECTOR_SPAN_DEG_DEFAULT)
    parser.add_argument(
        "--sector-boundary-kind",
        choices=("anti-periodic", "periodic"),
        default=base.SECTOR_BOUNDARY_KIND_DEFAULT,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_existing_solution and (
        args.current_rms_a != equivalent.EQUIVALENT_CURRENT_RMS_A_DEFAULT
        or args.current_angle_deg != DEFAULT_CURRENT_ANGLE_DEG
    ):
        print(
            "warning: --use-existing-solution is set, so --current-rms-a and "
            "--current-angle-deg will not be applied"
        )

    if not args.use_existing_fem:
        dims = equivalent.build_model(
            output_path=args.output,
            stator_outer_radius=args.stator_outer_radius,
            shaft_radius=args.shaft_radius,
            current_rms_a=args.current_rms_a,
            turns_per_layer=args.turns_per_layer,
            w1_mm=args.w1_mm,
            w2_mm=args.w2_mm,
            equivalent_radial_shift_mm=args.equivalent_radial_shift_mm,
            wb1_mm=args.wb1_mm,
            hb1_mm=args.hb1_mm,
            wb2_mm=args.wb2_mm,
            alpha_deg=args.alpha_deg,
            stator_rotation_deg=args.stator_rotation_deg,
            rotor_rotation_deg=args.rotor_rotation_deg,
            sector_start_deg=args.sector_start_deg,
            sector_span_deg=args.sector_span_deg,
            sector_boundary_kind=args.sector_boundary_kind,
        )
        print(
            f"saved {args.output} "
            f"(alpha={dims['alpha_deg']:.6g}deg, alpha1={dims['alpha1_deg']:.6g}deg, "
            f"Rf={dims['rf']:.6g}mm, Rm={dims['rm']:.6g}mm, Rl={dims['rl']:.6g}mm)"
        )
    else:
        print(f"using existing FEM: {Path(args.output).absolute()}")

    solve_and_export_br_bt(
        fem_path=args.output,
        result_dir=args.result_dir,
        current_rms_a=args.current_rms_a,
        current_angle_deg=args.current_angle_deg,
        airgap_radius_mm=args.airgap_radius_mm,
        arc_start_deg=args.arc_start_deg,
        arc_end_deg=args.arc_end_deg,
        arc_sample_margin_deg=args.arc_sample_margin_deg,
        num_field_points=args.num_field_points,
        periodic_multiplier=args.periodic_multiplier,
        torque_groups=args.torque_groups,
        use_existing_solution=args.use_existing_solution,
    )


if __name__ == "__main__":
    main()

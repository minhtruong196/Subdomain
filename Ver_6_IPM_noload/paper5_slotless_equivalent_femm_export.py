import argparse
from pathlib import Path

import femm

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
from paper_vshape_torque_test import export_br_bt_arc


SLOTLESS_OUTPUT_DEFAULT = "paper5_slotless_equivalent_1over6.FEM"
DEFAULT_RESULT_DIR = "results/paper5_slotless_equivalent_1over6"
DEFAULT_AIRGAP_RADIUS_MM = 39.4
DEFAULT_ARC_START_DEG = 0.0
DEFAULT_ARC_END_DEG = 60.0
DEFAULT_ARC_SAMPLE_MARGIN_DEG = 0.01
DEFAULT_NUM_FIELD_POINTS = 301


def draw_slotless_stator(spec, stator_outer_radius, sector_start_rad, sector_end_rad):
    base.add_sector_arc(stator_outer_radius, sector_start_rad, sector_end_rad, maxseg=2.0, boundary="A0")
    base.add_sector_arc(spec.stator_inner_radius, sector_start_rad, sector_end_rad, maxseg=1.0)
    label_angle = 0.5 * (sector_start_rad + sector_end_rad)
    base.add_block_label(
        *base.polar(0.5 * (spec.stator_inner_radius + stator_outer_radius), label_angle),
        base.CORE_MATERIAL_NAME,
        group=equivalent.GROUP_STATOR_YOKE,
    )


def build_slotless_model(
    output_path=SLOTLESS_OUTPUT_DEFAULT,
    stator_outer_radius=base.STATOR_OUTER_RADIUS_MM_DEFAULT,
    shaft_radius=base.SHAFT_RADIUS_MM_DEFAULT,
    w1_mm=equivalent.EQUIVALENT_W1_MM_DEFAULT,
    w2_mm=equivalent.EQUIVALENT_W2_MM_DEFAULT,
    equivalent_radial_shift_mm=equivalent.EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
    wb1_mm=None,
    hb1_mm=None,
    wb2_mm=None,
    alpha_deg=None,
    rotor_rotation_deg=base.ROTOR_ROTATION_DEG_DEFAULT,
    sector_start_deg=base.SECTOR_START_DEG_DEFAULT,
    sector_span_deg=base.SECTOR_SPAN_DEG_DEFAULT,
    sector_boundary_kind=base.SECTOR_BOUNDARY_KIND_DEFAULT,
):
    spec = base.PAPER_SPECS["vshape"]
    rotor_rotation_rad = base.math.radians(rotor_rotation_deg)
    sector_start_rad = base.math.radians(sector_start_deg)
    sector_end_rad = base.math.radians(sector_start_deg + sector_span_deg)
    dims = equivalent.equivalent_pm_dimensions(
        spec,
        alpha_deg=alpha_deg,
        w1_mm=w1_mm,
        w2_mm=w2_mm,
        wb1_mm=wb1_mm,
        hb1_mm=hb1_mm,
        wb2_mm=wb2_mm,
        radial_shift_mm=equivalent_radial_shift_mm,
    )

    femm.openfemm(1)
    try:
        femm.newdocument(0)
        femm.mi_probdef(0, "millimeters", "planar", 1e-8, spec.stack_length, 30)
        femm.mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_addcircprop("Ia", 0, 1)
        femm.mi_addcircprop("Ib", 0, 1)
        femm.mi_addcircprop("Ic", 0, 1)
        base.add_materials(spec)

        draw_slotless_stator(spec, stator_outer_radius, sector_start_rad, sector_end_rad)
        equivalent.draw_rotor_without_center_core_label(
            spec, shaft_radius, dims, sector_start_rad, sector_end_rad
        )
        equivalent.add_sector_side_boundaries_with_subdomain_cuts(
            sector_start_rad,
            sector_end_rad,
            shaft_radius,
            spec,
            stator_outer_radius,
            dims,
            "SectorPeriodic",
            sector_boundary_kind,
        )
        base.add_block_label(
            *base.polar(
                0.5 * (spec.rotor_outer_radius + spec.stator_inner_radius),
                0.5 * (sector_start_rad + sector_end_rad),
            ),
            "Air",
            group=equivalent.GROUP_AIRGAP,
        )
        equivalent.draw_equivalent_vshape_magnets(
            spec,
            rotor_rotation_rad,
            dims,
            sector_start_rad,
            sector_end_rad,
        )
        equivalent.draw_rotor_analytical_subdomain_boundaries(
            spec,
            rotor_rotation_rad,
            dims,
            sector_start_rad,
            sector_end_rad,
        )
        equivalent.add_rotor_core_subdomain_labels(
            spec,
            dims,
            rotor_rotation_rad,
            sector_start_rad,
            sector_end_rad,
            shaft_radius,
        )
        equivalent.add_rotor_edge_core_labels(
            spec,
            dims,
            rotor_rotation_rad,
            sector_start_rad,
            sector_end_rad,
        )

        femm.mi_zoomnatural()
        output = Path(output_path).absolute()
        femm.mi_saveas(str(output))
    finally:
        femm.closefemm()
    return dims


def solve_and_export(
    fem_path,
    result_dir,
    airgap_radius_mm,
    arc_start_deg,
    arc_end_deg,
    arc_sample_margin_deg,
    num_field_points,
):
    model_path = Path(fem_path).absolute()
    femm.openfemm(1)
    try:
        femm.opendocument(model_path.as_posix())
        femm.mi_analyze(1)
        femm.mi_loadsolution()
        br_png, bt_png, csv_path = export_br_bt_arc(
            result_dir,
            airgap_radius_mm,
            arc_start_deg,
            arc_end_deg,
            arc_sample_margin_deg,
            num_field_points,
        )
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
    return br_png, bt_png, csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a slotless equivalent FEMM benchmark for paper [5] stage-1 validation."
    )
    parser.add_argument("--output", default=SLOTLESS_OUTPUT_DEFAULT)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--use-existing-fem", action="store_true")
    parser.add_argument("--stator-outer-radius", type=float, default=base.STATOR_OUTER_RADIUS_MM_DEFAULT)
    parser.add_argument("--shaft-radius", type=float, default=base.SHAFT_RADIUS_MM_DEFAULT)
    parser.add_argument("--w1-mm", type=float, default=equivalent.EQUIVALENT_W1_MM_DEFAULT)
    parser.add_argument("--w2-mm", type=float, default=equivalent.EQUIVALENT_W2_MM_DEFAULT)
    parser.add_argument(
        "--equivalent-radial-shift-mm",
        type=float,
        default=equivalent.EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
    )
    parser.add_argument("--wb1-mm", type=float)
    parser.add_argument("--hb1-mm", type=float)
    parser.add_argument("--wb2-mm", type=float)
    parser.add_argument("--alpha-deg", type=float)
    parser.add_argument("--rotor-rotation-deg", type=float, default=base.ROTOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--sector-start-deg", type=float, default=base.SECTOR_START_DEG_DEFAULT)
    parser.add_argument("--sector-span-deg", type=float, default=base.SECTOR_SPAN_DEG_DEFAULT)
    parser.add_argument(
        "--sector-boundary-kind",
        choices=("anti-periodic", "periodic"),
        default=base.SECTOR_BOUNDARY_KIND_DEFAULT,
    )
    parser.add_argument("--airgap-radius-mm", type=float, default=DEFAULT_AIRGAP_RADIUS_MM)
    parser.add_argument("--arc-start-deg", type=float, default=DEFAULT_ARC_START_DEG)
    parser.add_argument("--arc-end-deg", type=float, default=DEFAULT_ARC_END_DEG)
    parser.add_argument("--arc-sample-margin-deg", type=float, default=DEFAULT_ARC_SAMPLE_MARGIN_DEG)
    parser.add_argument("--num-field-points", type=int, default=DEFAULT_NUM_FIELD_POINTS)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.use_existing_fem:
        dims = build_slotless_model(
            output_path=args.output,
            stator_outer_radius=args.stator_outer_radius,
            shaft_radius=args.shaft_radius,
            w1_mm=args.w1_mm,
            w2_mm=args.w2_mm,
            equivalent_radial_shift_mm=args.equivalent_radial_shift_mm,
            wb1_mm=args.wb1_mm,
            hb1_mm=args.hb1_mm,
            wb2_mm=args.wb2_mm,
            alpha_deg=args.alpha_deg,
            rotor_rotation_deg=args.rotor_rotation_deg,
            sector_start_deg=args.sector_start_deg,
            sector_span_deg=args.sector_span_deg,
            sector_boundary_kind=args.sector_boundary_kind,
        )
        print(
            f"saved {args.output} "
            f"(Rf={dims['rf']:.6g} mm, Rm={dims['rm']:.6g} mm, Rl={dims['rl']:.6g} mm)"
        )
    else:
        print(f"using existing FEM: {Path(args.output).absolute()}")

    br_png, bt_png, csv_path = solve_and_export(
        args.output,
        args.result_dir,
        args.airgap_radius_mm,
        args.arc_start_deg,
        args.arc_end_deg,
        args.arc_sample_margin_deg,
        args.num_field_points,
    )
    print(f"Br PNG: {br_png}")
    print(f"Bt PNG: {bt_png}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

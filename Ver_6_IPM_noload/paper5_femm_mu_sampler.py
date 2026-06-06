import argparse
import csv
import math
from pathlib import Path

import femm

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent


MU0 = 4.0 * math.pi * 1e-7
DEFAULT_OUTPUT = "results/paper5_femm_mu_samples.csv"
PAPER5_USER_BRIDGE_POINTS = (
    {"name": "outer_bridge_lower", "x_mm": 37.7, "y_mm": 4.3},
    {"name": "outer_bridge_upper", "x_mm": 22.5, "y_mm": 30.4},
    {"name": "center_bridge", "x_mm": 21.9, "y_mm": 12.7},
)


def parse_point(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--point must be x_mm,y_mm or x_mm,y_mm,name")
    try:
        x_mm = float(parts[0])
        y_mm = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--point x/y must be numeric") from exc
    name = parts[2] if len(parts) == 3 and parts[2] else f"point_{x_mm:g}_{y_mm:g}"
    return {"name": name, "x_mm": x_mm, "y_mm": y_mm}


def read_points_csv(path):
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            try:
                x_mm = float(row["x_mm"])
                y_mm = float(row["y_mm"])
            except KeyError as exc:
                raise ValueError("points CSV must contain x_mm and y_mm columns") from exc
            name = row.get("name") or f"csv_point_{idx}"
            points.append({"name": name, "x_mm": x_mm, "y_mm": y_mm})
    return points


def read_fem_block_label_points(path, groups):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    wanted = {int(group) for group in groups}
    points = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("[NumBlockLabels]"):
            count = int(line.split("=")[1])
            idx += 1
            for label_idx in range(count):
                parts = lines[idx].split()
                x_mm = float(parts[0])
                y_mm = float(parts[1])
                group = int(parts[6])
                if group in wanted:
                    points.append(
                        {
                            "name": f"fem_group_{group}_label_{label_idx}",
                            "x_mm": x_mm,
                            "y_mm": y_mm,
                        }
                    )
                idx += 1
            break
        idx += 1
    return points


def parse_groups(text):
    groups = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        groups.append(int(chunk))
    if not groups:
        raise argparse.ArgumentTypeError("group list must not be empty")
    return groups


def polar_point(radius_mm, angle_rad, name):
    return {
        "name": name,
        "x_mm": radius_mm * math.cos(angle_rad),
        "y_mm": radius_mm * math.sin(angle_rad),
    }


def auto_equivalent_bridge_points(args):
    spec = base.PAPER_SPECS["vshape"]
    dims = equivalent.equivalent_pm_dimensions(
        spec,
        alpha_deg=args.alpha_deg,
        w1_mm=args.w1_mm,
        w2_mm=args.w2_mm,
        wb1_mm=args.wb1_mm,
        hb1_mm=args.hb1_mm,
        wb2_mm=args.wb2_mm,
        radial_shift_mm=args.equivalent_radial_shift_mm,
    )
    sector_start = math.radians(args.sector_start_deg)
    sector_end = math.radians(args.sector_start_deg + args.sector_span_deg)
    rotor_rotation = math.radians(args.rotor_rotation_deg)
    pole_pitch = 2.0 * math.pi / spec.poles

    center = None
    for pole in range(spec.poles):
        theta = rotor_rotation + pole * pole_pitch
        while theta < sector_start:
            theta += 2.0 * math.pi
        while theta > sector_end:
            theta -= 2.0 * math.pi
        if sector_start <= theta <= sector_end:
            center = theta
            break
    if center is None:
        raise ValueError("No PM pole center falls inside the requested sector.")

    alpha = dims["alpha_rad"]
    alpha1 = dims["alpha1_rad"]
    rf = dims["rf"]
    rm = dims["rm"]
    rl = dims["rl"]
    rr = spec.rotor_outer_radius
    outer_radius = 0.5 * (rl + rr)
    bridge_half_span = 0.25 * (alpha - alpha1)

    return [
        polar_point(0.5 * (rf + rm), center, "center_bridge"),
        polar_point(outer_radius, center - 0.5 * alpha + bridge_half_span, "left_outer_bridge"),
        polar_point(outer_radius, center + 0.5 * alpha - bridge_half_span, "right_outer_bridge"),
    ]


def point_values_to_row(point, values):
    bx_t = float(values[1])
    by_t = float(values[2])
    hx_a_per_m = float(values[5])
    hy_a_per_m = float(values[6])
    mu_x_femm = float(values[9])
    mu_y_femm = float(values[10])
    b_abs_t = math.hypot(bx_t, by_t)
    h_abs_a_per_m = math.hypot(hx_a_per_m, hy_a_per_m)
    mu_r_abs = b_abs_t / (MU0 * h_abs_a_per_m) if h_abs_a_per_m > 0.0 else float("nan")

    return {
        "name": point["name"],
        "x_mm": point["x_mm"],
        "y_mm": point["y_mm"],
        "A_Wb_per_m": float(values[0]),
        "Bx_T": bx_t,
        "By_T": by_t,
        "B_abs_T": b_abs_t,
        "Hx_A_per_m": hx_a_per_m,
        "Hy_A_per_m": hy_a_per_m,
        "H_abs_A_per_m": h_abs_a_per_m,
        "mu_r_abs_from_B_over_mu0H": mu_r_abs,
        "mu_x_femm": mu_x_femm,
        "mu_y_femm": mu_y_femm,
    }


def write_rows(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "x_mm",
        "y_mm",
        "A_Wb_per_m",
        "Bx_T",
        "By_T",
        "B_abs_T",
        "Hx_A_per_m",
        "Hy_A_per_m",
        "H_abs_A_per_m",
        "mu_r_abs_from_B_over_mu0H",
        "mu_x_femm",
        "mu_y_femm",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def sample_mu(fem_path, points, output_path, analyze=False):
    model_path = Path(fem_path).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"FEM file not found: {model_path}")
    if not points:
        raise ValueError("No sample points were provided. Use --point x_mm,y_mm,name or --points-csv.")

    femm.openfemm(1)
    try:
        femm.opendocument(model_path.as_posix())
        if analyze:
            femm.mi_analyze(1)
        femm.mi_loadsolution()

        rows = []
        for point in points:
            values = femm.mo_getpointvalues(point["x_mm"], point["y_mm"])
            rows.append(point_values_to_row(point, values))
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

    return write_rows(output_path, rows), rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample B/H and effective mu_r from FEMM at explicit user-provided points."
    )
    parser.add_argument("--fem", default=equivalent.EQUIVALENT_OUTPUT_DEFAULT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--point",
        action="append",
        type=parse_point,
        default=[],
        help="Sample point as x_mm,y_mm or x_mm,y_mm,name. Repeat for multiple points.",
    )
    parser.add_argument("--points-csv", help="CSV with columns name(optional), x_mm, y_mm.")
    parser.add_argument(
        "--fem-label-groups",
        type=parse_groups,
        help="Sample block label coordinates from the FEM file for these groups, e.g. 7 or 7,8.",
    )
    parser.add_argument(
        "--auto-equivalent-bridge-points",
        action="store_true",
        help="Legacy diagnostic only: sample deterministic bridge points derived from V_shape_equavalent.py geometry.",
    )
    parser.add_argument(
        "--paper5-user-bridge-points",
        action="store_true",
        help="Sample the three manually selected saturated bridge points used for the current paper [5] workflow.",
    )
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
        "--analyze",
        action="store_true",
        help="Run FEMM analysis before sampling. Default reads the existing solution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    points = list(args.point)
    if args.points_csv:
        points.extend(read_points_csv(args.points_csv))
    if args.auto_equivalent_bridge_points:
        points.extend(auto_equivalent_bridge_points(args))
    if args.fem_label_groups:
        points.extend(read_fem_block_label_points(args.fem, args.fem_label_groups))
    if args.paper5_user_bridge_points:
        points.extend(PAPER5_USER_BRIDGE_POINTS)
    output, rows = sample_mu(args.fem, points, args.output, analyze=args.analyze)
    print(f"sampled {len(rows)} FEMM point(s)")
    for row in rows:
        print(
            f"{row['name']}: x={row['x_mm']:g} mm, y={row['y_mm']:g} mm, "
            f"|B|={row['B_abs_T']:.6g} T, |H|={row['H_abs_A_per_m']:.6g} A/m, "
            f"mu_r_abs={row['mu_r_abs_from_B_over_mu0H']:.6g}, "
            f"mu_x_femm={row['mu_x_femm']:.6g}, mu_y_femm={row['mu_y_femm']:.6g}"
        )
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()

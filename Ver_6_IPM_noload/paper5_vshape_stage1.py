import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent


MU0 = 4.0 * math.pi * 1e-7
DEFAULT_BENCHMARK_CSV = "results/paper_vshape_equivalent_1over6/br_bt_arc.csv"
DEFAULT_RESULT_DIR = "results/paper5_stage1_slotless"


@dataclass(frozen=True)
class EquivalentGeometry:
    sector_start_rad: float
    sector_end_rad: float
    rotor_center_rad: float
    shaft_radius_m: float
    rf_m: float
    rm_m: float
    rl_minus_w1_m: float
    rr_m: float
    rs_m: float
    rso_m: float
    alpha_rad: float
    alpha1_rad: float
    center_gap_rad: float
    tangential_start_rad: float
    tangential_end_rad: float

    @property
    def sector_span_rad(self):
        return self.sector_end_rad - self.sector_start_rad


@dataclass(frozen=True)
class Domain:
    name: str
    r_inner: float
    r_outer: float
    mu_r: float
    br_r_cos: np.ndarray
    br_r_sin: np.ndarray
    br_t_cos: np.ndarray
    br_t_sin: np.ndarray


def meters(mm):
    return mm * 1e-3


def load_equivalent_geometry(args):
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

    centers = [rotor_rotation + pole * pole_pitch for pole in range(spec.poles)]
    center = None
    for theta in centers:
        shifted = theta
        while shifted < sector_start:
            shifted += 2.0 * math.pi
        while shifted > sector_end:
            shifted -= 2.0 * math.pi
        if sector_start <= shifted <= sector_end:
            center = shifted
            break
    if center is None:
        raise ValueError("No PM pole center falls inside the requested sector.")

    rm_m = meters(dims["rm"])
    center_gap = dims["wb2"] * 1e-3 / (2.0 * rm_m)
    center_air_gap = dims["w2"] * 1e-3 / rm_m
    tangential_start = center_gap + center_air_gap

    return EquivalentGeometry(
        sector_start_rad=sector_start,
        sector_end_rad=sector_end,
        rotor_center_rad=center,
        shaft_radius_m=meters(args.shaft_radius),
        rf_m=meters(dims["rf"]),
        rm_m=meters(dims["rm"]),
        rl_minus_w1_m=meters(dims["rl"] - dims["w1"]),
        rr_m=meters(spec.rotor_outer_radius),
        rs_m=meters(spec.stator_inner_radius),
        rso_m=meters(args.stator_outer_radius),
        alpha_rad=dims["alpha_rad"],
        alpha1_rad=dims["alpha1_rad"],
        center_gap_rad=center_gap,
        tangential_start_rad=tangential_start,
        tangential_end_rad=0.5 * dims["alpha1_rad"],
    )


def harmonic_lambdas(count, sector_span_rad):
    orders = np.arange(count, dtype=float)
    return (2.0 * orders + 1.0) * math.pi / sector_span_rad


def integrate_coefficients(theta, values, lambdas):
    span = theta[-1] - theta[0]
    cos_coeffs = []
    sin_coeffs = []
    for lam in lambdas:
        cos_coeffs.append((2.0 / span) * np.trapezoid(values * np.cos(lam * theta), theta))
        sin_coeffs.append((2.0 / span) * np.trapezoid(values * np.sin(lam * theta), theta))
    return np.array(cos_coeffs), np.array(sin_coeffs)


def profile_coefficients(geom, harmonic_count, brem_t, samples=6001):
    theta = np.linspace(geom.sector_start_rad, geom.sector_end_rad, samples)
    lam = harmonic_lambdas(harmonic_count, geom.sector_span_rad)

    br_radial = np.zeros_like(theta)
    br_tangential = np.zeros_like(theta)
    c = geom.rotor_center_rad

    def add_pm_window(target_r, target_t, start, end, magdir):
        mask = (start <= theta) & (theta <= end)
        target_r[mask] += brem_t * np.cos(magdir - theta[mask])
        target_t[mask] += brem_t * np.sin(magdir - theta[mask])

    tangential_windows = (
        (c - geom.tangential_end_rad, c - geom.tangential_start_rad),
        (c + geom.tangential_start_rad, c + geom.tangential_end_rad),
    )
    for start, end in tangential_windows:
        add_pm_window(br_radial, br_tangential, start, end, 0.5 * (start + end))

    side_windows = (
        (-1, c - 0.5 * geom.alpha_rad, c - 0.5 * geom.alpha1_rad),
        (1, c + 0.5 * geom.alpha1_rad, c + 0.5 * geom.alpha_rad),
    )
    for side, start, end in side_windows:
        side_mid = 0.5 * (start + end)
        add_pm_window(br_radial, br_tangential, start, end, side_mid - side * math.pi / 2.0)

    radial_cos, radial_sin = integrate_coefficients(theta, br_radial, lam)
    tangential_cos, tangential_sin = integrate_coefficients(theta, br_tangential, lam)
    zero = np.zeros(harmonic_count)
    return {
        "zero": (zero, zero),
        "radial_pm": (radial_cos, radial_sin, zero, zero),
        "side_pm": (zero, zero, tangential_cos, tangential_sin),
    }


def build_domains(geom, args, coeffs):
    n = args.harmonics
    zero = np.zeros(n)
    rr_cos, rr_sin, rt_cos, rt_sin = coeffs["radial_pm"]
    sr_cos, sr_sin, st_cos, st_sin = coeffs["side_pm"]

    return [
        Domain("rotor_inner_core", geom.shaft_radius_m, geom.rf_m, args.mu_core, zero, zero, zero, zero),
        Domain("tangential_pm_band", geom.rf_m, geom.rm_m, args.mu_pm, rr_cos, rr_sin, rt_cos, rt_sin),
        Domain("radial_pm_band", geom.rm_m, geom.rl_minus_w1_m, args.mu_pm, sr_cos, sr_sin, st_cos, st_sin),
        Domain("bridge_outer_core", geom.rl_minus_w1_m, geom.rr_m, args.mu_bridge, zero, zero, zero, zero),
        Domain("airgap", geom.rr_m, geom.rs_m, 1.0, zero, zero, zero, zero),
        Domain("stator_yoke", geom.rs_m, geom.rso_m, args.mu_core, zero, zero, zero, zero),
    ]


def mode_particular_coeff(domain, mode_idx, lam, r_ref, source_sign):
    br_r_cos = domain.br_r_cos[mode_idx]
    br_r_sin = domain.br_r_sin[mode_idx]
    br_t_cos = domain.br_t_cos[mode_idx]
    br_t_sin = domain.br_t_sin[mode_idx]

    # With remanence in tesla and x=r/r_ref:
    # laplacian(A) = source_sign * (Br_theta - d(Br_r)/dtheta) / r.
    src_cos = source_sign * (br_t_cos - lam * br_r_sin)
    src_sin = source_sign * (br_t_sin + lam * br_r_cos)
    denom = 1.0 - lam * lam
    return r_ref * src_cos / denom, r_ref * src_sin / denom


def basis_values(r, lam, r_ref, p_coeff):
    x = r / r_ref
    a_val = x**lam
    b_val = x ** (-lam)
    p_val = p_coeff * x
    da_dr = lam * x ** (lam - 1.0) / r_ref
    db_dr = -lam * x ** (-lam - 1.0) / r_ref
    dp_dr = p_coeff / r_ref
    return a_val, b_val, p_val, da_dr, db_dr, dp_dr


def solve_mode(domains, lam, mode_idx, r_ref, source_sign, component):
    nd = len(domains)
    unknowns = 2 * nd
    mat = np.zeros((unknowns, unknowns))
    rhs = np.zeros(unknowns)
    row = 0

    def p(domain):
        pc, ps = mode_particular_coeff(domain, mode_idx, lam, r_ref, source_sign)
        return pc if component == "cos" else ps

    # Inner magnetic insulation approximation at shaft radius.
    d0 = domains[0]
    _, _, _, da, db, dp = basis_values(d0.r_inner, lam, r_ref, p(d0))
    mat[row, 0] = da
    mat[row, 1] = db
    rhs[row] = -dp
    row += 1

    # Interface rows: Az continuity and Htheta continuity.
    for left_idx in range(nd - 1):
        left = domains[left_idx]
        right = domains[left_idx + 1]
        r = left.r_outer
        lp = p(left)
        rp = p(right)

        la, lb, lpart, lda, ldb, ldp = basis_values(r, lam, r_ref, lp)
        ra, rb, rpart, rda, rdb, rdp = basis_values(r, lam, r_ref, rp)

        li = 2 * left_idx
        ri = 2 * (left_idx + 1)

        mat[row, li] = la
        mat[row, li + 1] = lb
        mat[row, ri] = -ra
        mat[row, ri + 1] = -rb
        rhs[row] = rpart - lpart
        row += 1

        left_bt_rem = left.br_t_cos[mode_idx] if component == "cos" else left.br_t_sin[mode_idx]
        right_bt_rem = right.br_t_cos[mode_idx] if component == "cos" else right.br_t_sin[mode_idx]
        left_nu = 1.0 / (MU0 * left.mu_r)
        right_nu = 1.0 / (MU0 * right.mu_r)

        mat[row, li] = -left_nu * lda
        mat[row, li + 1] = -left_nu * ldb
        mat[row, ri] = right_nu * rda
        mat[row, ri + 1] = right_nu * rdb
        rhs[row] = (
            left_nu * (ldp + left_bt_rem)
            - right_nu * (rdp + right_bt_rem)
        )
        row += 1

    # Outer stator bore is treated as A=0 for the stage-1 slotless model.
    dout = domains[-1]
    pa, pb, pp, _, _, _ = basis_values(dout.r_outer, lam, r_ref, p(dout))
    mat[row, -2] = pa
    mat[row, -1] = pb
    rhs[row] = -pp

    sol = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    cond = np.linalg.cond(mat)
    return sol, cond


def solve_all_modes(domains, lambdas, r_ref, source_sign):
    cos_solutions = []
    sin_solutions = []
    conds = []
    for idx, lam in enumerate(lambdas):
        cos_sol, cos_cond = solve_mode(domains, lam, idx, r_ref, source_sign, "cos")
        sin_sol, sin_cond = solve_mode(domains, lam, idx, r_ref, source_sign, "sin")
        cos_solutions.append(cos_sol)
        sin_solutions.append(sin_sol)
        conds.append(max(cos_cond, sin_cond))
    return np.array(cos_solutions), np.array(sin_solutions), np.array(conds)


def evaluate_airgap(theta, radius_m, domains, lambdas, cos_solutions, sin_solutions, r_ref, source_sign):
    airgap_idx = next(idx for idx, domain in enumerate(domains) if domain.name == "airgap")
    br = np.zeros_like(theta)
    bt = np.zeros_like(theta)
    for mode_idx, lam in enumerate(lambdas):
        domain = domains[airgap_idx]
        pc, ps = mode_particular_coeff(domain, mode_idx, lam, r_ref, source_sign)
        a, b = cos_solutions[mode_idx, 2 * airgap_idx : 2 * airgap_idx + 2]
        c, d = sin_solutions[mode_idx, 2 * airgap_idx : 2 * airgap_idx + 2]

        va, vb, vp_c, da, db, dp_c = basis_values(radius_m, lam, r_ref, pc)
        _, _, vp_s, _, _, dp_s = basis_values(radius_m, lam, r_ref, ps)

        cos_t = np.cos(lam * theta)
        sin_t = np.sin(lam * theta)
        az_cos_amp = a * va + b * vb + vp_c
        az_sin_amp = c * va + d * vb + vp_s
        da_cos_amp = a * da + b * db + dp_c
        da_sin_amp = c * da + d * db + dp_s

        br += (lam / radius_m) * (-az_cos_amp * sin_t + az_sin_amp * cos_t)
        bt += -(da_cos_amp * cos_t + da_sin_amp * sin_t)
    return br, bt


def read_benchmark(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    theta = np.radians([float(row["angle_deg"]) for row in rows])
    br = np.array([float(row["Br_T"]) for row in rows])
    bt = np.array([float(row["Bt_T"]) for row in rows])
    return theta, br, bt


def stats(values):
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def relative_l2(model, ref):
    denom = np.linalg.norm(ref)
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(model - ref) / denom)


def best_scalar(model, ref):
    denom = float(np.dot(model, model))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(model, ref) / denom)


def write_outputs(out_dir, theta, br, bt, benchmark=None, metadata=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "stage1_slotless_br_bt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for angle, br_value, bt_value in zip(np.degrees(theta), br, bt):
            writer.writerow([angle, br_value, bt_value])

    summary_path = out / "stage1_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        if metadata:
            for key, value in metadata.items():
                writer.writerow([key, value])
        for prefix, values in (("model_Br", stats(br)), ("model_Bt", stats(bt))):
            for key, value in values.items():
                writer.writerow([f"{prefix}_{key}", value])
        if benchmark is not None:
            _, br_ref, bt_ref = benchmark
            for prefix, values in (("femm_Br", stats(br_ref)), ("femm_Bt", stats(bt_ref))):
                for key, value in values.items():
                    writer.writerow([f"{prefix}_{key}", value])
            writer.writerow(["Br_relative_l2", relative_l2(br, br_ref)])
            writer.writerow(["Bt_relative_l2", relative_l2(bt, bt_ref)])
            br_scale = best_scalar(br, br_ref)
            bt_scale = best_scalar(bt, bt_ref)
            writer.writerow(["Br_best_scalar_to_FEMM", br_scale])
            writer.writerow(["Bt_best_scalar_to_FEMM", bt_scale])
            writer.writerow(["Br_relative_l2_after_best_scalar", relative_l2(br * br_scale, br_ref)])
            writer.writerow(["Bt_relative_l2_after_best_scalar", relative_l2(bt * bt_scale, bt_ref)])

    png_path = out / "stage1_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    angles = np.degrees(theta)
    axes[0].plot(angles, br, label="stage1 Br")
    axes[1].plot(angles, bt, label="stage1 Bt")
    if benchmark is not None:
        _, br_ref, bt_ref = benchmark
        axes[0].plot(angles, br_ref, "--", label="FEMM Br")
        axes[1].plot(angles, bt_ref, "--", label="FEMM Bt")
    axes[0].set_ylabel("Br (T)")
    axes[1].set_ylabel("Bt (T)")
    axes[1].set_xlabel("Mechanical angle (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return csv_path, summary_path, png_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 slotless polar subdomain baseline for paper [5] V-shape equivalent model."
    )
    parser.add_argument("--benchmark-csv", default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--harmonics", type=int, default=8)
    parser.add_argument("--brem-t", type=float, default=base.PAPER_SPECS["vshape"].magnet_remanence_t)
    parser.add_argument("--mu-core", type=float, default=1000.0)
    parser.add_argument(
        "--mu-bridge",
        type=float,
        default=1.0,
        help="Placeholder fixed bridge permeability. Replace with a FEMM-sampled value before treating amplitudes as validated.",
    )
    parser.add_argument("--mu-pm", type=float, default=1.05)
    parser.add_argument("--source-sign", type=float, default=-1.0)
    parser.add_argument("--airgap-radius-mm", type=float, default=39.4)
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
    return parser.parse_args()


def main():
    args = parse_args()
    geom = load_equivalent_geometry(args)
    lambdas = harmonic_lambdas(args.harmonics, geom.sector_span_rad)
    coeffs = profile_coefficients(geom, args.harmonics, args.brem_t)
    domains = build_domains(geom, args, coeffs)
    r_ref = geom.rr_m
    cos_solutions, sin_solutions, conds = solve_all_modes(domains, lambdas, r_ref, args.source_sign)

    benchmark = read_benchmark(args.benchmark_csv)
    if benchmark is None:
        theta = np.linspace(geom.sector_start_rad, geom.sector_end_rad, 301)
    else:
        theta = benchmark[0]

    br, bt = evaluate_airgap(
        theta,
        meters(args.airgap_radius_mm),
        domains,
        lambdas,
        cos_solutions,
        sin_solutions,
        r_ref,
        args.source_sign,
    )
    csv_path, summary_path, png_path = write_outputs(
        args.result_dir,
        theta,
        br,
        bt,
        benchmark,
        metadata={
            "source_audit": "paper5_source_audit.md",
            "mu_bridge": args.mu_bridge,
            "mu_bridge_source": "CLI/user input; default 1.0 is a placeholder",
            "brem_t": args.brem_t,
            "brem_t_source": "build_paper_vshape_model.py PAPER_SPECS['vshape'].magnet_remanence_t unless overridden",
        },
    )

    print("=== Paper [5] stage-1 slotless baseline ===")
    print(f"sector = {math.degrees(geom.sector_start_rad):g}..{math.degrees(geom.sector_end_rad):g} deg")
    print(f"pole center = {math.degrees(geom.rotor_center_rad):.6g} deg")
    print(
        "radii mm: "
        f"shaft={geom.shaft_radius_m * 1e3:.6g}, Rf={geom.rf_m * 1e3:.6g}, "
        f"Rm={geom.rm_m * 1e3:.6g}, Rl-w1={geom.rl_minus_w1_m * 1e3:.6g}, "
        f"Rr={geom.rr_m * 1e3:.6g}, Rs={geom.rs_m * 1e3:.6g}, "
        f"Rso={geom.rso_m * 1e3:.6g}"
    )
    print(f"harmonics = {args.harmonics}; max condition = {np.max(conds):.3e}")
    print(f"mu_bridge = {args.mu_bridge:g} (CLI/user input; default is only a placeholder)")
    print(f"model Br rms = {stats(br)['rms']:.6g} T; Bt rms = {stats(bt)['rms']:.6g} T")
    if benchmark is not None:
        _, br_ref, bt_ref = benchmark
        print(f"FEMM Br rms = {stats(br_ref)['rms']:.6g} T; Bt rms = {stats(bt_ref)['rms']:.6g} T")
        print(f"relative L2: Br = {relative_l2(br, br_ref):.6g}; Bt = {relative_l2(bt, bt_ref):.6g}")
        br_scale = best_scalar(br, br_ref)
        bt_scale = best_scalar(bt, bt_ref)
        print(
            "best scalar to FEMM: "
            f"Br x {br_scale:.6g} -> L2 {relative_l2(br * br_scale, br_ref):.6g}; "
            f"Bt x {bt_scale:.6g} -> L2 {relative_l2(bt * bt_scale, bt_ref):.6g}"
        )
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()

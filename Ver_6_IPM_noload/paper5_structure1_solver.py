import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_paper_vshape_model as base
import V_shape_equavalent as equivalent
from paper5_vshape_stage1 import read_benchmark, relative_l2, best_scalar, stats


MU0 = 4.0 * math.pi * 1e-7
DEFAULT_RESULT_DIR = "results/paper5_structure1"
DEFAULT_BENCHMARK_CSV = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"


@dataclass(frozen=True)
class Structure1Params:
    p: int
    rf: float
    rb: float
    rr: float
    rs: float
    alpha_prime: float
    gamma_prime: float
    alpha: float
    alpha1: float
    gamma: float
    brem: float
    mu_m: float
    mu_b: float


def mm_to_m(value):
    return value * 1e-3


def integrate(func, start, end, samples=2401):
    theta = np.linspace(start, end, samples)
    return float(np.trapezoid(func(theta), theta))


def build_params(args):
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
    pole_pairs = spec.poles // 2
    beta = dims["alpha_rad"]
    alpha1 = dims["alpha1_rad"]
    alpha = dims["delta_rad"]
    alpha_prime = math.pi / pole_pairs - alpha1
    gamma_prime = alpha_prime + spec.bridge_width_1 / spec.rotor_outer_radius
    gamma = spec.bridge_length_1 / (spec.rotor_outer_radius - spec.bridge_width_1)
    gamma += 2.0 * spec.bridge_width_1 / spec.rotor_outer_radius

    return Structure1Params(
        p=pole_pairs,
        rf=mm_to_m(dims["rm"]),
        rb=mm_to_m(spec.rotor_outer_radius - spec.bridge_width_1),
        rr=mm_to_m(spec.rotor_outer_radius),
        rs=mm_to_m(spec.stator_inner_radius),
        alpha_prime=alpha_prime,
        gamma_prime=gamma_prime,
        alpha=alpha,
        alpha1=alpha1,
        gamma=gamma,
        brem=args.brem_t,
        mu_m=MU0 * args.mu_pm,
        mu_b=MU0 * args.mu_bridge,
    )


def compute_gammas(params, n_count, k_count, g_count):
    p = params.p
    ap = params.alpha_prime
    gp = params.gamma_prime
    center = math.pi / (2.0 * p)
    a0 = center - ap / 2.0
    a1 = center + ap / 2.0
    g0 = center - gp / 2.0
    g1 = center + gp / 2.0

    gamma1 = np.zeros(n_count)
    gamma2 = np.zeros((n_count, k_count))
    gamma3 = np.zeros(k_count)
    gamma4 = np.zeros(g_count)
    gamma5 = np.zeros((k_count, g_count))
    gamma6 = np.zeros(n_count)
    gamma7 = np.zeros((n_count, g_count))

    for ni in range(n_count):
        n = ni + 1
        gamma1[ni] = integrate(lambda th, n=n: np.sin(n * p * th), a0, a1)
        gamma6[ni] = integrate(lambda th, n=n: np.sin(n * p * th), g0, g1)
        for ki in range(k_count):
            k = ki + 1
            gamma2[ni, ki] = integrate(
                lambda th, n=n, k=k: np.cos(k * math.pi / ap * (th - center + ap / 2.0))
                * np.sin(n * p * th),
                a0,
                a1,
            )
        for gi in range(g_count):
            g = gi + 1
            gamma7[ni, gi] = integrate(
                lambda th, n=n, g=g: np.sin(n * p * th)
                * np.cos(g * math.pi / gp * (th - center + gp / 2.0)),
                g0,
                g1,
            )

    for ki in range(k_count):
        k = ki + 1
        gamma3[ki] = integrate(
            lambda th, k=k: np.cos(k * math.pi / ap * (th - center + ap / 2.0)),
            a0,
            a1,
        )
        for gi in range(g_count):
            g = gi + 1
            gamma5[ki, gi] = integrate(
                lambda th, k=k, g=g: np.cos(k * math.pi / ap * (th - center + ap / 2.0))
                * np.cos(g * math.pi / gp * (th - center + gp / 2.0)),
                a0,
                a1,
            )

    for gi in range(g_count):
        g = gi + 1
        gamma4[gi] = integrate(
            lambda th, g=g: np.cos(g * math.pi / gp * (th - center + gp / 2.0)),
            a0,
            a1,
        )

    return gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7


class Index:
    def __init__(self, n_count, k_count, g_count):
        cursor = 0
        self.a1 = slice(cursor, cursor + n_count)
        cursor += n_count
        self.a02 = cursor
        self.b02 = cursor + 1
        cursor += 2
        self.a2 = slice(cursor, cursor + k_count)
        cursor += k_count
        self.b2 = slice(cursor, cursor + k_count)
        cursor += k_count
        self.a03 = cursor
        self.b03 = cursor + 1
        cursor += 2
        self.a3 = slice(cursor, cursor + g_count)
        cursor += g_count
        self.b3 = slice(cursor, cursor + g_count)
        cursor += g_count
        self.a4 = slice(cursor, cursor + n_count)
        cursor += n_count
        self.b4 = slice(cursor, cursor + n_count)
        cursor += n_count
        self.total = cursor


def add_row(rows, rhs, coeffs, value):
    rows.append(coeffs)
    rhs.append(value)


def assemble_system(params, n_count, k_count, g_count):
    idx = Index(n_count, k_count, g_count)
    rows = []
    rhs = []
    g1, g2, g3, g4, g5, g6, g7 = compute_gammas(params, n_count, k_count, g_count)
    p = params.p
    rf = params.rf
    rb = params.rb
    rr = params.rr
    rs = params.rs
    ap = params.alpha_prime
    gp = params.gamma_prime
    br = params.brem
    mu_m = params.mu_m
    mu_b = params.mu_b

    # Eq. (47)
    row = np.zeros(idx.total)
    row[idx.a02] = 1.0
    row[idx.b02] = math.log(rf)
    row[idx.a1] = -(1.0 / ap) * g1
    add_row(rows, rhs, row, -br * rf)

    # Eq. (48)
    for ki in range(k_count):
        k = ki + 1
        qk = k * math.pi / ap
        row = np.zeros(idx.total)
        row[idx.a2.start + ki] = 1.0
        row[idx.b2.start + ki] = (rf / rb) ** (-qk)
        row[idx.a1] = -(2.0 / ap) * g2[:, ki]
        add_row(rows, rhs, row, 0.0)

    # Eq. (49)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row = np.zeros(idx.total)
        row[idx.a1.start + ni] = lam / (MU0 * rf)
        row[idx.b02] = -(2.0 / (math.pi * mu_m)) * g1[ni] / rf
        for ki in range(k_count):
            k = ki + 1
            qk = k * math.pi / ap
            row[idx.a2.start + ki] += -(2.0 / (math.pi * mu_m)) * (qk / rf) * g2[ni, ki]
            row[idx.b2.start + ki] += (
                2.0
                / (math.pi * mu_m)
                * (qk / rb)
                * (rf / rb) ** (-qk - 1.0)
                * g2[ni, ki]
            )
        add_row(rows, rhs, row, (2.0 / (math.pi * mu_m)) * br * g1[ni])

    # Eq. (50)
    row = np.zeros(idx.total)
    row[idx.a03] = 1.0
    row[idx.b03] = math.log(rb)
    row[idx.a02] = -ap / gp
    row[idx.b02] = -(ap / gp) * math.log(rb)
    for ki in range(k_count):
        k = ki + 1
        qk = k * math.pi / ap
        row[idx.a2.start + ki] += -(1.0 / gp) * (rb / rf) ** qk * g3[ki]
        row[idx.b2.start + ki] += -(1.0 / gp) * g3[ki]
    add_row(rows, rhs, row, (ap / gp) * br * rb)

    # Eq. (51)
    for gi in range(g_count):
        g = gi + 1
        qg = g * math.pi / gp
        row = np.zeros(idx.total)
        row[idx.a3.start + gi] = 1.0
        row[idx.b3.start + gi] = (rb / rr) ** (-qg)
        row[idx.a02] = -(2.0 / gp) * g4[gi]
        row[idx.b02] = -(2.0 / gp) * math.log(rb) * g4[gi]
        for ki in range(k_count):
            k = ki + 1
            qk = k * math.pi / ap
            row[idx.a2.start + ki] += -(2.0 / gp) * (rb / rf) ** qk * g5[ki, gi]
            row[idx.b2.start + ki] += -(2.0 / gp) * g5[ki, gi]
        add_row(rows, rhs, row, (2.0 / gp) * br * rb * g4[gi])

    # Eq. (52)
    row = np.zeros(idx.total)
    row[idx.b02] = 1.0 / (mu_m * rb)
    row[idx.b03] = -1.0 / (mu_b * rb)
    for gi in range(g_count):
        g = gi + 1
        qg = g * math.pi / gp
        row[idx.a3.start + gi] += -(1.0 / (ap * mu_b)) * (qg / rb) * g4[gi]
        row[idx.b3.start + gi] += (
            (1.0 / (ap * mu_b))
            * (qg / rr)
            * (rb / rr) ** (-qg - 1.0)
            * g4[gi]
        )
    add_row(rows, rhs, row, -br / mu_m)

    # Eq. (53)
    for ki in range(k_count):
        k = ki + 1
        qk = k * math.pi / ap
        row = np.zeros(idx.total)
        row[idx.a2.start + ki] = (1.0 / mu_m) * (qk / rf) * (rb / rf) ** (qk - 1.0)
        row[idx.b2.start + ki] = -(1.0 / mu_m) * qk / rb
        row[idx.b03] = -2.0 * g3[ki] / (mu_b * ap * rb)
        for gi in range(g_count):
            g = gi + 1
            qg = g * math.pi / gp
            row[idx.a3.start + gi] += -(2.0 / (ap * mu_b)) * (qg / rb) * g5[ki, gi]
            row[idx.b3.start + gi] += (
                (2.0 / (ap * mu_b))
                * (qg / rr)
                * (rb / rr) ** (-qg - 1.0)
                * g5[ki, gi]
            )
        add_row(rows, rhs, row, 0.0)

    # Eq. (54)
    row = np.zeros(idx.total)
    row[idx.a03] = 1.0
    row[idx.b03] = math.log(rr)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row[idx.a4.start + ni] += -(1.0 / gp) * g6[ni]
        row[idx.b4.start + ni] += -(1.0 / gp) * (rr / rs) ** (-lam) * g6[ni]
    add_row(rows, rhs, row, 0.0)

    # Eq. (55)
    for gi in range(g_count):
        g = gi + 1
        qg = g * math.pi / gp
        row = np.zeros(idx.total)
        row[idx.a3.start + gi] = (rr / rb) ** qg
        row[idx.b3.start + gi] = 1.0
        for ni in range(n_count):
            n = ni + 1
            lam = n * p
            row[idx.a4.start + ni] += -(2.0 / gp) * g7[ni, gi]
            row[idx.b4.start + ni] += -(2.0 / gp) * (rr / rs) ** (-lam) * g7[ni, gi]
        add_row(rows, rhs, row, 0.0)

    # Eq. (56)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row = np.zeros(idx.total)
        row[idx.a4.start + ni] = -(1.0 / MU0) * lam / rr
        row[idx.b4.start + ni] = (1.0 / MU0) * (lam / rs) * (rr / rs) ** (-lam - 1.0)
        row[idx.b03] = 2.0 * g6[ni] / (mu_b * math.pi * rr)
        for gi in range(g_count):
            g = gi + 1
            qg = g * math.pi / gp
            row[idx.a3.start + gi] += (
                2.0
                / (mu_b * math.pi)
                * (qg / rb)
                * (rr / rb) ** (qg - 1.0)
                * g7[ni, gi]
            )
            row[idx.b3.start + gi] += -(2.0 / (mu_b * math.pi)) * (qg / rr) * g7[ni, gi]
        add_row(rows, rhs, row, 0.0)

    # Eq. (57)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row = np.zeros(idx.total)
        row[idx.a4.start + ni] = (1.0 / rr) * (rs / rr) ** (lam - 1.0)
        row[idx.b4.start + ni] = -1.0 / rs
        add_row(rows, rhs, row, 0.0)

    mat = np.vstack(rows)
    vec = np.array(rhs)
    return mat, vec, idx


def solve_structure1(params, n_count, k_count, g_count):
    mat, vec, idx = assemble_system(params, n_count, k_count, g_count)
    sol = np.linalg.solve(mat, vec)
    return sol, idx, np.linalg.cond(mat)


def evaluate_airgap(params, sol, idx, theta, radius):
    br = np.zeros_like(theta)
    bt = np.zeros_like(theta)
    a4 = sol[idx.a4]
    b4 = sol[idx.b4]
    for ni in range(len(a4)):
        n = ni + 1
        lam = n * params.p
        br += (
            lam * a4[ni] / radius * (radius / params.rr) ** lam
            + lam * b4[ni] / radius * (radius / params.rs) ** (-lam)
        ) * np.cos(lam * theta)
        bt += -(
            lam * a4[ni] / params.rr * (radius / params.rr) ** (lam - 1.0)
            - lam * b4[ni] / params.rs * (radius / params.rs) ** (-lam - 1.0)
        ) * np.sin(lam * theta)
    return br, bt


def apply_structure1_correction(params, theta_physical_rad, br, bt, center_rad, kmod):
    """Paper Eq. (24)-(27): bridge/core span correction for Structure 1."""
    span = params.alpha1 + params.alpha
    gamma = params.gamma
    inner_half = 0.5 * (span - gamma)
    outer_half = 0.5 * (span + gamma)
    max_abs_br = float(np.max(np.abs(br)))

    corrected_br = np.zeros_like(br)
    corrected_bt = np.zeros_like(bt)
    rel = (theta_physical_rad - center_rad + math.pi) % (2.0 * math.pi) - math.pi

    center_mask = np.abs(rel) <= inner_half
    left_mask = (-outer_half <= rel) & (rel < -inner_half)
    right_mask = (inner_half < rel) & (rel <= outer_half)
    bt_mask = np.abs(rel) <= outer_half

    corrected_br[center_mask] = br[center_mask]
    corrected_br[left_mask] = max_abs_br / gamma * (rel[left_mask] + outer_half)
    corrected_br[right_mask] = -max_abs_br / gamma * (rel[right_mask] - outer_half)
    corrected_bt[bt_mask] = bt[bt_mask]
    return corrected_br * kmod, corrected_bt * kmod


def write_outputs(out_dir, theta_deg, br, bt, benchmark, cond, params):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "structure1_br_bt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for a, br_v, bt_v in zip(theta_deg, br, bt):
            writer.writerow([a, br_v, bt_v])

    summary_path = out / "structure1_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        writer.writerow(["condition", cond])
        writer.writerow(["alpha_prime_deg", math.degrees(params.alpha_prime)])
        writer.writerow(["gamma_prime_deg", math.degrees(params.gamma_prime)])
        writer.writerow(["alpha_deg", math.degrees(params.alpha)])
        writer.writerow(["alpha1_deg", math.degrees(params.alpha1)])
        writer.writerow(["gamma_deg", math.degrees(params.gamma)])
        writer.writerow(["Rf_mm", params.rf * 1e3])
        writer.writerow(["Rb_mm", params.rb * 1e3])
        writer.writerow(["Rr_mm", params.rr * 1e3])
        writer.writerow(["Rs_mm", params.rs * 1e3])
        for prefix, values in (("model_Br", stats(br)), ("model_Bt", stats(bt))):
            for key, value in values.items():
                writer.writerow([f"{prefix}_{key}", value])
        if benchmark is not None:
            _, br_ref, bt_ref = benchmark
            for prefix, values in (("femm_Br", stats(br_ref)), ("femm_Bt", stats(bt_ref))):
                for key, value in values.items():
                    writer.writerow([f"{prefix}_{key}", value])
            br_scale = best_scalar(br, br_ref)
            bt_scale = best_scalar(bt, bt_ref)
            writer.writerow(["Br_relative_l2", relative_l2(br, br_ref)])
            writer.writerow(["Bt_relative_l2", relative_l2(bt, bt_ref)])
            writer.writerow(["Br_best_scalar_to_FEMM", br_scale])
            writer.writerow(["Bt_best_scalar_to_FEMM", bt_scale])
            writer.writerow(["Br_relative_l2_after_best_scalar", relative_l2(br * br_scale, br_ref)])
            writer.writerow(["Bt_relative_l2_after_best_scalar", relative_l2(bt * bt_scale, bt_ref)])

    png_path = out / "structure1_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(theta_deg, br, label="Structure 1 Br")
    axes[1].plot(theta_deg, bt, label="Structure 1 Bt")
    if benchmark is not None:
        _, br_ref, bt_ref = benchmark
        axes[0].plot(theta_deg, br_ref, "--", label="slotless FEMM Br")
        axes[1].plot(theta_deg, bt_ref, "--", label="slotless FEMM Bt")
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
    parser = argparse.ArgumentParser(description="Paper [5] Structure 1 solver from Appendix A.")
    parser.add_argument("--benchmark-csv", default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--n-harmonics", type=int, default=8)
    parser.add_argument("--k-harmonics", type=int, default=8)
    parser.add_argument("--g-harmonics", type=int, default=8)
    parser.add_argument("--brem-t", type=float, default=base.PAPER_SPECS["vshape"].magnet_remanence_t)
    parser.add_argument("--mu-pm", type=float, default=1.05)
    parser.add_argument("--mu-bridge", type=float, default=1.0)
    parser.add_argument("--airgap-radius-mm", type=float, default=39.4)
    parser.add_argument("--theta-offset-deg", type=float, default=0.0)
    parser.add_argument("--apply-correction", action="store_true")
    parser.add_argument("--kmod", type=float, default=1.0)
    parser.add_argument("--correction-center-deg", type=float, default=30.0)
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
    return parser.parse_args()


def main():
    args = parse_args()
    params = build_params(args)
    sol, idx, cond = solve_structure1(params, args.n_harmonics, args.k_harmonics, args.g_harmonics)
    benchmark = read_benchmark(args.benchmark_csv)
    if benchmark is None:
        theta_deg = np.linspace(0.01, 59.99, 301)
    else:
        theta_deg = np.degrees(benchmark[0])
    theta = np.radians(theta_deg + args.theta_offset_deg)
    br, bt = evaluate_airgap(params, sol, idx, theta, mm_to_m(args.airgap_radius_mm))
    if args.apply_correction:
        br, bt = apply_structure1_correction(
            params,
            np.radians(theta_deg),
            br,
            bt,
            math.radians(args.correction_center_deg),
            args.kmod,
        )
    csv_path, summary_path, png_path = write_outputs(args.result_dir, theta_deg, br, bt, benchmark, cond, params)

    print("=== Paper [5] Structure 1 Appendix-A solve ===")
    print(f"N/K/G = {args.n_harmonics}/{args.k_harmonics}/{args.g_harmonics}; condition = {cond:.3e}")
    print(
        f"alpha'={math.degrees(params.alpha_prime):.6g} deg, "
        f"gamma'={math.degrees(params.gamma_prime):.6g} deg"
    )
    print(f"mu_bridge = {args.mu_bridge:g}")
    if args.apply_correction:
        print(f"Structure 1 correction Eq.(24)-(27) enabled: Kmod={args.kmod:g}, center={args.correction_center_deg:g} deg")
    print(f"model Br rms = {stats(br)['rms']:.6g} T; Bt rms = {stats(bt)['rms']:.6g} T")
    if benchmark is not None:
        _, br_ref, bt_ref = benchmark
        print(f"FEMM Br rms = {stats(br_ref)['rms']:.6g} T; Bt rms = {stats(bt_ref)['rms']:.6g} T")
        print(f"relative L2: Br = {relative_l2(br, br_ref):.6g}; Bt = {relative_l2(bt, bt_ref):.6g}")
        br_scale = best_scalar(br, br_ref)
        bt_scale = best_scalar(bt, bt_ref)
        print(
            f"best scalar: Br x {br_scale:.6g} -> L2 {relative_l2(br * br_scale, br_ref):.6g}; "
            f"Bt x {bt_scale:.6g} -> L2 {relative_l2(bt * bt_scale, bt_ref):.6g}"
        )
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()

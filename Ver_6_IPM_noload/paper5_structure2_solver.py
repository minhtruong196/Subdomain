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
DEFAULT_RESULT_DIR = "results/paper5_structure2"
DEFAULT_BENCHMARK_CSV = "results/paper5_slotless_equivalent_1over6/br_bt_arc.csv"


@dataclass(frozen=True)
class Structure2Params:
    p: int
    rm_inner: float
    rf_outer: float
    rr: float
    rs: float
    alpha: float
    alpha1: float
    gamma: float
    span: float
    brem: float
    mu_m: float


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
    alpha = dims["delta_rad"]
    alpha1 = dims["alpha1_rad"]
    gamma = spec.bridge_length_1 / (spec.rotor_outer_radius - spec.bridge_width_1)
    gamma += 2.0 * spec.bridge_width_1 / spec.rotor_outer_radius
    span = alpha1 + alpha + gamma

    # Paper notation: Rf is the PM outer radius and Rm is the PM inner radius.
    return Structure2Params(
        p=pole_pairs,
        rm_inner=mm_to_m(dims["rf"]),
        rf_outer=mm_to_m(dims["rm"]),
        rr=mm_to_m(spec.rotor_outer_radius),
        rs=mm_to_m(spec.stator_inner_radius),
        alpha=alpha,
        alpha1=alpha1,
        gamma=gamma,
        span=span,
        brem=args.brem_t,
        mu_m=MU0 * args.mu_pm,
    )


class Index:
    def __init__(self, n_count, m_count, k_count):
        cursor = 0
        self.a5 = slice(cursor, cursor + n_count)
        cursor += n_count
        self.b5 = slice(cursor, cursor + n_count)
        cursor += n_count
        self.a06 = cursor
        self.b06 = cursor + 1
        cursor += 2
        self.a6 = slice(cursor, cursor + m_count)
        cursor += m_count
        self.b6 = slice(cursor, cursor + m_count)
        cursor += m_count
        self.a7 = slice(cursor, cursor + k_count)
        cursor += k_count
        self.b7 = slice(cursor, cursor + k_count)
        cursor += k_count
        self.total = cursor


def compute_gammas(params, n_count, m_count, k_count):
    p = params.p
    gamma = params.gamma
    span = params.span
    center = 0.5 * (params.alpha1 + params.alpha)
    start = center - 0.5 * gamma
    end = center + 0.5 * gamma

    gamma8 = np.zeros(n_count)
    gamma9 = np.zeros((n_count, m_count))
    gamma10 = np.zeros(k_count)
    gamma11 = np.zeros((m_count, k_count))

    for ni in range(n_count):
        n = ni + 1
        gamma8[ni] = integrate(lambda th, n=n: np.sin(n * p * th), start, end)
        for mi in range(m_count):
            m = mi + 1
            qm = 2.0 * m * math.pi / gamma
            gamma9[ni, mi] = integrate(
                lambda th, n=n, qm=qm: np.sin(n * p * th) * np.cos(qm * (th - center)),
                start,
                end,
            )

    for ki in range(k_count):
        k = ki
        qk = (2 * k + 1) * math.pi / span
        gamma10[ki] = integrate(lambda th, qk=qk: np.sin(qk * th), start, end)
        for mi in range(m_count):
            m = mi + 1
            qm = 2.0 * m * math.pi / gamma
            gamma11[mi, ki] = integrate(
                lambda th, qk=qk, qm=qm: np.sin(qk * th) * np.cos(qm * (th - center)),
                start,
                end,
            )

    return gamma8, gamma9, gamma10, gamma11


def delta_k(params, k):
    span = params.span
    odd = 2 * k + 1
    return (
        ((-1.0) ** k)
        * 4.0
        * params.brem
        * span
        / (odd * odd * math.pi * math.pi - span * span)
    )


def add_row(rows, rhs, row, value):
    rows.append(row)
    rhs.append(value)


def assemble_system(params, n_count, m_count, k_count):
    idx = Index(n_count, m_count, k_count)
    rows = []
    rhs = []
    g8, g9, g10, g11 = compute_gammas(params, n_count, m_count, k_count)

    p = params.p
    ri = params.rm_inner
    rf = params.rf_outer
    rr = params.rr
    rs = params.rs
    gamma = params.gamma
    span = params.span
    mu_m = params.mu_m

    # Eq. (65)
    row = np.zeros(idx.total)
    row[idx.a06] = 1.0
    row[idx.b06] = math.log(rr)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row[idx.a5.start + ni] += -(1.0 / gamma) * (rr / rs) ** lam * g8[ni]
        row[idx.b5.start + ni] += -(1.0 / gamma) * g8[ni]
    add_row(rows, rhs, row, 0.0)

    # Eq. (66)
    for mi in range(m_count):
        m = mi + 1
        qm = 2.0 * m * math.pi / gamma
        row = np.zeros(idx.total)
        row[idx.a6.start + mi] = 1.0
        row[idx.b6.start + mi] = (rf / rr) ** qm
        for ni in range(n_count):
            n = ni + 1
            lam = n * p
            row[idx.a5.start + ni] += -(2.0 / gamma) * (rr / rs) ** lam * g9[ni, mi]
            row[idx.b5.start + ni] += -(2.0 / gamma) * g9[ni, mi]
        add_row(rows, rhs, row, 0.0)

    # Eq. (67)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row = np.zeros(idx.total)
        row[idx.a5.start + ni] = -(n / MU0) * (rr / rs) ** lam
        row[idx.b5.start + ni] = n / MU0
        row[idx.b06] = -(2.0 / (math.pi * mu_m)) * (((-1.0) ** n) - 1.0) * g8[ni]
        for mi in range(m_count):
            m = mi + 1
            qm = 2.0 * m * math.pi / gamma
            row[idx.a6.start + mi] += (
                -(2.0 / (math.pi * mu_m))
                * (((-1.0) ** n) - 1.0)
                * qm
                * g9[ni, mi]
            )
            row[idx.b6.start + mi] += (
                (2.0 / (math.pi * mu_m))
                * (((-1.0) ** n) - 1.0)
                * qm
                * (rf / rr) ** qm
                * g9[ni, mi]
            )
        add_row(rows, rhs, row, 0.0)

    # Eq. (68)
    row = np.zeros(idx.total)
    row[idx.a06] = 1.0
    row[idx.b06] = math.log(rf)
    for ki in range(k_count):
        k = ki
        qk = (2 * k + 1) * math.pi / span
        dk = delta_k(params, k)
        row[idx.a7.start + ki] += -(1.0 / gamma) * g10[ki]
        row[idx.b7.start + ki] += -(1.0 / gamma) * (ri / rf) ** qk * g10[ki]
        # Eq. (68) has -Rf*Delta_k inside the bracket on RHS.
        rhs_part = -(1.0 / gamma) * rf * dk * g10[ki]
        # Move RHS contribution to the equation right-hand side.
        # row terms are left - RHS_terms = RHS_constant.
        if ki == 0:
            total_rhs = 0.0
        total_rhs += rhs_part
    add_row(rows, rhs, row, total_rhs)

    # Eq. (69)
    for mi in range(m_count):
        m = mi + 1
        qm = 2.0 * m * math.pi / gamma
        row = np.zeros(idx.total)
        row[idx.a6.start + mi] = (rf / rr) ** qm
        row[idx.b6.start + mi] = 1.0
        value = 0.0
        for ki in range(k_count):
            k = ki
            qk = (2 * k + 1) * math.pi / span
            dk = delta_k(params, k)
            row[idx.a7.start + ki] += -(2.0 / gamma) * g11[mi, ki]
            row[idx.b7.start + ki] += -(2.0 / gamma) * (ri / rf) ** qk * g11[mi, ki]
            value += -(2.0 / gamma) * rf * dk * g11[mi, ki]
        add_row(rows, rhs, row, value)

    # Eq. (70)
    for ki in range(k_count):
        k = ki
        qk = (2 * k + 1) * math.pi / span
        dk = delta_k(params, k)
        row = np.zeros(idx.total)
        row[idx.a7.start + ki] = qk / mu_m
        row[idx.b7.start + ki] = -(qk / mu_m) * (ri / rf) ** qk
        row[idx.b06] = -(4.0 / (MU0 * span)) * g10[ki]
        for mi in range(m_count):
            m = mi + 1
            qm = 2.0 * m * math.pi / gamma
            row[idx.a6.start + mi] += -(4.0 / (MU0 * span)) * qm * (rf / rr) ** qm * g11[mi, ki]
            row[idx.b6.start + mi] += (4.0 / (MU0 * span)) * qm * g11[mi, ki]
        add_row(rows, rhs, row, rf * dk / mu_m)

    # Eq. (71)
    for ni in range(n_count):
        n = ni + 1
        lam = n * p
        row = np.zeros(idx.total)
        row[idx.a5.start + ni] = 1.0 / rs
        row[idx.b5.start + ni] = -(1.0 / rr) * (rs / rr) ** (-lam - 1.0)
        add_row(rows, rhs, row, 0.0)

    # Eq. (72)
    for ki in range(k_count):
        k = ki
        qk = (2 * k + 1) * math.pi / span
        row = np.zeros(idx.total)
        row[idx.a7.start + ki] = qk * (1.0 / rf) * (ri / rf) ** (qk - 1.0)
        row[idx.b7.start + ki] = -qk / ri
        add_row(rows, rhs, row, delta_k(params, k))

    mat = np.vstack(rows)
    vec = np.array(rhs)
    return mat, vec, idx


def solve_structure2(params, n_count, m_count, k_count):
    mat, vec, idx = assemble_system(params, n_count, m_count, k_count)
    sol = np.linalg.solve(mat, vec)
    return sol, idx, np.linalg.cond(mat)


def evaluate_airgap(params, sol, idx, theta, radius):
    br = np.zeros_like(theta)
    bt = np.zeros_like(theta)
    a5 = sol[idx.a5]
    b5 = sol[idx.b5]
    for ni in range(len(a5)):
        n = ni + 1
        lam = n * params.p
        br += (
            lam * a5[ni] / radius * (radius / params.rs) ** lam
            + lam * b5[ni] / radius * (radius / params.rr) ** (-lam)
        ) * np.cos(lam * theta)
        bt += -(
            lam * a5[ni] / params.rs * (radius / params.rs) ** (lam - 1.0)
            - lam * b5[ni] / params.rr * (radius / params.rr) ** (-lam - 1.0)
        ) * np.sin(lam * theta)
    return br, bt


def write_outputs(out_dir, theta_deg, br, bt, benchmark, cond, params):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "structure2_br_bt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "Br_T", "Bt_T"])
        for a, br_v, bt_v in zip(theta_deg, br, bt):
            writer.writerow([a, br_v, bt_v])

    summary_path = out / "structure2_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        writer.writerow(["condition", cond])
        writer.writerow(["alpha_deg", math.degrees(params.alpha)])
        writer.writerow(["alpha1_deg", math.degrees(params.alpha1)])
        writer.writerow(["gamma_deg", math.degrees(params.gamma)])
        writer.writerow(["span_deg", math.degrees(params.span)])
        writer.writerow(["Rm_inner_mm", params.rm_inner * 1e3])
        writer.writerow(["Rf_outer_mm", params.rf_outer * 1e3])
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

    png_path = out / "structure2_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(theta_deg, br, label="Structure 2 Br")
    axes[1].plot(theta_deg, bt, label="Structure 2 Bt")
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
    parser = argparse.ArgumentParser(description="Paper [5] Structure 2 solver from Appendix B.")
    parser.add_argument("--benchmark-csv", default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--n-harmonics", type=int, default=5)
    parser.add_argument("--m-harmonics", type=int, default=5)
    parser.add_argument("--k-harmonics", type=int, default=5)
    parser.add_argument("--brem-t", type=float, default=base.PAPER_SPECS["vshape"].magnet_remanence_t)
    parser.add_argument("--mu-pm", type=float, default=1.05)
    parser.add_argument("--airgap-radius-mm", type=float, default=39.4)
    parser.add_argument("--theta-offset-deg", type=float, default=0.0)
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
    sol, idx, cond = solve_structure2(params, args.n_harmonics, args.m_harmonics, args.k_harmonics)
    benchmark = read_benchmark(args.benchmark_csv)
    if benchmark is None:
        theta_deg = np.linspace(0.01, 59.99, 301)
    else:
        theta_deg = np.degrees(benchmark[0])
    theta = np.radians(theta_deg + args.theta_offset_deg)
    br, bt = evaluate_airgap(params, sol, idx, theta, mm_to_m(args.airgap_radius_mm))
    csv_path, summary_path, png_path = write_outputs(args.result_dir, theta_deg, br, bt, benchmark, cond, params)

    print("=== Paper [5] Structure 2 Appendix-B solve ===")
    print(f"N/M/K = {args.n_harmonics}/{args.m_harmonics}/{args.k_harmonics}; condition = {cond:.3e}")
    print(
        f"alpha={math.degrees(params.alpha):.6g} deg, "
        f"alpha1={math.degrees(params.alpha1):.6g} deg, "
        f"gamma={math.degrees(params.gamma):.6g} deg"
    )
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

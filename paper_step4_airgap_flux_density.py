from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from paper_step3_boundary_matrix import (
    BoundaryMatrixParams,
    UnknownLayout,
    selected_segment_radii_m,
    solve_boundary_matrix,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AirgapFluxParams:
    """Step 4 parameters for no-load air-gap flux density."""

    boundary: BoundaryMatrixParams = BoundaryMatrixParams()
    # Paper value from Eq. (30)'s text.
    airgap_radius_m: float = 88.4e-3
    sample_count: int = 721
    output_dir: str = "outputs/step4_airgap_flux_density"


def log_abs_e(order: float, x: float, y: float) -> tuple[float, float]:
    """Return sign and log(abs(E_order(x,y))) without overflowing."""

    a = float(order * np.log(x / y))
    if np.isclose(a, 0.0):
        return 0.0, -np.inf
    sign = 1.0 if a > 0.0 else -1.0
    abs_a = abs(a)
    if abs_a > 25.0:
        return sign, abs_a
    return sign, float(np.log(abs(2.0 * np.sinh(a))))


def p_over_e(order: float, p_x: float, p_y: float, e_x: float, e_y: float) -> float:
    """Return P_order(p_x,p_y) / E_order(e_x,e_y)."""

    a = float(order * np.log(p_x / p_y))
    log_p = float(np.logaddexp(a, -a))
    sign_e, log_abs_den = log_abs_e(order, e_x, e_y)
    return float(sign_e * np.exp(log_p - log_abs_den))


def e_over_e(order: float, n_x: float, n_y: float, d_x: float, d_y: float) -> float:
    """Return E_order(n_x,n_y) / E_order(d_x,d_y)."""

    sign_num, log_abs_num = log_abs_e(order, n_x, n_y)
    if sign_num == 0.0:
        return 0.0
    sign_den, log_abs_den = log_abs_e(order, d_x, d_y)
    return float(sign_num * sign_den * np.exp(log_abs_num - log_abs_den))


def az2_radial_terms(
    r: float,
    Ru: float,
    Rs: float,
    mc_values: np.ndarray,
    Am2: np.ndarray,
    Bm2: np.ndarray,
    Cm2: np.ndarray,
    Dm2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return alpha, beta and their r-derivatives for Eq. (21)."""

    alpha = np.empty_like(mc_values, dtype=float)
    beta = np.empty_like(mc_values, dtype=float)
    alpha_prime = np.empty_like(mc_values, dtype=float)
    beta_prime = np.empty_like(mc_values, dtype=float)

    for idx, mc in enumerate(mc_values):
        pm_rs = p_over_e(mc, r, Rs, Ru, Rs)
        pm_ru = p_over_e(mc, r, Ru, Rs, Ru)
        em_rs = e_over_e(mc, r, Rs, Ru, Rs)
        em_ru = e_over_e(mc, r, Ru, Rs, Ru)

        a_part = Ru / mc * pm_rs
        b_part = Rs / mc * pm_ru
        a_prime_part = Ru / r * em_rs
        b_prime_part = Rs / r * em_ru

        alpha[idx] = Am2[idx] * a_part + Bm2[idx] * b_part
        beta[idx] = Cm2[idx] * a_part + Dm2[idx] * b_part
        alpha_prime[idx] = Am2[idx] * a_prime_part + Bm2[idx] * b_prime_part
        beta_prime[idx] = Cm2[idx] * a_prime_part + Dm2[idx] * b_prime_part

    return alpha, beta, alpha_prime, beta_prime


def az2_airgap(
    r: float,
    theta_mech_rad: np.ndarray,
    solution: np.ndarray,
    layout: UnknownLayout,
    mc_values: np.ndarray,
    Ru: float,
    Rs: float,
) -> np.ndarray:
    """Magnetic vector potential in Subdomain II from Eq. (21)."""

    Am2 = solution[layout.off_Am2 : layout.off_Am2 + layout.m_count]
    Bm2 = solution[layout.off_Bm2 : layout.off_Bm2 + layout.m_count]
    Cm2 = solution[layout.off_Cm2 : layout.off_Cm2 + layout.m_count]
    Dm2 = solution[layout.off_Dm2 : layout.off_Dm2 + layout.m_count]
    alpha, beta, _, _ = az2_radial_terms(r, Ru, Rs, mc_values, Am2, Bm2, Cm2, Dm2)

    phase = np.outer(mc_values, theta_mech_rad)
    return np.sum(alpha[:, None] * np.cos(phase) + beta[:, None] * np.sin(phase), axis=0)


def segment_flux_density(
    r: float,
    theta_mech_rad: np.ndarray,
    solution: np.ndarray,
    layout: UnknownLayout,
    mc_values: np.ndarray,
    Ru: float,
    Rs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Air-gap Br(j), Btheta(j) from Eq. (28)."""

    Am2 = solution[layout.off_Am2 : layout.off_Am2 + layout.m_count]
    Bm2 = solution[layout.off_Bm2 : layout.off_Bm2 + layout.m_count]
    Cm2 = solution[layout.off_Cm2 : layout.off_Cm2 + layout.m_count]
    Dm2 = solution[layout.off_Dm2 : layout.off_Dm2 + layout.m_count]
    alpha, beta, alpha_prime, beta_prime = az2_radial_terms(r, Ru, Rs, mc_values, Am2, Bm2, Cm2, Dm2)

    phase = np.outer(mc_values, theta_mech_rad)
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)

    # Eq. (28): Br = (1/r) dAz2/dtheta.
    Br = np.sum((-mc_values * alpha / r)[:, None] * sin_phase + (mc_values * beta / r)[:, None] * cos_phase, axis=0)

    # Eq. (28): Btheta = -dAz2/dr.
    Btheta = -np.sum(alpha_prime[:, None] * cos_phase + beta_prime[:, None] * sin_phase, axis=0)
    return Br, Btheta


def total_no_load_flux_density(params: AirgapFluxParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Superpose all Nz segment-pair fields according to Eq. (29)."""

    p = params.boundary.geometry.pole_pairs
    theta_elec_deg = np.linspace(-90.0, 90.0, params.sample_count)
    # Fig. 9 uses electrical position centered on the PM d-axis. Therefore
    # 0 Elec.Deg. maps to theta = 0, not to the middle of [0, pi/p].
    theta_mech_rad = np.deg2rad(theta_elec_deg) / p

    Br_total = np.zeros_like(theta_mech_rad)
    Btheta_total = np.zeros_like(theta_mech_rad)
    residuals: list[float] = []

    for segment_j in range(1, params.boundary.geometry.Nz + 1):
        boundary_params = BoundaryMatrixParams(
            geometry=params.boundary.geometry,
            segment_j=segment_j,
            max_pole_harmonic=params.boundary.max_pole_harmonic,
            slot_harmonics=params.boundary.slot_harmonics,
            magnetization_model=params.boundary.magnetization_model,
            Br_T=params.boundary.Br_T,
            mu0=params.boundary.mu0,
            mu_r=params.boundary.mu_r,
            stator_inner_radius_m=params.boundary.stator_inner_radius_m,
            slot_depth_m=params.boundary.slot_depth_m,
            slot_width_m=params.boundary.slot_width_m,
            Jui_A_per_m2=params.boundary.Jui_A_per_m2,
            Jdi_A_per_m2=params.boundary.Jdi_A_per_m2,
            delta_rad=params.boundary.delta_rad,
            output_dir=params.boundary.output_dir,
        )
        solution, _, _, layout, meta, residual = solve_boundary_matrix(boundary_params)
        residuals.append(residual)
        Ru, _ = selected_segment_radii_m(boundary_params)
        mc_values = np.asarray(meta["mc_values"], dtype=float)

        Br_j, Btheta_j = segment_flux_density(
            params.airgap_radius_m,
            theta_mech_rad,
            solution,
            layout,
            mc_values,
            Ru,
            boundary_params.R_s_m,
        )
        Br_total += Br_j
        Btheta_total += Btheta_j

    return theta_elec_deg, Br_total, Btheta_total, residuals


def save_flux_outputs(
    theta_elec_deg: np.ndarray,
    Br: np.ndarray,
    Btheta: np.ndarray,
    residuals: list[float],
    params: AirgapFluxParams,
) -> tuple[Path, Path]:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = params.boundary.magnetization_model

    data_path = output_dir / f"fig9_no_load_flux_density_{model}_data.npz"
    np.savez_compressed(
        data_path,
        theta_elec_deg=theta_elec_deg,
        Br_T=Br,
        Btheta_T=Btheta,
        residuals=np.asarray(residuals),
        airgap_radius_m=params.airgap_radius_m,
        magnetization_model=model,
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), constrained_layout=True)
    axes[0].plot(
        theta_elec_deg,
        Br,
        color="#E68613",
        linewidth=2.2,
        label=f"{model}: Br max={np.max(Br):.4f} T",
    )
    axes[0].set_ylabel("Radial flux density (T)", fontsize=12)
    axes[0].set_xlim(-90, 90)
    axes[0].set_ylim(min(-0.05, float(np.min(Br)) * 1.1), max(1.0, float(np.max(Br)) * 1.1))
    axes[0].set_xticks([-90, -45, 0, 45, 90])
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=11, loc="upper right")
    axes[0].text(0.5, -0.22, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontsize=14)

    axes[1].plot(
        theta_elec_deg,
        Btheta,
        color="#E68613",
        linewidth=2.2,
        label=f"{model}: Btheta max={np.max(np.abs(Btheta)):.4f} T",
    )
    axes[1].set_xlabel("Position (Elec.Deg.)", fontsize=12)
    axes[1].set_ylabel("Tangential flux density (T)", fontsize=12)
    axes[1].set_xlim(-90, 90)
    axes[1].set_ylim(min(-0.2, float(np.min(Btheta)) * 1.1), max(0.2, float(np.max(Btheta)) * 1.1))
    axes[1].set_xticks([-90, -45, 0, 45, 90])
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=11, loc="upper right")
    axes[1].text(0.5, -0.22, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontsize=14)

    plot_path = output_dir / f"fig9_no_load_flux_density_{model}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    # Keep the old filename as the latest-run convenience output. Use the
    # model-specific filename above for comparisons to avoid viewer/cache mixups.
    latest_plot_path = output_dir / "fig9_no_load_flux_density.png"
    fig.savefig(latest_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path, data_path


def main() -> None:
    params = AirgapFluxParams()
    theta_elec_deg, Br, Btheta, residuals = total_no_load_flux_density(params)
    plot_path, data_path = save_flux_outputs(theta_elec_deg, Br, Btheta, residuals, params)

    print("Step 4 air-gap flux density")
    print(f"segments superposed       : {params.boundary.geometry.Nz}")
    print(f"magnetization model       : {params.boundary.magnetization_model}")
    print(f"air-gap radius Rg [m]     : {params.airgap_radius_m:.9f}")
    print(f"theta range [Elec.Deg.]   : {theta_elec_deg[0]:.1f} to {theta_elec_deg[-1]:.1f}")
    print(f"Br min/max [T]            : {np.min(Br): .6e}, {np.max(Br): .6e}")
    print(f"Btheta min/max [T]        : {np.min(Btheta): .6e}, {np.max(Btheta): .6e}")
    print(f"max solve residual        : {max(residuals):.6e}")
    print()
    print(f"plot: {plot_path.resolve()}")
    print(f"data: {data_path.resolve()}")


if __name__ == "__main__":
    main()

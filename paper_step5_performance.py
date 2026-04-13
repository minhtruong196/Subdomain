from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from paper_step3_boundary_matrix import (
    BoundaryMatrixParams,
    UnknownLayout,
    build_boundary_matrix,
    selected_segment_radii_m,
    xprime_mj_at_ru,
)
from paper_step4_airgap_flux_density import segment_flux_density

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PerformanceParams:
    """Step 5 parameters for cogging torque and no-load back-EMF."""

    boundary: BoundaryMatrixParams = BoundaryMatrixParams()
    active_axial_length_m: float = 88.0e-3
    airgap_radius_m: float = 88.4e-3
    rated_speed_rpm: float = 900.0
    series_turns_per_phase: int = 144
    coil_pitch_slots: int = 5

    torque_position_count: int = 121
    torque_theta_count: int = 1440
    emf_sample_count: int = 361
    output_dir: str = "outputs/step5_performance"

    @property
    def omega_mech_rad_s(self) -> float:
        return 2.0 * np.pi * self.rated_speed_rpm / 60.0

    @property
    def pole_pairs(self) -> int:
        return self.boundary.geometry.pole_pairs


@dataclass(frozen=True)
class SegmentSolutionSet:
    layout: UnknownLayout
    meta: dict[str, np.ndarray | float]
    solutions: np.ndarray


def boundary_params_for_segment(params: BoundaryMatrixParams, segment_j: int, delta_rad: float = 0.0) -> BoundaryMatrixParams:
    return BoundaryMatrixParams(
        geometry=params.geometry,
        segment_j=segment_j,
        max_pole_harmonic=params.max_pole_harmonic,
        slot_harmonics=params.slot_harmonics,
        magnetization_model=params.magnetization_model,
        Br_T=params.Br_T,
        mu0=params.mu0,
        mu_r=params.mu_r,
        stator_inner_radius_m=params.stator_inner_radius_m,
        slot_depth_m=params.slot_depth_m,
        slot_width_m=params.slot_width_m,
        Jui_A_per_m2=params.Jui_A_per_m2,
        Jdi_A_per_m2=params.Jdi_A_per_m2,
        delta_rad=delta_rad,
        output_dir=params.output_dir,
    )


def rhs_for_delta(
    base_y: np.ndarray,
    layout: UnknownLayout,
    meta: dict[str, np.ndarray | float],
    boundary_params: BoundaryMatrixParams,
    delta_rad: float,
) -> np.ndarray:
    """Update the Eq. Appendix-2 source terms for a rotor angle Delta."""

    y = base_y.copy()
    mc_values = np.asarray(meta["mc_values"], dtype=float)
    Mr_values = np.asarray(meta["Mr_values"], dtype=float)
    Mt_values = np.asarray(meta["Mt_values"], dtype=float)
    Ru = float(meta["R_u_m"])
    Rl = float(meta["R_l_m"])

    first_appendix_2_row = 2 * layout.m_count
    for mi, (mc, Mr_m, Mt_m) in enumerate(zip(mc_values, Mr_values, Mt_values)):
        # Keep this in sync with Appendix 2 in paper_step3_boundary_matrix.py:
        # tangential H continuity adds the missing mu0*Mtheta source term.
        source = xprime_mj_at_ru(mc, Mr_m, Mt_m, Rl, Ru, boundary_params.mu0) + boundary_params.mu0 * Mt_m
        y[first_appendix_2_row + 2 * mi] = -(source / boundary_params.mu_r) * np.sin(mc * delta_rad)
        y[first_appendix_2_row + 2 * mi + 1] = (source / boundary_params.mu_r) * np.cos(mc * delta_rad)
    return y


def solve_segments_for_deltas(params: BoundaryMatrixParams, delta_rad: np.ndarray) -> list[SegmentSolutionSet]:
    """Solve K X = Y for all PM segment pairs and all requested rotor angles."""

    solved: list[SegmentSolutionSet] = []
    for segment_j in range(1, params.geometry.Nz + 1):
        segment_params = boundary_params_for_segment(params, segment_j, delta_rad=0.0)
        K, Y0, layout, meta = build_boundary_matrix(segment_params)
        rhs = np.column_stack([rhs_for_delta(Y0, layout, meta, segment_params, float(delta)) for delta in delta_rad])
        solutions = np.linalg.solve(K, rhs)
        solved.append(SegmentSolutionSet(layout=layout, meta=meta, solutions=solutions))
    return solved


def full_airgap_flux_density(
    solved_segments: list[SegmentSolutionSet],
    delta_index: int,
    theta_mech_rad: np.ndarray,
    params: PerformanceParams,
) -> tuple[np.ndarray, np.ndarray]:
    Br_total = np.zeros_like(theta_mech_rad)
    Btheta_total = np.zeros_like(theta_mech_rad)

    for segment in solved_segments:
        solution = segment.solutions[:, delta_index]
        mc_values = np.asarray(segment.meta["mc_values"], dtype=float)
        Ru = float(segment.meta["R_u_m"])
        Rs = float(segment.meta["R_s_m"])
        Br_j, Btheta_j = segment_flux_density(
            params.airgap_radius_m,
            theta_mech_rad,
            solution,
            segment.layout,
            mc_values,
            Ru,
            Rs,
        )
        Br_total += Br_j
        Btheta_total += Btheta_j

    return Br_total, Btheta_total


def cogging_torque_waveform(params: PerformanceParams) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (30): T = Lef Rg^2 / mu0 * integral Br(Rg,theta) Btheta(Rg,theta) dtheta."""

    rotor_pos_deg = np.linspace(0.0, 30.0, params.torque_position_count)
    delta_rad = np.deg2rad(rotor_pos_deg)
    theta = np.linspace(0.0, 2.0 * np.pi, params.torque_theta_count, endpoint=False)

    solved_segments = solve_segments_for_deltas(params.boundary, delta_rad)
    torque_nm = np.empty_like(delta_rad)

    for idx in range(len(delta_rad)):
        Br, Btheta = full_airgap_flux_density(solved_segments, idx, theta, params)
        integral = 2.0 * np.pi * np.mean(Br * Btheta)
        torque_nm[idx] = params.active_axial_length_m * params.airgap_radius_m**2 / params.boundary.mu0 * integral

    torque_nm -= np.mean(torque_nm)
    return rotor_pos_deg, 1.0e3 * torque_nm


def odd_periodic_slot_map(slot_number: int, boundary_params: BoundaryMatrixParams) -> tuple[int, float]:
    """Map a global slot to the reduced Z/(2c) slot index and odd-periodic sign."""

    reduced = boundary_params.reduced_slot_count
    zero_based = (slot_number - 1) % boundary_params.geometry.slots
    sector = zero_based // reduced
    local_slot_idx = zero_based % reduced
    sign = -1.0 if sector % 2 else 1.0
    return local_slot_idx, sign


def phase_a_coils(boundary_params: BoundaryMatrixParams) -> list[tuple[int, int, float]]:
    """Return (start_slot, return_slot, sign) for a q=2, y=5 phase-A winding.

    Only the A+ upper-layer starts are listed. Including the A- starts as new
    coils double-counts the same physical short-pitch coil set and halves the
    assumed turns per coil when N1 is interpreted as series turns per phase.
    """

    coils: list[tuple[int, int, float]] = []
    Z = boundary_params.geometry.slots
    p = boundary_params.geometry.pole_pairs
    y = 5

    for slot in range(1, Z + 1):
        theta_center = (2 * slot - 1) * np.pi / Z
        elec_deg = np.rad2deg(p * theta_center) % 360.0

        if 0.0 <= elec_deg < 60.0:
            return_slot = ((slot + y - 1) % Z) + 1
            coils.append((slot, return_slot, 1.0))

    return coils


def slot_constant_potential(
    solution: np.ndarray,
    layout: UnknownLayout,
    slot_number: int,
    upper_half: bool,
    boundary_params: BoundaryMatrixParams,
) -> float:
    local_slot_idx, periodic_sign = odd_periodic_slot_map(slot_number, boundary_params)
    if upper_half:
        return periodic_sign * float(solution[layout.B03(local_slot_idx)])
    return periodic_sign * float(solution[layout.B04(local_slot_idx)])


def phase_a_flux_linkage(
    solved_segments: list[SegmentSolutionSet],
    delta_index: int,
    params: PerformanceParams,
) -> float:
    """Eq. (31) checkpoint for phase-A flux linkage.

    No-load slot current terms are zero. The integral of the cosine slot
    harmonics over a full slot opening is zero because Gn * bs = n*pi, so the
    slot-average term is governed by B03i and B04i. We use B03i/B04i as the
    area-normalized slot averages from Eq. (31); explicitly multiplying by a
    separately approximated slot area scales the phase EMF above the Maxwell
    radial checkpoint.
    """

    coils = phase_a_coils(params.boundary)
    turns_per_coil = params.series_turns_per_phase / len(coils)
    flux = 0.0

    for segment in solved_segments:
        solution = segment.solutions[:, delta_index]
        for start_slot, return_slot, coil_sign in coils:
            a_start = slot_constant_potential(solution, segment.layout, start_slot, True, params.boundary)
            a_return = slot_constant_potential(solution, segment.layout, return_slot, False, params.boundary)
            flux += coil_sign * params.active_axial_length_m * turns_per_coil * (a_start - a_return)

    return float(flux)


def no_load_back_emf_waveform(params: PerformanceParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elec_deg = np.linspace(0.0, 360.0, params.emf_sample_count, endpoint=False)
    delta_rad = np.deg2rad(elec_deg) / params.pole_pairs
    solved_segments = solve_segments_for_deltas(params.boundary, delta_rad)

    flux_linkage = np.array(
        [phase_a_flux_linkage(solved_segments, idx, params) for idx in range(len(delta_rad))],
        dtype=float,
    )

    step = delta_rad[1] - delta_rad[0]
    dflux_ddelta = (np.roll(flux_linkage, -1) - np.roll(flux_linkage, 1)) / (2.0 * step)
    emf_v = -params.omega_mech_rad_s * dflux_ddelta
    emf_v -= np.mean(emf_v)
    return elec_deg, emf_v, flux_linkage


def periodic_shift_deg(x_deg: np.ndarray, signal: np.ndarray, shift_deg: float) -> np.ndarray:
    """Evaluate a 360-electrical-degree periodic waveform at x_deg - shift_deg."""

    period = 360.0
    x = np.asarray(x_deg, dtype=float)
    y = np.asarray(signal, dtype=float)
    xp = np.concatenate([x, [x[0] + period]])
    fp = np.concatenate([y, [y[0]]])
    return np.interp((x - shift_deg) % period, xp, fp)


def line_to_line_back_emf_from_phase_a(elec_deg: np.ndarray, phase_a_emf_v: np.ndarray) -> np.ndarray:
    """Balanced-Y conversion: e_ab(theta) = e_a(theta) - e_b(theta)."""

    phase_b_emf_v = periodic_shift_deg(elec_deg, phase_a_emf_v, 120.0)
    line_line_emf_v = phase_a_emf_v - phase_b_emf_v
    line_line_emf_v -= np.mean(line_line_emf_v)
    return line_line_emf_v


def save_performance_plots(
    torque_deg: np.ndarray,
    torque_mNm: np.ndarray,
    emf_deg: np.ndarray,
    phase_emf_v: np.ndarray,
    line_line_emf_v: np.ndarray,
    flux_linkage: np.ndarray,
    params: PerformanceParams,
) -> tuple[Path, Path, Path]:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "step5_performance_data.npz"
    np.savez_compressed(
        data_path,
        torque_rotor_pos_mech_deg=torque_deg,
        cogging_torque_mNm=torque_mNm,
        emf_position_elec_deg=emf_deg,
        no_load_phase_back_emf_v=phase_emf_v,
        no_load_line_line_back_emf_v=line_line_emf_v,
        phase_a_flux_linkage_wb=flux_linkage,
        magnetization_model=params.boundary.magnetization_model,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(torque_deg, torque_mNm, color="#0072BD", linewidth=2.2, label="Proposed new technique")
    ax.set_xlabel("Rotor position (Mech.Deg.)")
    ax.set_ylabel("Cogging torque (mN.m)")
    ax.set_xlim(0, 30)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    torque_path = output_dir / "fig11_cogging_torque.png"
    fig.savefig(torque_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(emf_deg, line_line_emf_v, color="#D95319", linewidth=2.2, label="Line-line estimate")
    ax.plot(emf_deg, phase_emf_v, color="#7E7E7E", linewidth=1.5, linestyle="--", label="Phase-A checkpoint")
    ax.set_xlabel("Position (Elec.Deg.)")
    ax.set_ylabel("No-load back-EMF (V)")
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    emf_path = output_dir / "fig20a_no_load_back_emf.png"
    fig.savefig(emf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return torque_path, emf_path, data_path


def fundamental_peak(signal: np.ndarray) -> float:
    coeff = np.fft.rfft(signal) / len(signal)
    return float(2.0 * abs(coeff[1]))


def main() -> None:
    params = PerformanceParams()
    torque_deg, torque_mNm = cogging_torque_waveform(params)
    emf_deg, phase_emf_v, flux_linkage = no_load_back_emf_waveform(params)
    line_line_emf_v = line_to_line_back_emf_from_phase_a(emf_deg, phase_emf_v)
    torque_path, emf_path, data_path = save_performance_plots(
        torque_deg, torque_mNm, emf_deg, phase_emf_v, line_line_emf_v, flux_linkage, params
    )

    print("Step 5 electromagnetic performance")
    print(f"magnetization model       : {params.boundary.magnetization_model}")
    print(f"speed [r/min]             : {params.rated_speed_rpm:.3f}")
    print(f"phase-A coils             : {len(phase_a_coils(params.boundary))}")
    print(f"turns per coil assumed    : {params.series_turns_per_phase / len(phase_a_coils(params.boundary)):.6f}")
    print(f"cogging min/max [mN.m]    : {np.min(torque_mNm): .6e}, {np.max(torque_mNm): .6e}")
    print(f"cogging amplitude [mN.m]  : {0.5 * np.ptp(torque_mNm): .6e}")
    print(f"phase back-EMF min/max [V]: {np.min(phase_emf_v): .6e}, {np.max(phase_emf_v): .6e}")
    print(f"phase back-EMF fund. [V]  : {fundamental_peak(phase_emf_v): .6e}")
    print(f"line back-EMF min/max [V] : {np.min(line_line_emf_v): .6e}, {np.max(line_line_emf_v): .6e}")
    print(f"line back-EMF fund. [V]   : {fundamental_peak(line_line_emf_v): .6e}")
    print()
    print(f"torque plot: {torque_path.resolve()}")
    print(f"emf plot   : {emf_path.resolve()}")
    print(f"data       : {data_path.resolve()}")


if __name__ == "__main__":
    main()

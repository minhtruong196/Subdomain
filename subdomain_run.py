from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from subdomain_airgap import total_no_load_flux_density
from subdomain_boundary import solve_boundary_matrix
from subdomain_config import MachineConfig
from subdomain_geometry import segment_radii_mm, validate_rotor_geometry
from subdomain_magnetization import (
    exact_mr_pair,
    magnetization_coefficients,
    reconstruct_mr,
    theta_grid_half_pole,
)
from subdomain_performance import (
    cogging_torque_waveform,
    fundamental_peak,
    line_to_line_back_emf_from_phase_a,
    no_load_back_emf_waveform,
    phase_a_coils,
)
from subdomain_plots import (
    save_boundary_solution,
    save_flux_outputs,
    save_geometry_plot,
    save_geometry_table,
    save_magnetization_plot,
    save_performance_plots,
)


def parse_args(default_task: str = "all") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the refactored subdomain model.")
    parser.add_argument(
        "task",
        nargs="?",
        default=default_task,
        choices=["geometry", "magnetization", "boundary", "airgap", "performance", "all"],
        help=f"Calculation to run. Default: {default_task}.",
    )
    parser.add_argument("--segment", type=int, default=1, help="PM segment index for boundary/magnetization tasks.")
    parser.add_argument("--slots", type=int, default=None, help="Override slot count.")
    parser.add_argument("--poles", type=int, default=None, help="Override pole count.")
    parser.add_argument("--nz", type=int, default=None, help="Override PM segment count.")
    parser.add_argument("--alpha", type=float, default=None, help="Override pole-arc coefficient.")
    parser.add_argument("--hp-mm", type=float, default=None, help="Override PM inner radius hp.")
    parser.add_argument("--h-mm", type=float, default=None, help="Override shaping offset h.")
    parser.add_argument(
        "--edge-radius-mode",
        choices=["midpoint", "profile", "side_length"],
        default=None,
        help="Rule for the last PM segment radius: midpoint, outer profile, or legacy PM side length.",
    )
    parser.add_argument("--edge-pm-side-length-mm", type=float, default=None, help="PM edge side length for side_length mode.")
    parser.add_argument("--g-mm", type=float, default=None, help="Set air-gap length; derives Ror = Ris - g.")
    parser.add_argument("--dm-mm", type=float, default=None, help="Set magnet depth; derives hp = Ror - dm.")
    parser.add_argument("--stator-inner-radius-mm", type=float, default=None, help="Override stator inner radius Ris.")
    parser.add_argument("--airgap-radius-mm", type=float, default=None, help="Override air-gap evaluation radius Rg.")
    parser.add_argument("--br", type=float, default=None, help="Override PM remanence Br in tesla.")
    parser.add_argument("--max-pole-harmonic", type=int, default=None, help="Override maximum pole harmonic nu.")
    parser.add_argument("--slot-harmonics", type=int, default=None, help="Override slot harmonic count.")
    parser.add_argument("--magnetization-model", choices=["parallel", "radial"], default=None)
    parser.add_argument("--quick", action="store_true", help="Use smaller sample counts for fast smoke runs.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MachineConfig:
    config = MachineConfig()

    stator_changes: dict[str, float] = {}
    if args.stator_inner_radius_mm is not None:
        stator_changes["stator_inner_radius_m"] = args.stator_inner_radius_mm * 1.0e-3
    if args.g_mm is not None:
        stator_changes["airgap_length_m"] = args.g_mm * 1.0e-3
    if args.airgap_radius_mm is not None:
        stator_changes["airgap_radius_m"] = args.airgap_radius_mm * 1.0e-3
    if stator_changes:
        config = replace(config, stator=replace(config.stator, **stator_changes))

    geometry_changes: dict[str, float | int] = {}
    if args.slots is not None:
        geometry_changes["slots"] = args.slots
    if args.poles is not None:
        geometry_changes["poles"] = args.poles
    if args.nz is not None:
        geometry_changes["Nz"] = args.nz
    if args.alpha is not None:
        geometry_changes["alpha_p"] = args.alpha
    if args.h_mm is not None:
        geometry_changes["h_mm"] = args.h_mm
    if args.edge_radius_mode is not None:
        geometry_changes["edge_radius_mode"] = args.edge_radius_mode
    if args.edge_pm_side_length_mm is not None:
        geometry_changes["edge_pm_side_length_mm"] = args.edge_pm_side_length_mm

    if args.dm_mm is not None:
        geometry_changes["hp_mm"] = config.rotor_outer_radius_mm - args.dm_mm
    if args.hp_mm is not None:
        geometry_changes["hp_mm"] = args.hp_mm
    if geometry_changes:
        config = replace(config, geometry=replace(config.geometry, **geometry_changes))

    magnet_changes: dict[str, float | str] = {}
    if args.br is not None:
        magnet_changes["Br_T"] = args.br
    if args.magnetization_model is not None:
        magnet_changes["magnetization_model"] = args.magnetization_model
    if magnet_changes:
        config = replace(config, magnet=replace(config.magnet, **magnet_changes))

    solver_changes: dict[str, int] = {}
    if args.max_pole_harmonic is not None:
        solver_changes["max_pole_harmonic"] = args.max_pole_harmonic
    if args.slot_harmonics is not None:
        solver_changes["slot_harmonics"] = args.slot_harmonics
    if args.quick:
        solver_changes.setdefault("max_pole_harmonic", 20)
        solver_changes.setdefault("slot_harmonics", 4)
        solver_changes["magnetization_sample_count"] = 800
        solver_changes["airgap_sample_count"] = 181
        config = replace(
            config,
            operating=replace(
                config.operating,
                torque_position_count=25,
                torque_theta_count=360,
                emf_sample_count=91,
            ),
        )
    if solver_changes:
        config = replace(config, solver=replace(config.solver, **solver_changes))

    config.validate_dimensions()
    validate_rotor_geometry(config)
    return config


def run_geometry(config: MachineConfig) -> None:
    orders, Ru, Rl = segment_radii_mm(config)
    plot_path = save_geometry_plot(config)
    table_path = save_geometry_table(config)

    print("Subdomain geometry")
    print(f"slots, poles, pole pairs  : {config.geometry.slots}, {config.geometry.poles}, {config.geometry.pole_pairs}")
    print(f"segments Nz               : {config.geometry.Nz}")
    print(f"Ror, hp, dm [mm]          : {config.rotor_outer_radius_mm:.6f}, {config.geometry.hp_mm:.6f}, {config.magnet_depth_mm:.6f}")
    print(f"Rp, h [mm]                : {config.Rp_mm:.6f}, {config.geometry.h_mm:.6f}")
    print(f"g, Rg [mm]                : {config.airgap_length_m * 1.0e3:.6f}, {config.airgap_radius_m * 1.0e3:.6f}")
    print(f"edge radius mode          : {config.geometry.edge_radius_mode}")
    print(f"Ru min/max [mm]           : {np.min(Ru):.6f}, {np.max(Ru):.6f}")
    print(f"Rl min/max [mm]           : {np.min(Rl):.6f}, {np.max(Rl):.6f}")
    print(f"first/last segment        : {orders[0]}, {orders[-1]}")
    print(f"plot                      : {plot_path.resolve()}")
    print(f"table                     : {table_path.resolve()}")


def run_magnetization(config: MachineConfig, segment_j: int) -> None:
    mc, Mr_m, Mt_m = magnetization_coefficients(segment_j, config)
    theta = theta_grid_half_pole(config)
    Mr_recon = reconstruct_mr(theta, mc, Mr_m)
    Mr_exact = exact_mr_pair(theta, segment_j, config)
    plot_path = save_magnetization_plot(theta, Mr_exact, Mr_recon, segment_j, config)

    print("Subdomain magnetization")
    print(f"segment j                 : {segment_j}")
    print(f"magnetization model       : {config.magnet.magnetization_model}")
    print(f"max pole harmonic nu      : {config.solver.max_pole_harmonic}")
    print(f"kept harmonic count       : {len(mc)}")
    print(f"first mc values           : {mc[:8].astype(int).tolist()}")
    print(f"M0 = Br/mu0 [A/m]         : {config.magnet.M0_A_per_m:.6e}")
    print(f"first Mr/M0, Mt/M0        : {Mr_m[0] / config.magnet.M0_A_per_m:.8f}, {Mt_m[0] / config.magnet.M0_A_per_m:.8f}")
    print(f"plot                      : {plot_path.resolve()}")


def run_boundary(config: MachineConfig, segment_j: int) -> None:
    solution, K, Y, layout, meta, residual = solve_boundary_matrix(config, segment_j=segment_j)
    output_path = save_boundary_solution(solution, K, Y, layout, meta, residual, config, segment_j)

    print("Subdomain boundary matrix")
    print(f"segment j                 : {segment_j}")
    print(f"reduced slot count        : {layout.slot_count}")
    print(f"pole harmonics kept       : {layout.m_count} (nu <= {config.solver.max_pole_harmonic}, odd only)")
    print(f"slot harmonics n          : {layout.n_count}")
    print(f"matrix shape              : {K.shape}")
    print(f"rhs norm                  : {np.linalg.norm(Y):.6e}")
    print(f"relative residual         : {residual:.6e}")
    print(f"Ru, Rl [m]                : {meta['R_u_m']:.9f}, {meta['R_l_m']:.9f}")
    print(f"Rs, Rsc, Rsa [m]          : {meta['R_s_m']:.9f}, {meta['R_sc_m']:.9f}, {meta['R_sa_m']:.9f}")
    print(f"airgap radius Rg [m]      : {config.airgap_radius_m:.9f}")
    print(f"saved                     : {output_path.resolve()}")


def run_airgap(config: MachineConfig) -> None:
    theta_elec_deg, Br, Btheta, residuals = total_no_load_flux_density(config)
    plot_path, data_path = save_flux_outputs(theta_elec_deg, Br, Btheta, residuals, config)

    print("Subdomain air-gap flux density")
    print(f"segments superposed       : {config.geometry.Nz}")
    print(f"magnetization model       : {config.magnet.magnetization_model}")
    print(f"air-gap length g [m]      : {config.airgap_length_m:.9f}")
    print(f"air-gap radius Rg [m]     : {config.airgap_radius_m:.9f}")
    print(f"theta range [Elec.Deg.]   : {theta_elec_deg[0]:.1f} to {theta_elec_deg[-1]:.1f}")
    print(f"Br min/max [T]            : {np.min(Br): .6e}, {np.max(Br): .6e}")
    print(f"Btheta min/max [T]        : {np.min(Btheta): .6e}, {np.max(Btheta): .6e}")
    print(f"max solve residual        : {max(residuals):.6e}")
    print(f"plot                      : {plot_path.resolve()}")
    print(f"data                      : {data_path.resolve()}")


def run_performance(config: MachineConfig) -> None:
    torque_deg, torque_mNm = cogging_torque_waveform(config)
    emf_deg, phase_emf_v, flux_linkage = no_load_back_emf_waveform(config)
    line_line_emf_v = line_to_line_back_emf_from_phase_a(emf_deg, phase_emf_v)
    torque_path, emf_path, data_path = save_performance_plots(
        torque_deg, torque_mNm, emf_deg, phase_emf_v, line_line_emf_v, flux_linkage, config
    )

    print("Subdomain electromagnetic performance")
    print(f"magnetization model       : {config.magnet.magnetization_model}")
    print(f"air-gap length g [m]      : {config.airgap_length_m:.9f}")
    print(f"air-gap radius Rg [m]     : {config.airgap_radius_m:.9f}")
    print(f"speed [r/min]             : {config.operating.rated_speed_rpm:.3f}")
    print(f"phase-A coils             : {len(phase_a_coils(config))}")
    print(f"cogging min/max [mN.m]    : {np.min(torque_mNm): .6e}, {np.max(torque_mNm): .6e}")
    print(f"cogging amplitude [mN.m]  : {0.5 * np.ptp(torque_mNm): .6e}")
    print(f"phase back-EMF fund. [V]  : {fundamental_peak(phase_emf_v): .6e}")
    print(f"line back-EMF fund. [V]   : {fundamental_peak(line_line_emf_v): .6e}")
    print(f"torque plot               : {torque_path.resolve()}")
    print(f"emf plot                  : {emf_path.resolve()}")
    print(f"data                      : {data_path.resolve()}")


def main(default_task: str = "all") -> None:
    args = parse_args(default_task=default_task)
    config = build_config(args)

    if args.task in {"geometry", "all"}:
        run_geometry(config)
    if args.task in {"magnetization", "all"}:
        run_magnetization(config, args.segment)
    if args.task in {"boundary", "all"}:
        run_boundary(config, args.segment)
    if args.task in {"airgap", "all"}:
        run_airgap(config)
    if args.task in {"performance", "all"}:
        run_performance(config)


if __name__ == "__main__":
    main()

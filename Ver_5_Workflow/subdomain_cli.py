from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from subdomain_config import MachineConfig
from subdomain_geometry import validate_rotor_geometry


def add_runtime_arguments(parser: argparse.ArgumentParser, *, allow_current: bool) -> None:
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=None,
        help="Override the initial mechanical rotor position [deg].",
    )
    if allow_current:
        parser.add_argument(
            "--i-peak-a",
            type=float,
            default=None,
            help="Override rotating three-phase current peak [A].",
        )
    parser.add_argument("--quick", action="store_true", help="Use reduced discretization for a smoke run.")


def config_from_runtime_args(args: argparse.Namespace, *, allow_current: bool) -> MachineConfig:
    config = MachineConfig()

    if args.delta_deg is not None:
        config = replace(
            config,
            operating=replace(config.operating, delta_rad=float(np.deg2rad(args.delta_deg))),
        )
    if allow_current and args.i_peak_a is not None:
        config = replace(config, current=replace(config.current, I_peak_A=args.i_peak_a))
    if args.quick:
        config = replace(
            config,
            solver=replace(
                config.solver,
                max_pole_harmonic=20,
                slot_harmonics=4,
                magnetization_sample_count=800,
                airgap_sample_count=181,
            ),
            operating=replace(
                config.operating,
                torque_position_count=25,
                torque_theta_count=360,
                emf_sample_count=91,
            ),
        )

    config.validate_dimensions()
    validate_rotor_geometry(config)
    return config

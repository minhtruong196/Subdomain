from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from subdomain_cli import add_runtime_arguments, config_from_runtime_args
from subdomain_config import CurrentConfig
from subdomain_performance import cogging_torque_waveform
from subdomain_plots import save_cogging_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate no-load cogging torque.")
    add_runtime_arguments(parser, allow_current=False)
    config = config_from_runtime_args(parser.parse_args(), allow_current=False)
    config = replace(config, current=CurrentConfig())

    rotor_pos_deg, torque_mnm = cogging_torque_waveform(config)
    plot_path, data_path = save_cogging_outputs(rotor_pos_deg, torque_mnm, config)

    print("Cogging torque (no-load)")
    print(f"delta0 [Mech.Deg.]         : {np.rad2deg(config.operating.delta_rad):.6f}")
    print(f"cogging min/max [mN.m]     : {np.min(torque_mnm):.6f}, {np.max(torque_mnm):.6f}")
    print(f"cogging amplitude [mN.m]   : {0.5 * np.ptp(torque_mnm):.6f}")
    print(f"plot/data                  : {plot_path.resolve()}, {data_path.resolve()}")


if __name__ == "__main__":
    main()

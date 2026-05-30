from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from subdomain_boundary import phase_currents_for_rotor_position, slot_current_densities
from subdomain_cli import add_runtime_arguments, config_from_runtime_args
from subdomain_config import CurrentConfig
from subdomain_performance import (
    electromagnetic_torque_waveform,
    line_to_line_back_emf_from_phase_a,
    no_load_back_emf_waveform,
)
from subdomain_plots import save_back_emf_outputs, save_loaded_torque_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate loaded torque and no-load back-EMF.")
    add_runtime_arguments(parser, allow_current=True)
    config = config_from_runtime_args(parser.parse_args(), allow_current=True)
    if config.current.I_peak_A is None:
        parser.error("set current.I_peak_A in subdomain_config.py or provide --i-peak-a")

    torque_deg, torque_nm = electromagnetic_torque_waveform(config)
    currents = phase_currents_for_rotor_position(config, config.operating.delta_rad)
    Jui, Jdi = slot_current_densities(config, delta_rad=config.operating.delta_rad)

    no_load_config = replace(config, current=CurrentConfig())
    emf_deg, phase_emf_v, flux_linkage = no_load_back_emf_waveform(no_load_config)
    line_emf_v = line_to_line_back_emf_from_phase_a(emf_deg, phase_emf_v)
    torque_plot, torque_data = save_loaded_torque_outputs(torque_deg, torque_nm, config)
    emf_plot, emf_data = save_back_emf_outputs(
        emf_deg, phase_emf_v, line_emf_v, flux_linkage, no_load_config
    )

    print("Electromagnetic performance")
    print(f"delta0 [Mech.Deg.]         : {np.rad2deg(config.operating.delta_rad):.6f}")
    print(f"I_peak / I_rms [A]         : {config.current.I_peak_A:.6f}, {config.current.I_peak_A / np.sqrt(2.0):.6f}")
    print(f"ia, ib, ic at delta0 [A]  : {currents[0]:.6f}, {currents[1]:.6f}, {currents[2]:.6f}")
    print(f"max |Jui| / |Jdi| [A/m2]  : {np.max(np.abs(Jui)):.6e}, {np.max(np.abs(Jdi)):.6e}")
    print(f"torque avg [N.m]           : {np.mean(torque_nm):.6f}")
    print(f"torque min/max [N.m]       : {np.min(torque_nm):.6f}, {np.max(torque_nm):.6f}")
    print(f"torque ripple p-p [N.m]    : {np.ptp(torque_nm):.6f}")
    print(f"phase EMF peak/rms [V]     : {np.max(np.abs(phase_emf_v)):.6f}, {np.sqrt(np.mean(phase_emf_v**2)):.6f}")
    print(f"line EMF peak/rms [V]      : {np.max(np.abs(line_emf_v)):.6f}, {np.sqrt(np.mean(line_emf_v**2)):.6f}")
    print(f"torque plot/data           : {torque_plot.resolve()}, {torque_data.resolve()}")
    print(f"EMF plot/data              : {emf_plot.resolve()}, {emf_data.resolve()}")


if __name__ == "__main__":
    main()

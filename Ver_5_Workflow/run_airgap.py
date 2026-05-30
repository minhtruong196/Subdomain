from __future__ import annotations

import argparse

import numpy as np

from subdomain_airgap import total_airgap_flux_density
from subdomain_boundary import phase_currents_for_rotor_position
from subdomain_cli import add_runtime_arguments, config_from_runtime_args
from subdomain_geometry import pm_edge_dimensions_mm
from subdomain_plots import save_flux_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the air-gap flux-density snapshot.")
    add_runtime_arguments(parser, allow_current=True)
    config = config_from_runtime_args(parser.parse_args(), allow_current=True)

    theta_elec_deg, Br, Btheta, residuals = total_airgap_flux_density(config)
    plot_path, data_path = save_flux_outputs(theta_elec_deg, Br, Btheta, residuals, config)
    side_length_mm, upper_edge_x_mm = pm_edge_dimensions_mm(config)
    delta_deg = float(np.rad2deg(config.operating.delta_rad))

    print("Air-gap flux density")
    print(f"delta [Mech.Deg.]          : {delta_deg:.6f}")
    if config.current.I_peak_A is None:
        print("excitation                 : no-load")
    else:
        currents = phase_currents_for_rotor_position(config, config.operating.delta_rad)
        print(f"I_peak [A]                 : {config.current.I_peak_A:.6f}")
        print(f"ia, ib, ic [A]             : {currents[0]:.6f}, {currents[1]:.6f}, {currents[2]:.6f}")
    print(f"PM side length [mm]        : {side_length_mm:.6f}")
    print(f"upper edge x [mm]          : {upper_edge_x_mm:.6f}")
    print(f"g, Rg [mm]                 : {config.airgap_length_m * 1.0e3:.6f}, {config.airgap_radius_m * 1.0e3:.6f}")
    print(f"Br min/max [T]             : {np.min(Br):.6e}, {np.max(Br):.6e}")
    print(f"Btheta min/max [T]         : {np.min(Btheta):.6e}, {np.max(Btheta):.6e}")
    print(f"max solve residual         : {max(residuals):.6e}")
    print(f"plot                       : {plot_path.resolve()}")
    print(f"data                       : {data_path.resolve()}")


if __name__ == "__main__":
    main()

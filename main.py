from __future__ import annotations

import numpy as np

from config import LinearPMModelParams
from plotter import (
    save_5region_pm_only_cogging_png,
    save_5region_pm_only_fields_png,
    output_directory,
    save_air_gap_fields_png,
    save_current_loading_png,
    save_harmonics_png,
    save_motor_schematic_png,
    save_torque_angle_png,
)
from post_processing import (
    convergence_rows,
    dominant_current_loading_rows,
    dominant_harmonic_rows,
)
from solver import (
    prepare_linear_subdomain_scaffold,
    solve_linear_5region_pm_only,
    solve_smooth_air_gap_baseline,
)


def main() -> None:
    params = LinearPMModelParams()
    solution = solve_smooth_air_gap_baseline(params)
    scaffold = prepare_linear_subdomain_scaffold(params)
    pm_only_5region = solve_linear_5region_pm_only(params)
    output_dir = output_directory(params.output_dir)

    current_loading_png = save_current_loading_png(output_dir, solution)
    air_gap_png = save_air_gap_fields_png(output_dir, solution)
    harmonics_png = save_harmonics_png(output_dir, solution)
    torque_angle_png = save_torque_angle_png(output_dir, solution)
    motor_schematic_png = save_motor_schematic_png(output_dir, solution)
    pm_only_5region_fields_png = save_5region_pm_only_fields_png(output_dir, pm_only_5region)
    pm_only_5region_cogging_png = save_5region_pm_only_cogging_png(output_dir, pm_only_5region)

    ia, ib, ic = solution.phase_currents

    print("Linear PM baseline")
    print(f"pole pairs               : {params.pole_pairs}")
    print(f"stator slots             : {params.stator_slots}")
    print(f"magnet arc ratio         : {params.magnet_arc_ratio:.3f}")
    print(f"M0 [A/m]                 : {params.M0:.3f}")
    print(f"air-gap mid radius [m]   : {params.mid_gap_radius:.6f}")
    print(f"odd harmonics kept       : {len(solution.pm_spectrum.electrical_orders)}")
    print()

    print("Dominant PM harmonics")
    for nu, k, amp in dominant_harmonic_rows(solution.pm_spectrum):
        print(f"nu={nu:2d}  mech_order={k:3d}  Mr_amp={amp:12.3f} A/m")
    print()

    print("Magnetization convergence")
    for limit, err in convergence_rows(solution.theta_mech, params, solution.pm_spectrum):
        print(f"up to nu={limit:2d}  rms_error={err:12.3f} A/m")
    print()

    print("Waveform checks")
    print(f"exact Mr peak [A/m]      : {np.max(np.abs(solution.exact_Mr)):12.3f}")
    print(f"recon Mr peak [A/m]      : {np.max(np.abs(solution.recon_Mr)):12.3f}")
    print()

    print("Slot-current load model")
    print(f"torque angle [deg el.]   : {np.rad2deg(params.torque_angle_elec):12.3f}")
    print(f"phase currents [A]       : ia={ia:8.3f}  ib={ib:8.3f}  ic={ic:8.3f}")
    print(f"turns per coil in slot   : {params.turns_per_coil_in_slot:12d}")
    print(f"slot area S [m^2]        : {params.slot_area_region_v:12.6e}")
    print(f"coil width d [rad]       : {params.coil_width_mech:12.6f}")
    print(f"slot pitch delta [rad]   : {params.slot_pitch_mech:12.6f}")
    print(f"max |J1| [A/m^2]         : {np.max(np.abs(solution.J1_slots)):12.6f}")
    print(f"max |J2| [A/m^2]         : {np.max(np.abs(solution.J2_slots)):12.6f}")
    print(f"max |Kslot| [A/m]        : {np.max(np.abs(solution.current_loading)):12.6f}")
    print()

    print("Dominant current-loading harmonics")
    for order, cosine, sine, magnitude in dominant_current_loading_rows(solution.current_spectrum):
        print(
            f"n={order:2d}  Kc={cosine:12.3f}  Ks={sine:12.3f}  |K|={magnitude:12.3f}"
        )
    print()

    print("Mid-gap field peaks")
    print(f"PM-only Br peak [T]      : {np.max(np.abs(solution.Br_pm_gap)):12.6f}")
    print(f"PM-only Btheta peak [T]  : {np.max(np.abs(solution.Btheta_pm_gap)):12.6f}")
    print(f"I-only Br peak [T]       : {np.max(np.abs(solution.Br_current_gap)):12.6f}")
    print(f"I-only Btheta peak [T]   : {np.max(np.abs(solution.Btheta_current_gap)):12.6f}")
    print(f"Total Br peak [T]        : {np.max(np.abs(solution.Br_total_gap)):12.6f}")
    print(f"Total Btheta peak [T]    : {np.max(np.abs(solution.Btheta_total_gap)):12.6f}")
    print()

    print("Average torque from Maxwell stress")
    print(f"PM-only torque [Nm]      : {solution.torque_pm:12.6f}")
    print(f"I-only torque [Nm]       : {solution.torque_current:12.6f}")
    print(f"Total torque [Nm]        : {solution.torque_total:12.6f}")
    print(f"Interaction torque [Nm]  : {solution.torque_interaction:12.6f}")
    print()

    print("Equivalent dq and back-EMF")
    print(f"id [A peak]              : {solution.id_current:12.6f}")
    print(f"iq [A peak]              : {solution.iq_current:12.6f}")
    print(f"lambda_pm [Wb-turn]      : {solution.lambda_pm_equiv:12.6f}")
    print(f"speed [rpm mech]         : {params.mechanical_speed_rpm:12.3f}")
    print(f"freq [Hz elec]           : {params.electrical_frequency_hz:12.3f}")
    print(f"phase back-EMF pk [V]    : {solution.emf_phase_peak:12.6f}")
    print(f"phase back-EMF rms [V]   : {solution.emf_phase_rms:12.6f}")
    print(f"mech power [W]           : {solution.mech_power:12.6f}")
    print()

    print("5-region scaffold")
    print(f"regions                   : {', '.join(scaffold.region_names)}")
    print(f"unknowns per harmonic     : {scaffold.unknowns_per_harmonic}")
    print(f"boundary conditions       : {scaffold.boundary_condition_count}")
    for note in scaffold.notes:
        print(f"note                      : {note}")
    for missing in scaffold.missing_equations:
        print(f"missing                   : {missing}")
    print()

    print("5-region PM-only preview")
    print(f"harmonics used [count]     : {len(pm_only_5region.harmonic_orders):12d}")
    print(f"block system size          : {pm_only_5region.system_size:12d}")
    print(f"solve residual             : {pm_only_5region.solve_residual:12.3e}")
    print(f"eval radius [m]           : {pm_only_5region.eval_radius:12.6f}")
    print(f"Br peak [T]               : {np.max(np.abs(pm_only_5region.Br_gap)):12.6f}")
    print(f"Btheta peak [T]           : {np.max(np.abs(pm_only_5region.Btheta_gap)):12.6f}")
    print(
        f"cogging pk-pk [Nm]        : "
        f"{np.ptp(pm_only_5region.cogging_torque_nm):12.6f}"
    )
    print(
        f"cogging max |T| [Nm]      : "
        f"{np.max(np.abs(pm_only_5region.cogging_torque_nm)):12.6f}"
    )
    print()

    print("PNG outputs")
    print(f"output dir               : {output_dir.resolve()}")
    print(f"current loading          : {current_loading_png.resolve()}")
    print(f"air-gap fields           : {air_gap_png.resolve()}")
    print(f"harmonics                : {harmonics_png.resolve()}")
    print(f"torque vs angle          : {torque_angle_png.resolve()}")
    print(f"motor schematic          : {motor_schematic_png.resolve()}")
    print(f"5-region PM-only fields  : {pm_only_5region_fields_png.resolve()}")
    print(f"5-region PM-only cogging : {pm_only_5region_cogging_png.resolve()}")


if __name__ == "__main__":
    main()

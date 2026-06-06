# Paper [5] Source Audit

This audit only lists sources explicitly used for the current paper [5] workflow.

Allowed sources for this stage:

- `V_shape_equavalent.py`: equivalent V-shape geometry builder. Do not edit.
- `V_shape_equavalent_br_bt_export.py`: FEMM Br/Bt benchmark exporter. Do not edit.
- `build_paper_vshape_model.py`: imported by `V_shape_equavalent.py`; contains the motor spec used by the equivalent builder.
- Local PDF `[5] ... .pdf`: paper reference. Formula extraction is not automated yet in this workspace.

## Motor Spec From Code

Source: `build_paper_vshape_model.py`, `PAPER_SPECS["vshape"]`.

| Parameter | Value | Source Line |
| --- | ---: | --- |
| `stator_inner_radius` | 40.0 mm | `build_paper_vshape_model.py:85` |
| `rotor_outer_radius` | 38.8 mm | `build_paper_vshape_model.py:86` |
| `poles` | 6 | `build_paper_vshape_model.py:87` |
| `slots` | 36 | `build_paper_vshape_model.py:88` |
| `stack_length` | 80.0 mm | `build_paper_vshape_model.py:89` |
| `slot_opening_span` | 0.05 rad | `build_paper_vshape_model.py:90` |
| `slot_span` | 0.105 rad | `build_paper_vshape_model.py:91` |
| `slot_top_radius` | 40.8 mm | `build_paper_vshape_model.py:92` |
| `slot_bottom_radius` | 60.0 mm | `build_paper_vshape_model.py:93` |
| `bridge_width_1` | 1.5 mm | `build_paper_vshape_model.py:94` |
| `bridge_length_1` | 3.6 mm | `build_paper_vshape_model.py:95` |
| `bridge_width_2` | 1.5 mm | `build_paper_vshape_model.py:96` |
| `bridge_length_2` | 3.8 mm | `build_paper_vshape_model.py:97` |
| `magnet_width` | 14.4 mm | `build_paper_vshape_model.py:98` |
| `magnet_thickness` | 4.0 mm | `build_paper_vshape_model.py:99` |
| `magnet_remanence_t` | 1.26 T | `build_paper_vshape_model.py:101` |
| `magnet_coercivity_ka_per_m` | 955.0 kA/m | `build_paper_vshape_model.py:102` |

## Equivalent Geometry Defaults

Source: `V_shape_equavalent.py`.

| Parameter | Value | Source Line |
| --- | ---: | --- |
| `EQUIVALENT_OUTPUT_DEFAULT` | `paper_ipm_vshape_equivalent_1over6.FEM` | `V_shape_equavalent.py:10` |
| `EQUIVALENT_W1_MM_DEFAULT` | 2.5 mm | `V_shape_equavalent.py:11` |
| `EQUIVALENT_W2_MM_DEFAULT` | 1.2 mm | `V_shape_equavalent.py:12` |
| `EQUIVALENT_CURRENT_RMS_A_DEFAULT` | 0 A | `V_shape_equavalent.py:13` |
| `EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT` | 1.7 mm | `V_shape_equavalent.py:14` |

Equivalent dimensions are calculated by `equivalent_pm_dimensions()` in `V_shape_equavalent.py:30`.

Current default calculated values used by `paper5_vshape_stage1.py`:

| Derived Quantity | Value |
| --- | ---: |
| `alpha` | 55.5699139552 deg |
| `alpha1` | 41.5525978663 deg |
| `Rf` | 23.3516790536 mm |
| `Rm` | 27.3516790536 mm |
| `Rl` | 36.9 mm |
| `Rl - w1` | 34.4 mm |

## FEMM Benchmark Defaults

Source: `V_shape_equavalent_br_bt_export.py`.

| Parameter | Value | Source Line |
| --- | ---: | --- |
| `DEFAULT_AIRGAP_RADIUS_MM` | 39.4 mm | `V_shape_equavalent_br_bt_export.py:13` |
| `DEFAULT_ARC_START_DEG` | 0.0 deg | `V_shape_equavalent_br_bt_export.py:14` |
| `DEFAULT_ARC_END_DEG` | 60.0 deg | `V_shape_equavalent_br_bt_export.py:15` |
| `DEFAULT_ARC_SAMPLE_MARGIN_DEG` | 0.01 deg | `V_shape_equavalent_br_bt_export.py:16` |
| `DEFAULT_NUM_FIELD_POINTS` | 301 | `V_shape_equavalent_br_bt_export.py:17` |
| `RUN_CURRENT_RMS_A` | 0.0 A | `V_shape_equavalent_br_bt_export.py:25` |
| `DEFAULT_PERIODIC_MULTIPLIER` | 6 | `V_shape_equavalent_br_bt_export.py:19` |

Generated benchmark from the current FEMM run:

- `results/paper_vshape_equivalent_1over6/br_bt_arc.csv`
- `results/paper_vshape_equivalent_1over6/br_bt_export_summary.csv`

This is the slotted equivalent FEMM benchmark and should be used after adding conformal mapping.

Stage-1 slotless benchmark:

- builder/exporter: `paper5_slotless_equivalent_femm_export.py`
- FEMM model: `paper5_slotless_equivalent_1over6.FEM`
- Br/Bt CSV: `results/paper5_slotless_equivalent_1over6/br_bt_arc.csv`

## FEMM Permeability Sampling

New helper:

- `paper5_femm_mu_sampler.py`

This helper does not choose bridge sample points automatically. Pass explicit points:

```powershell
python .\paper5_femm_mu_sampler.py --point x_mm,y_mm,bridge_name
```

or pass a CSV with columns `name`, `x_mm`, `y_mm`:

```powershell
python .\paper5_femm_mu_sampler.py --points-csv bridge_points.csv
```

The output column to pass into the stage-1 solver is usually:

- `mu_r_abs_from_B_over_mu0H`

FEMM's returned directional values are also saved:

- `mu_x_femm`
- `mu_y_femm`

Legacy auto-sampled bridge values from the slotless equivalent FEMM. These are diagnostic only and are not the
current source of bridge permeability:

- `results/paper5_slotless_femm_mu_bridge_auto.csv`

Current user-selected saturated bridge samples from the actual equivalent FEMM:

- FEMM file: `paper_ipm_vshape_equivalent_1over6.FEM`
- CSV: `results/paper5_equivalent_femm_mu_user_bridge_points.csv`
- geometry audit plot: `results/paper5_geometry_audit/equivalent_geometry_user_bridge_zoom.png`
- geometry audit summary: `results/paper5_geometry_audit/equivalent_geometry_user_bridge_summary.csv`

Current values:

| Point | x (mm) | y (mm) | `|B|` (T) | `|H|` (A/m) | `mu_r` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `outer_bridge_lower` | 37.7 | 4.3 | 1.82172 | 99174.2 | 14.6175 |
| `outer_bridge_upper` | 22.5 | 30.4 | 1.82432 | 101246 | 14.3388 |
| `center_bridge` | 21.9 | 12.7 | 1.89457 | 157227 | 9.58905 |

For analytical runs, use the two outer bridge points as the side-bridge permeability estimate
(`mu_outer_avg ~= 14.4782`) and the center point as the center/inner saturated permeability estimate.

## Unknown Or User-Supplied For Stage 1

These values must not be silently invented:

| Parameter | Current Handling |
| --- | --- |
| `mu_bridge` | CLI argument in `paper5_vshape_stage1.py`; default `1.0` placeholder only |
| nonlinear bridge permeability update | TODO: extract/sample from FEMM or derive from paper MEC |
| exact paper matrix rows | TODO: extract from PDF/manual transcription |
| conformal mapping slotting function | TODO: implement only after slotless PM/bridge source is validated |

# Paper [5] Formula Notes For Current Implementation

This file records what has been extracted from the PDF and what the current code does differently.

Rendered formula pages:

- `results/paper5_pdf_pages/page_03.png`
- `results/paper5_pdf_pages/page_04.png`
- `results/paper5_pdf_pages/page_05.png`
- `results/paper5_pdf_pages/page_06.png`
- `results/paper5_pdf_pages/page_09.png`
- `results/paper5_pdf_pages/page_10.png`

Text extraction:

- `results/paper5_pdf_text.txt`

## Important Paper Structure

The paper does not solve the equivalent V-shape model as one radial stack with theta-windowed PM sources.

It splits the motor into two superposed structures:

- Structure 1: magnets distributed along radial direction.
- Structure 2: magnets distributed along tangential direction.

Then it solves each simplified model and combines:

```text
Br = B'IVr + B'Vr
Btheta = B'IVtheta + B'Vtheta
```

These are equations (40) and (41) on page 6.

## Structure 1 Notes

Page 4 gives the Structure 1 vector potential forms:

- Subdomain I: air slot.
- Subdomain II: PM.
- Subdomain III: magnetic bridge.
- Subdomain IV: airgap.

The PM subdomain has the particular term:

```text
AII = A02 + B02 ln(r) + Brem r + harmonic terms
```

This is important. The current `paper5_vshape_stage1.py` instead uses theta-windowed remanence Fourier coefficients in one radial stack, so it is not yet the same model as the paper.

Page 4 also gives:

```text
HIVtheta(Rs, theta) = 0
```

for the slotless airgap outer boundary, assuming high-permeability stator iron. The current scaffold uses a finite stator yoke and `A=0` at stator outer radius, because that matches the FEMM benchmark geometry. A later paper-faithful solver should use the paper boundary condition for analytical equations.

## Structure 2 Notes

Page 6 gives the Structure 2 airgap field:

```text
BVr = sum_n [np A5n/r (r/Rs)^np + np B5n/r (r/Rr)^(-np)] cos(np theta)
BVtheta = -sum_n [np A5n/Rs (r/Rs)^(np-1) - np B5n/Rr (r/Rr)^(-np-1)] sin(np theta)
```

Then the paper applies an amplitude correction:

```text
K'mod = phi_g2 / phi'_g2
B'Vr = BVr K'mod
B'Vtheta = BVtheta K'mod
```

## Corrections Missing From Current Scaffold

The current scaffold is useful for data plumbing and benchmark comparison, but it is not paper-faithful yet.

Missing:

- Structure 1 and Structure 2 separate solves.
- Paper boundary equations from Appendix A/B.
- MEC amplitude correction `Kmod` and `K'mod`.
- Piecewise bridge-region correction from equations (24), (25), (38), and (39).
- Conformal mapping slotting equations (42), (43).

## Current Benchmarks

Slotted equivalent FEMM:

- `results/paper_vshape_equivalent_1over6/br_bt_arc.csv`

Slotless equivalent FEMM, created specifically for stage-1 validation:

- `paper5_slotless_equivalent_1over6.FEM`
- `results/paper5_slotless_equivalent_1over6/br_bt_arc.csv`

Legacy slotless bridge permeability samples. These came from an automatic point guess and are retained only as
diagnostics:

- `results/paper5_slotless_femm_mu_bridge_auto.csv`

Current trusted saturated samples from the user's equivalent FEMM:

```text
outer_bridge_lower: mu_r ~= 14.6175 at (37.7, 4.3)
outer_bridge_upper: mu_r ~= 14.3388 at (22.5, 30.4)
center_bridge: mu_r ~= 9.58905 at (21.9, 12.7)
```

The source CSV is `results/paper5_equivalent_femm_mu_user_bridge_points.csv`.
The geometry check is `results/paper5_geometry_audit/equivalent_geometry_user_bridge_zoom.png`.

## Current Structure 1 Test

New helper:

- `paper5_structure1_solver.py`

This implements Appendix A equations (47)-(57) for Structure 1 as a first paper-faithful slice.

Best current run:

```powershell
python .\paper5_structure1_solver.py --mu-bridge 17.23 --theta-offset-deg 30 --n-harmonics 3 --k-harmonics 3 --g-harmonics 3 --result-dir results/paper5_structure1_mub1723_offset30_h3
```

Result against slotless equivalent FEMM:

```text
Structure 1 Br rms = 0.0621685 T
Slotless FEMM Br rms = 0.522822 T
Br relative L2 = 0.882968
Best scalar Br x 8.29262 -> L2 = 0.166317
```

Interpretation:

- Structure 1 alone has a recognizable Br shape after the correct theta offset.
- Its amplitude is far too small before the paper's correction factors and Structure 2 superposition.
- The matrix becomes highly ill-conditioned as harmonic count increases.

## Paper Applicability Warning

The PDF text around page 9 states the proposed model is more suitable when the relative permeability of magnetic bridges is greater than 28 under no-load condition.

Current FEMM bridge samples are below that:

```text
center_bridge: mu_r ~= 10.0342
outer_bridges: mu_r ~= 17.2
```

So the user's current geometry appears to be in the strongly saturated bridge regime where the paper itself warns accuracy may degrade.

## Current Structure 2 Test

New helper:

- `paper5_structure2_solver.py`

This implements Appendix B equations (65)-(72) for Structure 2.

Current best direct run:

```powershell
python .\paper5_structure2_solver.py --theta-offset-deg 30 --n-harmonics 5 --m-harmonics 5 --k-harmonics 5 --result-dir results/paper5_structure2_offset30_h5
```

Result against slotless equivalent FEMM:

```text
Structure 2 Br rms = 0.534482 T
Slotless FEMM Br rms = 0.522822 T
Br relative L2 = 0.143382
Best scalar Br x 0.968586 -> L2 = 0.139739
```

This indicates Structure 2 is the dominant term for the current equivalent geometry.

## Current Combined Slotless Test

New helper:

- `paper5_combine_structures.py`

Direct paper-style superposition using current Structure 1 and 2 outputs:

```powershell
python .\paper5_combine_structures.py
```

Result:

```text
Combined Br rms = 0.596559 T
Slotless FEMM Br rms = 0.522822 T
Br relative L2 = 0.207156
```

Structure 2 alone is currently better than the direct sum. This suggests at least one of the following remains unresolved:

- Structure 1 sign/phase convention.
- Structure 1 correction Eq. (24)-(27), including `Kmod`.
- MEC correction factors are not yet implemented.
- The paper's simplified model is strained because the current bridge permeability is below the paper's stated favorable range.

Diagnostic least-squares coefficients against slotless FEMM:

```text
Br: Structure1 coefficient ~= -6.34, Structure2 coefficient ~= 1.70, L2 ~= 0.133
Bt: Structure1 coefficient ~= -2.45, Structure2 coefficient ~= 1.19, L2 ~= 0.646
```

These coefficients are diagnostic only and are not used as physical correction factors.

## Current Slotting Correction Test

New helper:

- `paper5_slotting_correction.py`

This applies the same coupling form as the paper's conformal mapping step:

```text
Bslot_r = Br lambda_a + Btheta lambda_b
Bslot_theta = Btheta lambda_a - Br lambda_b
```

For now, `lambda_a` and `lambda_b` are inferred empirically from the slotless and slotted FEMM benchmarks,
then fit as a 36-slot periodic Fourier waveform. This is a diagnostic use of Eq. (42)-(43), not yet an
independent conformal-mapping prediction.

Best current diagnostic run with Structure 2 after harmonic sweep:

```powershell
python .\paper5_structure2_solver.py --theta-offset-deg 30 --n-harmonics 7 --m-harmonics 7 --k-harmonics 7 --result-dir results/paper5_structure2_offset30_h7
python .\paper5_slotting_correction.py --model-csv results/paper5_structure2_offset30_h7/structure2_br_bt.csv --result-dir results/paper5_slotting_structure2_h7_empirical_lh6 --lambda-harmonics 6
```

Result against slotted equivalent FEMM:

```text
Corrected Br rms = 0.536554 T
Slotted FEMM Br rms = 0.510678 T
Br relative L2 = 0.139729

Corrected Bt rms = 0.0499568 T
Slotted FEMM Bt rms = 0.0490444 T
Bt relative L2 = 0.496111
```

Plots:

- `results/paper5_slotting_structure2_h7_empirical_lh6/lambda_empirical_fit.png`
- `results/paper5_slotting_structure2_h7_empirical_lh6/slotting_corrected_field_comparison.png`

Self-check using the slotless FEMM field as input:

```powershell
python .\paper5_slotting_correction.py --model-csv results/paper5_slotless_equivalent_1over6/br_bt_arc.csv --result-dir results/paper5_slotting_selfcheck_h6 --lambda-harmonics 6
```

Result:

```text
Br relative L2 = 0.0287018
Bt relative L2 = 0.167008
```

Interpretation:

- The slotting correction now creates the tooth/slot ripple visible in FEMM.
- The tangential component RMS becomes close to slotted FEMM, but its local waveform still has notable error.
- The remaining Br error is mostly inherited from the slotless Structure 2 waveform rather than from the
  Eq. (42)-(43) mixing form.

Harmonic audit helper:

- `paper5_field_harmonic_audit.py`
- plot: `results/paper5_harmonic_audit_current/field_harmonic_spectrum.png`
- coefficients: `results/paper5_harmonic_audit_current/field_harmonic_coefficients.csv`

The spectrum shows slotting correctly adds strong slot harmonics around sector harmonics 6 and 12, while the
slotless analytical waveform already differs from slotless FEMM in the low-order components.

## Edge Error Investigation

The current error is concentrated near the two sector edges. A compact report helper was added:

- `paper5_current_best_report.py`

Previous checkpoint:

- `results/paper5_current_best/current_best_focus_report.png`
- `results/paper5_current_best/current_best_region_errors.csv`

This confirmed the middle region is much better than the two edges.

Structure 1 diagnostic:

- `paper5_edge_structure1_sweep.py`
- `results/paper5_edge_structure1_sweep_h3_scaled/best_edge_structure1_diagnostic.png`

Result: adding the current Structure 1 implementation does not naturally fix the edge error. Fits that strongly
reduce edge Br require nonphysical scaling or damage the middle region. Row/column scaling reduces the S1 matrix
condition from roughly `2.6e14` to `5.2e3`, but the resulting field shape is essentially unchanged.

Structure 2 geometry/window sweep:

Changing the equivalent span parameter used by the Structure 2 solver improves the edge shape. The best current
balanced checkpoint is:

```powershell
python .\paper5_structure2_solver.py --theta-offset-deg 30 --n-harmonics 7 --m-harmonics 7 --k-harmonics 7 --alpha-deg 52 --result-dir results/paper5_structure2_alpha52_offset30_h7
python .\paper5_slotting_correction.py --model-csv results/paper5_structure2_alpha52_offset30_h7/structure2_br_bt.csv --result-dir results/paper5_slotting_structure2_alpha52_h7_lh6 --lambda-harmonics 6
python .\paper5_current_best_report.py --result-dir results/paper5_current_best_alpha52 --subdomain results/paper5_structure2_alpha52_offset30_h7/structure2_br_bt.csv --corrected results/paper5_slotting_structure2_alpha52_h7_lh6/slotting_corrected_br_bt.csv
```

Result against slotted FEMM:

```text
Br relative L2 = 0.103413
Bt relative L2 = 0.380680
```

Regional errors for `subdomain + FEMM-derived slotting`:

```text
Br left edge  = 0.2146
Br middle     = 0.0654
Br right edge = 0.2138

Bt left edge  = 0.5170
Bt middle     = 0.1599
Bt right edge = 0.5534
```

Interpretation: a significant part of the edge error came from the Structure 2 span/window geometry rather than
from slotting. The remaining Bt edge error still needs either a more faithful Structure 1/correction implementation
or direct extraction of Structure 2 window parameters from the equivalent FEMM geometry instead of using the
paper-equivalence formula blindly.

## Geometry-Only Slotting Step

New helper:

- `paper5_slotting_geometry_model.py`

This replaces the earlier FEMM-derived `lambda_a/lambda_b` with a geometry-only slot permeance waveform. It uses:

- slot count: `Qs = 36`
- slot centers in the 60 deg sector: `5, 15, 25, 35, 45, 55 deg`
- slot opening span: `0.05 rad = 2.86479 deg`
- a normalized Gaussian notch for `lambda_a`
- a derivative-like quadrature term for `lambda_b`

This is still a parametric relative-permeance approximation, not a full Schwarz-Christoffel conformal mapping.
However, it no longer derives `lambda_a/lambda_b` from FEMM field values.

Current default run:

```powershell
python .\paper5_slotting_geometry_model.py --result-dir results/paper5_slotting_geometry_alpha52
python .\paper5_current_best_report.py --result-dir results/paper5_current_best_alpha52_geometry_slot --subdomain results/paper5_structure2_alpha52_offset30_h7/structure2_br_bt.csv --corrected results/paper5_slotting_geometry_alpha52/geometry_slotting_br_bt.csv --corrected-label "subdomain + geometry slotting"
```

Current tuned parameters:

```text
lambda_drop = 0.32
lambda_b_gain = 0.14
width_scale = 1.0
normalize_mean = True
```

Result against slotted FEMM:

```text
Br relative L2 = 0.100290
Bt relative L2 = 0.371349
```

Regional errors:

```text
Br left edge  = 0.1983
Br middle     = 0.0686
Br right edge = 0.1988

Bt left edge  = 0.5204
Bt middle     = 0.1628
Bt right edge = 0.5195
```

Compared with FEMM-derived slotting at the same Structure 2 alpha=52 (`Br=0.1034`, `Bt=0.3807`), the
geometry-only slotting is slightly better on global L2. This suggests the slotting direction is usable, but the
remaining error still comes mainly from the slotless/edge field model rather than from missing tooth ripple.

## Structure 1 Correction Test

`paper5_structure1_solver.py` now supports Eq. (24)-(27):

```powershell
python .\paper5_structure1_solver.py --mu-bridge 17.23 --theta-offset-deg 30 --n-harmonics 3 --k-harmonics 3 --g-harmonics 3 --apply-correction --result-dir results/paper5_structure1_mub1723_offset30_h3_corrected
```

Structure 1 corrected alone:

```text
Br rms = 0.0623147 T
Br relative L2 = 0.882334
Best scalar Br x 8.29552 -> L2 = 0.149668
Bt relative L2 = 0.92254
```

Direct corrected Structure 1 + Structure 2:

```text
Br relative L2 = 0.205196
Bt relative L2 = 0.664704
```

Diagnostic with Structure 2 fixed and only `Kmod` fitted for `Br`:

```text
Kmod ~= -0.25837
Br relative L2 = 0.140036
```

This is diagnostic only. It indicates the current Structure 1 sign/correction path is still not physically settled.

Updated overview plot:

- `results/paper5_plots_corrected/field_comparison.png`
- `results/paper5_plots_corrected/geometry_bridge_points.png`

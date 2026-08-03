# Stanford WSS label audit

Updated: 2026-08-03

## Scope and decision boundary

This audit covers all 12 **development** cases. Locked cases `0033`, `0039`,
and `0042` were not accessed. No WSS target was clipped, replaced, or otherwise
modified. Separate, reason-coded volume-quality masks define which nodes are
eligible for the primary loss and evaluation.

The audit independently re-read the original Stanford surface-result arrays at
the first, global-peak, and final phases; reconstructed the wall mapping; and
compared the source values against the canonical Zarr targets. It also measured
WSS tangentiality, temporal and spatial peak coherence, cycle closure, open-boundary
distance, triangle quality, and semantic-region distribution.

## Findings

| Case | Max WSS | Source reproduction | Tangentiality p99 | Peak / max neighbour | Cycle endpoint mismatch | Provisional disposition |
|---|---:|---|---:|---:|---:|---|
| 0031 | 182.89 Pa | pass | 0.157% | 1.62 | 0.004% | Accept under primary mask |
| 0032 | 584.13 Pa | pass | 0.296% | 2.64 | 0.004% | Accept; unreliable focal peak excluded by primary extreme-volume mask |
| 0034 | 157.99 Pa | pass | 0.203% | 1.40 | 0.745% | Accept under primary mask |
| 0035 | 64.43 Pa | pass | 0.117% | 1.12 | 0.587% | Accept under primary mask |
| 0036 | 197.76 Pa | pass | 0.286% | 1.13 | 0.059% | Accept under primary mask |
| 0037 | 99.19 Pa | pass | 0.151% | 1.00 | 0.006% | Accept under primary mask |
| 0038 | 124.65 Pa | pass | 0.003% | 1.21 | 0.090% | Accept under primary mask |
| 0040 | 394.73 Pa | pass | 0.193% | 1.18 | 0.095% | Accept; global peak excluded by primary extreme-volume mask |
| 0041 | 330.81 Pa | pass | 0.004% | 1.12 | 7.61% | Accept after cycle-metric sensitivity |
| 0043 | 112.47 Pa | pass | 0.015% | 1.01 | 19.76% | Accept after cycle-metric sensitivity |
| 0044 | 77.80 Pa | pass | 0.009% | 2.002 | 0.150% | Accept; retain peak in primary analysis and test severe-mask sensitivity |
| 0045 | 111.53 Pa | pass | 0.123% | 1.07 | 0.318% | Accept under primary mask |

All 12 source archives reproduce the canonical values exactly when the
documented float32 `dyn/cm^2 -> Pa` conversion order is applied. Independent
float64 conversion differed by at most `3.01e-5 Pa`. The reconstructed mappings
were one-to-one and had maximum source-coordinate discrepancy `5.06e-4 cm`
(`0.0051 mm`).

None of the 12 cases showed a one-frame temporal spike, material WSS normal
component, concentration on the worst-aspect surface elements, or majority
concentration within 2 mm of an open boundary. This argues against a general
unit, mapping, vector-axis, surface-normal, boundary-ring, or poor-triangle
failure.

## Case 0032 focal peak

Case 0032 has a coherent high-WSS population: 1,968 nodes exceed 100 Pa at some
point in the cycle, and 1,788 exceed 100 Pa at the global-peak phase. However,
the 584.13 Pa global maximum is locally discontinuous:

- The four immediate neighbours are 205.47, 23.88, 50.01, and 221.30 Pa.
- The peak is 2.64 times the strongest neighbour and 4.57 times the median
  neighbour.
- Only two surface nodes exceed half the global peak, and they are not in the
  same above-half-peak connected component.
- The peak is not a temporal spike, open-boundary node, or worst-aspect-element
  node, and it is present in the original Stanford result.

The source volume-mesh follow-up found that the peak surface node touches the
globally worst aspect-ratio tetrahedron in the 2,571,027-cell mesh. That cell has
aspect ratio 1,768.37, scaled Jacobian 0.000185, and volume
`1.46e-9 cm^3`. The 584.13 Pa value is therefore rejected as a reliable training
label even though it is faithfully copied from the Stanford output.

This is a focal problem rather than evidence that all high case-0032 WSS is
invalid. Under the registered severe volume-QC sensitivity definition, only 58
of 1,968 nodes above 100 Pa touch a severe cell. Five high-WSS nodes touch an
extreme cell. Four of the six nodes above 200 Pa touch a severe cell, including
the only node above 300 Pa.

Canonical targets remain unchanged. The same volume-QC rule was frozen and then
evaluated across every development case so the exclusion was not chosen
specifically to improve case 0032.

That cohort audit is now complete. The extreme rule excludes 143 of 1,198,219
development nodes (0.0119%) and is adopted as the primary development loss and
evaluation mask. The broader severe rule affects 3.09% and remains sensitivity
only. Case 0040's 394.73 Pa global peak is also extreme-volume-mesh-adjacent;
case 0041's spatially coherent 330.81 Pa peak remains valid. See
`docs/VOLUME_MESH_QC.md`.

## Case 0041 cycle closure

Case 0041 contains the most spatially coherent high-WSS region: 3,832 nodes
exceed 100 Pa over the cycle, primarily assigned to renal branches. Its peak is
smooth in time and space. The concern is instead a 7.61% relative vector-L2
mismatch between the first and last WSS fields despite an exactly matched cycle
duration and inlet-flow endpoints.

The endpoint mismatch is region dependent: approximately 18.7% in the aorta,
7.6% in renal regions, 3.0% in mesenteric regions, 0.6% in
celiac/hepatic/splenic regions, 10.1% in iliac regions, and 35.7% along the small
explicit aneurysm path.

The registered endpoint sensitivity analysis compared the raw cycle with
first-endpoint, last-endpoint, and midpoint periodic closure. Across the whole
surface, the worst changes were 0.0228% relative TAWSS error, OSI MAE 0.0000317,
and 0.338% relative RRT error. The worst regional changes were 0.0403% TAWSS,
0.000273 OSI, and 0.587% RRT. These are comfortably within the registered 1%
TAWSS, 0.01 OSI, and 5% RRT sensitivity thresholds. Case 0041 phase-wise WSS
and derived TAWSS/OSI/RRT are accepted for development without modifying the
stored cycle.

## Case 0043 cycle closure

Case 0043 has a 19.76% relative vector-L2 mismatch between its first and last
WSS fields even though the target duration and boundary-condition period both
equal 0.857 seconds. The absolute pointwise differences remain modest: median
0.0194 Pa, p99 1.327 Pa, and maximum 3.234 Pa.

The same registered closure variants used for 0041 changed whole-surface TAWSS
by at most 0.0776%, OSI by 0.000237 MAE, and RRT by 1.358%. These pass the
registered 1% TAWSS, 0.01 OSI, and 5% RRT thresholds. The original fields and
their derived cycle metrics are accepted without endpoint replacement.

## Case 0044 local peak

Case 0044's 77.80 Pa maximum triggered the spatial screen because it is 2.0019
times its strongest immediate neighbour, only 0.096% beyond the registered 2.0
review threshold. It is exactly reproduced from the source, is 32.24 mm from an
open boundary, has ordinary incident surface-triangle quality, and sits within
a broader population of 1,488 nodes above half the phase maximum (largest
connected component: 816 nodes).

The peak touches a volume cell with minimum incident scaled Jacobian 0.00935,
so it enters the broad severe sensitivity mask, but it is not in the primary
extreme-invalid mask. It remains eligible for the primary analysis and must be
reported again under the registered severe-mask sensitivity. This rule was
frozen cohort-wide and was not selected to rescue or reject case 0044.

## Modeling implication

The highest WSS values occur predominantly in renal, mesenteric, celiac, and
iliac branches rather than across the aneurysm sac. Global WSS errors can
therefore be dominated by small, complex branches even when the central AAA
surface is predicted better. Training and evaluation must retain these branches
but report aneurysm/aorta and branch regions separately and use region-balanced,
robust losses.

## Artifacts

The machine-readable summary is
`data/raw/stanford_vmr/canonical/wss_label_audit_summary.json`. Each audited case
contains:

- `diagnostics/wss_label_audit.json`: source hashes and numerical evidence.
- `diagnostics/wss_label_audit.vtp`: ParaView-ready diagnostic arrays.
- `diagnostics/wss_label_audit.png`: full-surface preview with high-WSS nodes and
  global peak highlighted.

Follow-up artifacts include:

- `0032/diagnostics/0032_peak_volume_mesh_audit.json` and `.vtp`.
- `0032/diagnostics/0032_peak_incident_volume_cells.vtu`.
- `0041/diagnostics/0041_cycle_endpoint_sensitivity.json`.
- `0043/diagnostics/0043_cycle_endpoint_sensitivity.json`.
- `wss_followup_audit_summary.json` at the canonical root.
- `wss_label_audit_adjudication.json` at the canonical root records the final
  development disposition for all 12 cases.

The audit is reproducible with:

```powershell
.\.venv\Scripts\python.exe -m aorta_surrogate.data.wss_label_audit `
  --canonical-root data\raw\stanford_vmr\canonical `
  --source-root data\raw\stanford_vmr `
  --verify-source
```

## Next actions

1. Add region-balanced reporting so branch extremes do not conceal central-aorta
   performance.
2. Run primary-versus-severe-mask sensitivity on the next development experiment.
3. Keep the locked test cases closed until the model and all audit rules are
   frozen.

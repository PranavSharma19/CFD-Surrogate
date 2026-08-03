# Primary versus severe volume-mask ablation

Updated: 2026-08-03

## Decision

Retain the frozen **extreme-volume mask** as the primary loss/evaluation policy
and retain the broader **severe-volume mask** as a reported sensitivity only.
The severe mask does not materially change the cross-validated performance or
the conclusion that the current equivariant surrogate remains far from the 15%
research gate.

This means suspect volume-mesh neighborhoods are not the main cause of the
model's present error. Further work should focus on data scale, conditioning,
model capacity, sampling, and regional learning rather than broad label removal.

## Controlled design

- Frozen best checkpoints from `equivariant_long_fold0`, `fold1`, and `fold2`.
- Frozen fold-specific normalization, 4,096-node connected geodesic patches,
  validation phases 0, 5, 10, 15, and 20, and deterministic patch seeds.
- Every development patient appears in validation exactly once.
- The same checkpoint predictions are scored twice; only the target-valid mask
  changes.
- Primary: exclude the 143 cohort nodes registered as extreme-volume invalid.
- Sensitivity: additionally exclude every severe-volume node.
- Locked cases `0033`, `0039`, and `0042` were not opened.

The primary reevaluation reproduced each registered best-checkpoint magnitude
MAE within `0.0001 Pa`. The small run-to-run differences are consistent with
CUDA scatter-reduction ordering and are negligible relative to model error.

## Aggregate results

| Metric | Primary mask | Severe sensitivity | Change |
|---|---:|---:|---:|
| Sampled nodes | 245,752 | 239,835 | -5,917 (-2.41%) |
| WSS vector MAE | 5.33524 Pa | 5.33493 Pa | -0.00031 Pa |
| WSS magnitude MAE | 4.50379 Pa | 4.50347 Pa | -0.00031 Pa |
| WSS magnitude relative error | 65.236% | 65.500% | +0.264 percentage points |
| Mean angular error | 27.530 degrees | 27.685 degrees | +0.155 degrees |
| Supported-region macro relative error | 69.794% | 70.073% | +0.280 percentage points |
| All-region macro relative error | 101.561% | 101.794% | +0.233 percentage points |

The support-aware macro requires at least 1,000 sampled nodes and includes the
aorta, renal, mesenteric, celiac/hepatic/splenic, and iliac regions. The
all-region macro also retains the explicit-aneurysm-path result for transparency,
but that region has only 104 sampled nodes and a 260% relative error; it should
not carry equal evidentiary weight until coverage improves.

## Regional sensitivity

| Region | Primary nodes | Severe nodes | Primary relative error | Severe relative error | Change |
|---|---:|---:|---:|---:|---:|
| Aorta | 41,467 | 40,860 | 78.499% | 79.279% | +0.781 pp |
| Renal | 99,876 | 97,844 | 62.963% | 63.396% | +0.433 pp |
| Mesenteric | 25,791 | 25,015 | 66.482% | 66.637% | +0.155 pp |
| Celiac/hepatic/splenic | 52,263 | 50,363 | 54.432% | 54.389% | -0.043 pp |
| Iliac | 26,251 | 25,649 | 86.594% | 86.666% | +0.072 pp |
| Explicit aneurysm path | 104 | 104 | 260.396% | 260.396% | approximately 0 pp |

No supported anatomical region changes by one percentage point. The largest
patient-level change is case 0034 at +3.28 percentage points, followed by case
0040 at +0.54 points. Case 0044 changes by only +0.034 points, so its retained
77.8 Pa peak does not materially control cross-validated performance. Case 0032
changes by -0.013 points, confirming that its excluded 584 Pa extreme peak is
not driving the aggregate error.

## Boundaries

This is an evaluation-mask sensitivity analysis, not a retraining comparison.
That is intentional: keeping model weights and predictions fixed isolates the
effect of label eligibility. It uses sampled patches at five phases and does not
replace the later full-surface, complete-cycle cloud evaluation. A future
production-scale experiment should continue to report both registered masks.

## Reproduction

The full machine-readable report is
`data/raw/stanford_vmr/canonical/experiments/mask_ablation/equivariant_mask_ablation.json`.

```powershell
python -m aorta_surrogate.training.mask_ablation `
  --canonical-root data\raw\stanford_vmr\canonical `
  --output-dir data\raw\stanford_vmr\canonical\experiments\mask_ablation
```

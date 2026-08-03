# Development-cohort source volume-mesh QC

Updated: 2026-08-03

## Decision

Apply the frozen **extreme** source-volume mesh rule as a primary loss and
evaluation mask for development data:

- Invalid when minimum incident tetrahedral scaled Jacobian is below `0.001`, or
- Invalid when maximum incident tetrahedral aspect ratio exceeds `1000`.

Keep the broader **severe** rule (`scaled Jacobian < 0.01` or aspect ratio
`> 100`) for sensitivity analysis only. Canonical WSS targets remain unchanged.
The masks are stored separately in each development case's
`quality_masks.zarr`.

## Cohort result

The frozen rules were applied identically to all 12 development patients before
the cohort result was inspected. Locked patients 0033, 0039, and 0042 were not
opened.

| Case | Surface nodes | Severe nodes | Extreme nodes | Maximum WSS | Global peak extreme? |
|---|---:|---:|---:|---:|---|
| 0031 | 98,186 | 4,352 | 27 | 182.89 Pa | No |
| 0032 | 92,928 | 5,512 | 30 | 584.13 Pa | Yes |
| 0034 | 90,243 | 4,179 | 2 | 157.99 Pa | No |
| 0035 | 98,974 | 183 | 0 | 64.43 Pa | No |
| 0036 | 82,545 | 6,277 | 25 | 197.76 Pa | No |
| 0037 | 99,809 | 73 | 2 | 99.19 Pa | No |
| 0038 | 106,486 | 1,135 | 0 | 124.65 Pa | No |
| 0040 | 89,427 | 6,041 | 19 | 394.73 Pa | Yes |
| 0041 | 129,489 | 6,353 | 31 | 330.81 Pa | No |
| 0043 | 102,782 | 965 | 0 | 112.47 Pa | No |
| 0044 | 112,535 | 1,431 | 4 | 77.80 Pa | No |
| 0045 | 94,815 | 573 | 3 | 111.53 Pa | No |

Across 1,198,219 development surface nodes:

- The severe rule marks 37,074 nodes (3.094%).
- The extreme rule marks 143 nodes (0.0119%).
- Only 13 of 8,828 nodes above 100 Pa are extreme.
- Six of 89 nodes above 200 Pa are extreme.
- Three of four nodes above 300 Pa are extreme.
- The sole node above 500 Pa is extreme.

The severe rule is too broad for a primary mask. Its prevalence among nodes
above 100 Pa is 2.38%, lower than its 3.09% overall prevalence, so it would
discard many labels without specifically isolating unreliable high-WSS values.

The extreme rule is narrow and source-quality based. It captures the suspect
global peaks in both case 0032 and case 0040 while retaining the spatially
coherent 330.81 Pa peak in case 0041. This is stronger than a WSS-value cap:
labels are excluded because their source mesh is degenerate, not because their
hemodynamic value is large.

## Training contract

Every development `quality_masks.zarr` contains:

- `target_valid`: primary boolean loss/evaluation mask.
- `volume_mesh_extreme_invalid`: reason-coded invalid nodes.
- `volume_mesh_severe_sensitivity`: non-primary sensitivity mask.

The data adapter attaches `target_valid_mask` to every graph patch. Training
vector, magnitude, direction, and temporal losses use valid nodes only.
Validation, regional metrics, cycle evaluation, and target normalization also
exclude primary-invalid nodes. Missing mask artifacts default to all-valid,
which preserves compatibility while locked-test preprocessing remains closed.

Existing checkpoints and reported metrics predate this policy. New experiments
must record the quality-mask manifest and recompute fold-specific normalization.

## Artifacts

- Cohort summary: `data/raw/stanford_vmr/canonical/volume_mesh_qc_summary.json`
- Published mask manifest:
  `data/raw/stanford_vmr/canonical/quality_mask_manifest.json`
- Per-case evidence: `diagnostics/volume_mesh_qc.json` and
  `diagnostics/volume_mesh_qc.vtp`
- Mask arrays: `quality_masks.zarr`

Reproduce the audit and mask publication with:

```powershell
.\.venv\Scripts\python.exe -m aorta_surrogate.data.volume_mesh_qc `
  --canonical-root data\raw\stanford_vmr\canonical `
  --project-root data\raw\stanford_vmr\projects

.\.venv\Scripts\python.exe -m aorta_surrogate.data.volume_mesh_qc `
  --canonical-root data\raw\stanford_vmr\canonical `
  --project-root data\raw\stanford_vmr\projects `
  --publish-existing-extreme-mask
```

## Next verification

Before opening the locked test set, freeze this exact rule and apply it using
geometry/mesh quality only. Run future model evaluations both with the primary
extreme mask and with the severe sensitivity mask to show that conclusions do
not depend on a small number of questionable mesh locations.

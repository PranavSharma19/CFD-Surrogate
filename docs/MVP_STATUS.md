# AAA surrogate MVP status

Updated: 2026-08-03

## Outcome

The Stanford VMR data now support an end-to-end, patient-separated development
pipeline: canonical surface/centerline/BC/WSS artifacts, spatial label audit,
fold-specific normalization, bounded patient-cycle training, periodic validation,
and interruption-safe checkpoints.

This is a working modeling pipeline, not an accepted surrogate. The current GAT
checkpoint is a baseline used to verify the contract and expose the next
bottlenecks.

## Data and split

- 15 completed Stanford abdominal-aorta CFD cases, 21 phases per case.
- 1,542,377 wall nodes in the canonical cohort.
- Fold zero trains on 8 patients and validates on 4 patients.
- Cases 0033, 0039, and 0042 are locked and were not accessed by training or
  fold-zero normalization.
- Split SHA-256: `3e621e5e7e3e9baeba93444a36357334f4938b333dcb13c98e3e55c405e3fd63`.

## WSS label audit

All 15 cases have `diagnostics/wss_outliers.vtp` overlays containing maximum WSS,
per-node cycle p99, phase of peak, TAWSS, OSI, RRT, threshold occupancy, boundary
status, and nearest centerline branch.

Values above 100 Pa form spatial clusters in the high-range cases rather than
only isolated points. The largest clusters were 442 nodes in case 0032, 363 in
0040, and 1,260 in 0041. These labels remain unmodified. Before using them in an
accepted model, inspect the overlays against branch outlets, mesh resolution, and
the original solver fields.

A source-verified audit of all 12 development cases is now complete. Original
Stanford arrays reproduced the canonical values within `3.01e-5 Pa` under the
registered conversion, every surface mapping was one-to-one, WSS was tangential,
and the high regions were not generally explained by open boundaries or poor
surface triangles. No development case remains unaudited.

The four automated flags are adjudicated. Case 0032's unreliable 584 Pa focal
peak and case 0040's 395 Pa peak are excluded by the frozen primary
extreme-volume mask. Cases 0041 and 0043 retain their original cycle endpoints:
registered closure variants changed derived metrics within the TAWSS, OSI, and
RRT sensitivity limits. Case 0044's barely-discontinuous 77.8 Pa peak remains in
the primary analysis but is explicitly covered by severe-mask sensitivity. All
12 cases are accepted for development under this policy; canonical targets are
unchanged.
See `docs/WSS_LABEL_AUDIT.md` and
`data/raw/stanford_vmr/canonical/wss_label_audit_adjudication.json`.

The frozen volume-mesh QC rules were then applied to all 12 development cases.
The broad severe rule marks 3.09% of nodes and remains sensitivity-only. The
extreme rule marks 143 of 1,198,219 nodes (0.0119%) and is now the primary
development loss/evaluation mask. It captures three of four values above 300 Pa,
including the suspect global peaks in 0032 and 0040, while retaining the coherent
330.81 Pa peak in 0041. Targets remain unchanged; masks are separate,
reason-coded artifacts. See `docs/VOLUME_MESH_QC.md`.

A frozen-checkpoint, three-fold mask ablation is also complete. Replacing the
primary extreme mask with the broad severe sensitivity mask excludes 2.41% of
the sampled nodes but changes pooled magnitude relative error by only +0.264
percentage points, supported-region macro error by +0.280 points, magnitude MAE
by -0.00031 Pa, and angular error by +0.155 degrees. No supported region changes
by one percentage point. The primary mask is retained; mesh-quality masking is
not the main cause of current model error. See `docs/MASK_ABLATION.md`.

## Local fold-zero result

The first 240-step, 4,096-node run proved the training path. A second run used
8,192-node paired-phase patches, then resumed from its step-400 validation
checkpoint for another 400 optimization steps.

| Metric | Result |
|---|---:|
| Sampled unseen-patient validation nodes | 163,840 |
| WSS vector MAE | 10.055 Pa |
| WSS vector RMSE | 18.110 Pa |
| WSS magnitude MAE | 7.967 Pa |
| WSS magnitude relative error | 74.58% |
| Zero-field magnitude relative error | 100% |
| Mean angular error above 0.5 Pa | 63.25 degrees |
| Peak RTX 3050 Ti VRAM | 0.428 GB |
| Continued-run training time | 153.9 seconds |

The model has a cross-patient learning signal, but it fails the planned 15%
relative-error gate. The locked test set must stay closed.

## Three-fold architecture comparison

All models used the same three patient folds, five validation phases, 4,096-node
connected geodesic patches, and 300 paired-phase optimization steps.

| Model | Pooled magnitude relative error | Median patient relative error | Mean angular error |
|---|---:|---:|---:|
| GATv2 | 82.02% | 79.51% | 69.24 degrees |
| PointNet | 83.33% | 83.76% | 55.84 degrees |
| Tangent-conditioned equivariant GNN | 66.95% | 70.23% | 27.49 degrees |

The equivariant architecture is the development winner, but it remains far from
the 15% research gate. Its centerline tangent is essential: the version without
that oriented vector achieved 87.31% magnitude error and 83.05 degrees on fold
zero, versus 73.47% and 37.06 degrees after adding it.

Complete-cycle evaluation on fixed validation patches produced pooled 64.36%
TAWSS relative error, 3.42 Pa TAWSS MAE, 0.0379 OSI MAE above the registered
0.1 Pa stability floor, and 75.99% RRT relative error over 46,553 valid nodes.
OSI is promising; WSS/TAWSS magnitude and RRT remain the limiting metrics.

## Immediate engineering sequence

The WATcloud V1 scientific contract and 0.361 GiB development-only runtime-data
bundle are now frozen. See `docs/WATCLOUD_EXPERIMENT_V1.md`.

1. Upload the completed, checksum-verified WATcloud bundles and run the registered
   RTX 4090 memory sweep. See `docs/WATCLOUD_INFRASTRUCTURE.md`.
2. Submit fold zero only after the interrupted/resumed sweep selects the largest
   registered patch at or below 22 GiB allocated VRAM.
3. Extend the existing connected-geodesic sampler with multiscale context while
   preserving branch and aneurysm coverage.
4. Extend development training for the equivariant model with validation-selected
   early stopping and full-cycle TAWSS/OSI/RRT evaluation.
5. Add multiscale/coarse geodesic edges and inlet/outlet distance scalars; compare
   against the current equivariant checkpoint without opening the locked set.
6. Only after architecture selection, evaluate the three locked Stanford cases
   once and decide whether the model is strong enough to justify independent CFD
   validation or needs the 1,090-case dataset first.

## Reproduction commands

```powershell
$env:PYTHONPATH='C:\Repositories\steve'

.\.venv\Scripts\python.exe -m aorta_surrogate.data.outlier_diagnostics `
  --canonical-root data\raw\stanford_vmr\canonical

python -m aorta_surrogate.training.fold_trainer `
  --canonical-root data\raw\stanford_vmr\canonical `
  --output-dir data\raw\stanford_vmr\canonical\experiments\fold0_run `
  --fold-index 0 --steps 800 --max-nodes 8192 --validation-interval 100
```

## Compute boundary

The laptop is appropriate for preprocessing, tests, overlays, short baseline
runs, and inference profiling. It should not run full-resolution production
training or transient CFD. The current cloud handoff targets `trpro-slurm2`: one
RTX 4090 with 24 GiB VRAM, 12 allocated CPU cores, 48 GiB RAM, and 30 GiB
job-local scratch. Persistent bundles and checkpoints live under the user's
`/mnt/wato-drive*` allocation.

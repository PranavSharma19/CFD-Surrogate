# WATcloud trainer implementation

Updated: 2026-08-03

## Outcome

The contract-driven trainer for `watcloud-preop-aaa-v1` is implemented and has
passed a local CUDA interruption/resume smoke test. It is ready for packaging,
staging, and the registered RTX 4090 memory sweep. A local smoke run is not a
scientific result and cannot select a production patch size.

## Contract enforcement

The trainer:

- Verifies the frozen contract and runtime-tree hashes before loading a case.
- Rejects any split disagreement or locked-patient overlap.
- Requires registered patch-size candidates and the RTX 4090 in production
  mode.
- Builds the frozen 192-channel, six-layer tangent-equivariant model.
- Enables layer checkpointing during training.
- Recomputes normalization from the eight fold-training patients only.
- Selects a supported semantic region uniformly and then a seed node uniformly
  within that region for every microbatch.
- Applies the frozen regional reduction: half global mean and half supported-
  region macro mean.
- Accumulates four paired-phase microbatches per optimizer step under BF16.
- Monitors five phases and performs fixed-patch complete-cycle validation.
- Selects `best_checkpoint.pt` using complete-cycle TAWSS relative error.
- Re-evaluates the selected checkpoint under primary and severe-sensitivity
  masks.
- Fails rather than overwriting an existing run unless a resume checkpoint is
  explicitly supplied.

## Complete-state checkpoints

Both best and latest checkpoints contain:

- Model parameters.
- AdamW state.
- Scheduler state.
- Mixed-precision scaler state.
- Completed optimizer step and early-stopping counters.
- Python, NumPy, local NumPy-generator, CPU Torch, and CUDA RNG states.
- Resolved experiment manifest and most recent cycle metrics.

Resume rejects a checkpoint when the contract hash, runtime tree, fold, patch
size, or precision differs from the requested run.

## Local smoke evidence

The frozen 192-by-6 model completed BF16 paired-phase training on the local RTX
3050 Ti using a deliberately non-production 1,024-node patch. The test stopped
after optimizer step one, saved complete state, restarted in a new process,
completed step two, evaluated all 21 phases, and published both mask policies.

An uninterrupted twin used the same seed and inputs. The reusable resume gate
reported:

- Step sequence: identical.
- Sampled-region sequence: identical.
- Maximum parameter difference: `5.90e-4` (tolerance `1e-3`).
- Complete-cycle TAWSS-relative-error difference: `2.56e-6` (tolerance `1e-4`).

The small nonzero difference is permitted CUDA atomic-reduction ordering across
process launches. It is many orders smaller than model error and demonstrates
state-equivalent continuation rather than a reset.

Evidence is stored in
`data/raw/stanford_vmr/canonical/experiments/watcloud_trainer_smoke_v2_resumed_fold0/resume_equivalence_report.json`.

## Interfaces

Production fold example after the memory sweep selects a registered patch size:

```bash
aorta-train-watcloud \
  --contract configs/watcloud_preop_v1_frozen.json \
  --canonical-root /tmp/aorta/canonical \
  --freeze-manifest /tmp/aorta/canonical/experiments/watcloud_preop_v1/freeze_manifest.json \
  --output-dir /persistent/aorta/watcloud-preop-aaa-v1/fold0 \
  --fold-index 0 \
  --patch-nodes SELECTED_REGISTERED_SIZE
```

Resume example:

```bash
aorta-train-watcloud \
  --contract configs/watcloud_preop_v1_frozen.json \
  --canonical-root /tmp/aorta/canonical \
  --freeze-manifest /tmp/aorta/canonical/experiments/watcloud_preop_v1/freeze_manifest.json \
  --output-dir /persistent/aorta/watcloud-preop-aaa-v1/fold0 \
  --fold-index 0 \
  --patch-nodes SELECTED_REGISTERED_SIZE \
  --resume /persistent/aorta/watcloud-preop-aaa-v1/fold0/latest_checkpoint.pt
```

## Remaining cloud work

The trainer, locked container definition, deterministic bundles, Slurm jobs, and
submission guards are complete. The data archive was verified member-by-member:
1,715 files, 387,308,958 uncompressed bytes, the frozen runtime-tree hash, and
zero locked-patient files all reproduce.

The remaining work requires the user's WATcloud session:

1. Upload the two bundles and SHA-256 sidecars to `/mnt/wato-drive*`.
2. Submit the container-build job and its dependent RTX 4090 memory sweep.
3. Review `patch_selection.json`, GPU utilization, resume evidence, and logs.
4. Submit fold zero; submit folds one and two only after fold-zero health review.

Exact commands and failure boundaries are in `docs/WATCLOUD_INFRASTRUCTURE.md`.

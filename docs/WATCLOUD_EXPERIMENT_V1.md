# WATcloud preoperative AAA experiment V1

Frozen: 2026-08-03

## Status

`watcloud-preop-aaa-v1` is frozen as a development-only experiment contract.
The contract SHA-256 is
`2c04fec795409ec6d8f768aac4bd68f276e40acf76e157d44d31bb4e8ee75cf4`.

The frozen runtime-data manifest contains 1,715 files totaling 0.361 GiB. Its
tree SHA-256 is
`43000bd90a93afad3fefc41b5cef8cd9b3042a341ea59485138ee44fc6b8f17a`.
It contains all 12 development patients, no locked-patient files, and no
original Stanford source archives.

This freezes the scientific and evaluation decisions. The implementation code
and WATcloud environment will receive separate hashes after the trainer changes
are complete; they must satisfy this contract and cannot redefine it.

## Frozen choices

- Patient-level three-fold split exactly as recorded in `patient_split.json`.
- Locked cases `0033`, `0039`, and `0042` are not staged, mounted, normalized
  with, inspected, or evaluated.
- Tangent-conditioned `EquivariantSurfaceGNN`, 192 hidden channels and six
  message layers, without the experimental multiscale edges.
- Complete 21-phase transient WSS-vector target.
- Primary extreme-volume quality mask; severe-volume mask as sensitivity only.
- Five supported sampling strata: aorta, renal, mesenteric,
  celiac/hepatic/splenic, and iliac.
- Optional explicit-aneurysm nodes remain in the global loss and are reported
  separately, but do not become an equal stratum with the supported regions.
- Regional loss is `0.5 * global node mean + 0.5 * supported-region macro mean`.
- AdamW at `3e-4`, weight decay `1e-5`, BF16 mixed precision, accumulation four,
  gradient clipping at one, and cosine annealing.
- At most 20,000 steps per fold. Five-phase monitoring occurs every 250 steps;
  complete-cycle validation every 1,000 steps selects checkpoints using TAWSS
  relative error.
- Early stopping starts after 4,000 steps with patience of eight complete-cycle
  validations.
- Fold seeds are `20260802`, `20260803`, and `20260804`.

## Registered 4090 adaptation

Patch size is the only intentionally unresolved production value. It is not
chosen from validation accuracy. On `trpro-slurm2`, test 8,192, 12,288, 16,384,
20,480, and 24,576 nodes in order. Select the largest size that completes paired
forward/backward, validation, checkpoint save, and full-state resume while peak
allocated VRAM remains at or below 22.0 GiB. That one size applies to every
fold.

This is a pre-registered hardware feasibility rule, not a hyperparameter search.

## Change control

Permitted under V1:

- Select patch size using the registered memory rule.
- Fall back from BF16 to FP16 only after a recorded operator failure.
- Reduce data-loader workers after a recorded memory or I/O failure.
- Resume from a complete checkpoint.

A new experiment ID is required to change the split, staged patients,
architecture, loss, masks, optimizer, learning rate, seeds, selection metric, or
to inspect locked data. All fallbacks must be written into the resolved run
manifest.

## Launch gates

The production job cannot launch until:

1. Contract and staged-data hashes reproduce.
2. The staged case set equals the 12 development cases exactly and locked case
   directories are absent.
3. The WATcloud environment passes all tests.
4. The memory sweep passes forward, backward, validation, save, and resume.
5. VRAM is no more than 22.0 GiB and every numerical output is finite.
6. Fold normalization proves validation and locked-patient exclusion.
7. An intentional interruption restores optimizer, scheduler, scaler, step, and
   RNG state.

## Artifacts and reproduction

- Frozen contract: `configs/watcloud_preop_v1_frozen.json`.
- Machine-readable data freeze:
  `data/raw/stanford_vmr/canonical/experiments/watcloud_preop_v1/freeze_manifest.json`.

Reproduce the freeze before staging:

```powershell
aorta-freeze-experiment `
  --contract configs\watcloud_preop_v1_frozen.json `
  --canonical-root data\raw\stanford_vmr\canonical `
  --output-dir data\raw\stanford_vmr\canonical\experiments\watcloud_preop_v1
```

The trainer now conforms to this contract: regional seed selection and loss
aggregation, configurable 192-by-6 model construction, accumulation and layer
checkpointing, complete-state resume, and complete-cycle TAWSS checkpoint
selection are implemented. See `docs/WATCLOUD_TRAINER.md`. The next step is the
container and SLURM packaging used to run the registered RTX 4090 sweep.

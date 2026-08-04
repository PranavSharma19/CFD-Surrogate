# CFD Surrogate

Research software for fast prediction of aortic wall-shear-stress fields and derived hemodynamic metrics. The current MVP focuses on transient, preoperative abdominal aortic aneurysm geometries using completed Stanford CFD cases. A supplied-geometry EVAR comparison is planned after the preoperative model passes its offline gates.

## Current state

- Legacy synthetic bifurcation models remain in the separate `Toralis-Labs/Bifurcation` repository and are not duplicated here.
- `aorta_surrogate/` contains the patient-level aortic data, model, training, evaluation, and cloud-training code.
- The Stanford development experiment is frozen in `configs/watcloud_preop_v1_frozen.json`.
- WATcloud packaging targets the RTX 4090 on `trpro-slurm2` using a digest-pinned PyTorch container and Slurm.
- AWS packaging targets one `g6e.2xlarge` L40S using the same scientific protocol and a separately frozen hardware contract.
- The three locked Stanford patients are excluded from the upload bundle and remain unopened for final evaluation.

This repository intentionally excludes patient/CFD data, generated bundles, checkpoints, credentials, and the separate legacy bifurcation repository.

## Local verification

```powershell
python -m pytest
```

## WATcloud workflow

1. Generate and verify the development-only data and source bundles locally.
2. Upload the bundles and SHA-256 sidecars to persistent `/mnt/wato-drive*` storage.
3. Submit the image-build job and dependent RTX 4090 memory sweep.
4. Review the registered patch selection before submitting fold zero.
5. Submit folds one and two only after fold-zero health review.

See:

- `docs/WATCLOUD_EXPERIMENT_V1.md` for the frozen scientific contract.
- `docs/WATCLOUD_TRAINER.md` for trainer behavior and resume evidence.
- `docs/WATCLOUD_INFRASTRUCTURE.md` for upload and submission commands.
- `docs/AWS_INFRASTRUCTURE.md` for private-S3, EBS, L40S sweep, and fold commands.
- `docs/MVP_STATUS.md` for current model results and remaining work.

## Intended use

Research use only. The software does not provide diagnosis, rupture prediction, device selection, or clinical treatment recommendations.

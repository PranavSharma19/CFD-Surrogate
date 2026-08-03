# WATcloud infrastructure runbook

This package runs the frozen `watcloud-preop-aaa-v1` experiment on the RTX 4090 in `trpro-slurm2`. It does not stage the three locked patients and does not submit full training until the registered VRAM sweep has selected a patch size.

## What is packaged

- `watcloud-preop-aaa-v1-data.tar.gz`: the 12 frozen development cases, their runtime artifacts, and the freeze manifest.
- `watcloud-preop-aaa-v1-source.tar.gz`: trainer source, contract, tests, and WATcloud infrastructure.
- SHA-256 sidecars for both bundles and `bundle_manifest.json`.
- A digest-pinned PyTorch/CUDA Docker build with an exact Python dependency lock.
- Slurm jobs for image construction, the interrupted/resumed 4090 memory sweep, and three patient-level folds.

The Docker image is built once in Slurm and saved to WATO Drive because WATcloud's rootless Docker data directory lives in temporary job storage. Every later job reloads that immutable image archive.

## 1. Create bundles locally

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m aorta_surrogate.training.package_watcloud `
  --repository-root . `
  --canonical-root data\raw\stanford_vmr\canonical `
  --freeze-manifest data\raw\stanford_vmr\canonical\experiments\watcloud_preop_v1\freeze_manifest.json `
  --output-dir data\raw\stanford_vmr\exports\watcloud_preop_v1
```

Verify every member directly from the compressed archive without extracting a second copy:

```powershell
.\.venv\Scripts\python.exe -m aorta_surrogate.training.verify_watcloud_bundle `
  --data-bundle data\raw\stanford_vmr\exports\watcloud_preop_v1\watcloud-preop-aaa-v1-data.tar.gz
```

## 2. Upload to the WATcloud login node

Choose a persistent root that already exists under `/mnt/wato-drive*`; this is intentionally not guessed by the scripts. On the login node:

```bash
export WATCLOUD_PERSIST_ROOT=/mnt/wato-drive/REPLACE_WITH_YOUR_DIRECTORY
export EXPERIMENT_ROOT="$WATCLOUD_PERSIST_ROOT/watcloud-preop-aaa-v1"
mkdir -p "$EXPERIMENT_ROOT/bundles"
```

Upload the two archives, both `.sha256` files, and `bundle_manifest.json` into `$EXPERIMENT_ROOT/bundles`. Extract the small source archive into a working directory on the login node:

```bash
mkdir -p ~/aorta-watcloud
tar -xzf "$EXPERIMENT_ROOT/bundles/watcloud-preop-aaa-v1-source.tar.gz" -C ~/aorta-watcloud
```

## 3. Submit image build and memory sweep

```bash
cd ~/aorta-watcloud/infra/watcloud
export WATCLOUD_PERSIST_ROOT=/mnt/wato-drive/REPLACE_WITH_YOUR_DIRECTORY
bash submit_pipeline.sh
```

The submission helper checks that `trpro-slurm2` currently advertises `gpu:rtx_4090`, validates all uploaded checksums, submits the CPU image build, and makes the GPU sweep depend on a successful build. It deliberately does not submit full training.

Monitor it with:

```bash
squeue -u "$(whoami)"
tail -f logs/*-aorta-image.out
tail -f logs/*-aorta-vram.out
sacct --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Review the registered output:

```bash
python3 -m json.tool "$EXPERIMENT_ROOT/runs/memory_sweep/patch_selection.json"
```

Each candidate must complete step 1, save, resume in a new container process, complete step 2, and remain at or below 22 GiB allocated VRAM. The selector chooses the largest passing registered candidate.

## 4. Submit training

Start with fold 0:

```bash
bash submit_training.sh 0
```

After fold 0 demonstrates healthy throughput, loss, checkpointing, and complete-cycle validation, submit the remaining folds serially:

```bash
bash submit_training.sh '1-2%1'
```

The job automatically resumes from `latest_checkpoint.pt` when it exists. All checkpoints and reports are written directly to WATO Drive under `$EXPERIMENT_ROOT/runs/folds`; the canonical dataset is copied to `/tmp` for fast reads and is mounted read-only in the container.

## Failure boundaries

- A missing or altered file fails the frozen-runtime hash verification before training.
- A staged locked patient fails verification.
- A wrong node, missing 4090 GRES, absent patch selection, or non-WATO persistent root fails closed.
- An out-of-memory sweep candidate is recorded and the next smaller/larger registered candidate is still evaluated; no unregistered patch size is introduced.
- A repeated memory sweep refuses to overwrite an existing sweep. Move the old directory to a versioned archive explicitly before resubmission.
- Full training never falls back to CPU. BF16-to-FP16 fallback requires the documented contract exception and an explicit trainer argument.

The package assumes WATcloud's documented rootless Docker and NVIDIA Container Toolkit configuration. Current resource names must always be checked with `scontrol show node trpro-slurm2` before submission.

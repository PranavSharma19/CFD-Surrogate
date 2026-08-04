# AWS infrastructure runbook

This runbook executes the development-only Stanford pilot on one AWS `g6e.2xlarge` with an NVIDIA L40S. It is a separate hardware experiment, `aws-preop-aaa-v1`, derived from the WATcloud scientific contract without changing patients, splits, model, losses, optimization, masks, metrics, or the initial 22 GiB VRAM envelope.

## Required AWS resources

- An individual AWS identity; do not share root credentials or access keys.
- Accelerated-instance quota for `g6e.2xlarge` in the selected region.
- Systems Manager Session Manager access to the EC2 instance.
- A private, encrypted, versioned S3 prefix.
- An EC2 instance role scoped to that prefix plus SSM managed-instance access.
- An encrypted persistent EBS volume mounted under `/mnt`, with 200 GiB `gp3` recommended.
- A current AWS GPU/Deep Learning Ubuntu AMI whose NVIDIA driver supports the pinned CUDA 12.8 container.

Use On-Demand for preparation, the memory sweep, and fold zero. Spot remains disabled until interruption-triggered checkpoint synchronization passes a separate deliberate test.

## Private S3 layout

```text
s3://YOUR_PRIVATE_BUCKET/aorta-v1/
├── bundles/
│   ├── watcloud-preop-aaa-v1-data.tar.gz
│   └── watcloud-preop-aaa-v1-data.tar.gz.sha256
├── runs/
└── state/
```

The bundle name retains `watcloud` because it is the already-verified, byte-identical development dataset. The AWS freeze manifest is derived after staging and records a new contract hash over the same runtime-tree hash.

## Prepare the instance

Clone the public source repository onto the EC2 instance:

```bash
git clone https://github.com/PranavSharma19/CFD-Surrogate.git
cd CFD-Surrogate
```

Confirm the EBS mount, available space, GPU, Docker, and instance role:

```bash
df -h /mnt/aorta
nvidia-smi
docker info
aws sts get-caller-identity
aws s3 ls s3://YOUR_PRIVATE_BUCKET/aorta-v1/
```

Set the two required variables:

```bash
export AWS_PROJECT_ROOT=/mnt/aorta
export AWS_S3_URI=s3://YOUR_PRIVATE_BUCKET/aorta-v1
```

Prepare and validate everything:

```bash
bash infra/aws/prepare_instance.sh
```

The script downloads the private bundle, checks its hard-coded SHA-256 and sidecar, stages it without overwriting an existing canonical tree, builds the digest-pinned image, runs the test suite, verifies every archive member, derives the AWS freeze manifest, verifies the staged tree again, records the GPU/image environment, and synchronizes state to S3.

## Run the registered memory sweep

Run it detached from the SSM session:

```bash
nohup bash infra/aws/memory_sweep.sh > /mnt/aorta/logs/memory-sweep.out 2>&1 &
echo $!
```

Monitor:

```bash
tail -f /mnt/aorta/logs/memory-sweep.out
nvidia-smi
```

Review the selection only after the process exits successfully:

```bash
python3 -m json.tool /mnt/aorta/runs/memory_sweep/patch_selection.json
```

Each candidate must complete step one, save, exit, resume in a separate container, complete step two, evaluate, and remain at or below 22 GiB allocated VRAM. Existing sweep output is never overwritten automatically.

## Train folds

Start with fold zero:

```bash
nohup bash infra/aws/train_fold.sh 0 > /mnt/aorta/logs/fold-0.out 2>&1 &
```

The launcher synchronizes runs and state to S3 every five minutes and on exit. If an incomplete run has `latest_checkpoint.pt`, launching the same fold resumes it. A completed run containing `result.json` is never overwritten.

Only after fold zero shows healthy throughput, finite loss, correct validation/checkpoint cadence, and successful S3 synchronization should folds one and two be launched:

```bash
nohup bash infra/aws/train_fold.sh 1 > /mnt/aorta/logs/fold-1.out 2>&1 &
nohup bash infra/aws/train_fold.sh 2 > /mnt/aorta/logs/fold-2.out 2>&1 &
```

Run only one fold at a time on the single-GPU instance.

## Shutdown gate

Before stopping or terminating the instance:

```bash
bash infra/aws/sync_results.sh
aws s3 ls --recursive "$AWS_S3_URI/runs/"
```

Confirm the latest checkpoint, result, metrics, runtime manifest, and logs are present in S3. Stopping EC2 ends compute charges but EBS and S3 continue to incur storage charges. Terminate the instance only after confirming whether its EBS volume is configured for retention.

## Failure boundaries

- Non-L40S GPUs fail before any training command.
- A data archive whose actual SHA differs from the registered hash fails even when supplied with a matching malicious sidecar.
- Any locked case or altered runtime file fails manifest verification.
- AWS receives a new contract/freeze hash; it never impersonates the WATcloud experiment.
- Existing canonical, sweep, or completed-fold outputs are not overwritten.
- Spot is not authorized for the first fold.
- Missing S3, Docker, CUDA, or IAM access fails during preparation rather than after paid training begins.

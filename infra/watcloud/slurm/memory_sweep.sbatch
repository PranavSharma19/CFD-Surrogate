#!/usr/bin/env bash
#SBATCH --job-name=aorta-vram
#SBATCH --partition=compute
#SBATCH --nodelist=trpro-slurm2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx_4090:1,tmpdisk:30720
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j-%x.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
require_persistent_root
start_docker
load_training_image
stage_canonical_data

test "$(hostname -s)" = "trpro-slurm2" || { echo "Wrong host: $(hostname -s)" >&2; exit 5; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

SWEEP_ROOT="${RUN_ROOT}/memory_sweep"
if [[ -e "${SWEEP_ROOT}/patch_selection.json" ]] || compgen -G "${SWEEP_ROOT}/patch_*" >/dev/null; then
  echo "Memory-sweep output already exists; move it to a versioned archive before starting a new sweep: ${SWEEP_ROOT}" >&2
  exit 8
fi
mkdir -p "${SWEEP_ROOT}"
FREEZE=/tmp/aorta/canonical/experiments/watcloud_preop_v1/freeze_manifest.json
mapfile -t DOCKER_ARGS < <(docker_base_args)

docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.verify_staged_runtime \
  --canonical-root /tmp/aorta/canonical --freeze-manifest "${FREEZE}"

for PATCH_NODES in 8192 12288 16384 20480 24576; do
  OUTPUT="/runs/memory_sweep/patch_${PATCH_NODES}"
  HOST_OUTPUT="${SWEEP_ROOT}/patch_${PATCH_NODES}"
  mkdir -p "${HOST_OUTPUT}"
  set +e
  docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.watcloud_trainer \
    --contract /opt/aorta/configs/watcloud_preop_v1_frozen.json \
    --canonical-root /tmp/aorta/canonical --freeze-manifest "${FREEZE}" \
    --output-dir "${OUTPUT}" --fold-index 0 --patch-nodes "${PATCH_NODES}" \
    --smoke-steps 2 --stop-after-step 1
  FIRST_STATUS=$?
  if [[ ${FIRST_STATUS} -eq 0 ]]; then
    docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.watcloud_trainer \
      --contract /opt/aorta/configs/watcloud_preop_v1_frozen.json \
      --canonical-root /tmp/aorta/canonical --freeze-manifest "${FREEZE}" \
      --output-dir "${OUTPUT}" --fold-index 0 --patch-nodes "${PATCH_NODES}" \
      --smoke-steps 2 --resume "${OUTPUT}/latest_checkpoint.pt"
    SECOND_STATUS=$?
  else
    SECOND_STATUS=99
  fi
  set -e
  if [[ ${FIRST_STATUS} -ne 0 || ${SECOND_STATUS} -ne 0 ]]; then
    printf '{"reason":"trainer command failed","first_exit_code":%d,"resume_exit_code":%d}\n' \
      "${FIRST_STATUS}" "${SECOND_STATUS}" > "${HOST_OUTPUT}/failure.json"
  fi
done

docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.select_watcloud_patch \
  --contract /opt/aorta/configs/watcloud_preop_v1_frozen.json \
  --sweep-root /runs/memory_sweep
echo "Patch selection: ${SWEEP_ROOT}/patch_selection.json"

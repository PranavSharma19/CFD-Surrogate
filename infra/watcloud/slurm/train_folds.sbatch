#!/usr/bin/env bash
#SBATCH --job-name=aorta-fold
#SBATCH --partition=compute_dense
#SBATCH --nodelist=trpro-slurm2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx_4090:1,tmpdisk:30720
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%A-%a-%x.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
require_persistent_root
start_docker
load_training_image
stage_canonical_data

test "$(hostname -s)" = "trpro-slurm2" || { echo "Wrong host: $(hostname -s)" >&2; exit 5; }
SELECTION="${RUN_ROOT}/memory_sweep/patch_selection.json"
test -f "${SELECTION}" || { echo "Missing registered patch selection: ${SELECTION}" >&2; exit 6; }
PATCH_NODES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_patch_nodes"])' "${SELECTION}")"
FOLD_INDEX="${SLURM_ARRAY_TASK_ID:?This job must be submitted as an array.}"
OUTPUT="/runs/folds/fold_${FOLD_INDEX}"
HOST_OUTPUT="${RUN_ROOT}/folds/fold_${FOLD_INDEX}"
mkdir -p "${HOST_OUTPUT}"
FREEZE=/tmp/aorta/canonical/experiments/watcloud_preop_v1/freeze_manifest.json
mapfile -t DOCKER_ARGS < <(docker_base_args)

RESUME_ARGS=()
if [[ -f "${HOST_OUTPUT}/latest_checkpoint.pt" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT}/latest_checkpoint.pt")
fi

docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.watcloud_trainer \
  --contract /opt/aorta/configs/watcloud_preop_v1_frozen.json \
  --canonical-root /tmp/aorta/canonical --freeze-manifest "${FREEZE}" \
  --output-dir "${OUTPUT}" --fold-index "${FOLD_INDEX}" --patch-nodes "${PATCH_NODES}" \
  "${RESUME_ARGS[@]}"

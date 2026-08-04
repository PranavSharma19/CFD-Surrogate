#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_aws_environment
require_commands
verify_prepared_runtime

FOLD_INDEX="${1:?Usage: train_fold.sh FOLD_INDEX, where FOLD_INDEX is 0, 1, or 2}"
[[ "${FOLD_INDEX}" =~ ^[0-2]$ ]] || { echo "Fold index must be 0, 1, or 2" >&2; exit 9; }
SELECTION="${RUN_ROOT}/memory_sweep/patch_selection.json"
test -f "${SELECTION}" || { echo "Missing registered patch selection: ${SELECTION}" >&2; exit 9; }
PATCH_NODES="$(python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="patch_selected"; print(x["selected_patch_nodes"])' "${SELECTION}")"

HOST_OUTPUT="${RUN_ROOT}/folds/fold_${FOLD_INDEX}"
OUTPUT="/runs/folds/fold_${FOLD_INDEX}"
mkdir -p "${HOST_OUTPUT}"
if [[ -f "${HOST_OUTPUT}/result.json" ]]; then
  echo "Fold ${FOLD_INDEX} already has result.json; refusing to overwrite or resume a completed run." >&2
  exit 9
fi
mapfile -t DOCKER_ARGS < <(docker_base_args)
RESUME_ARGS=()
if [[ -f "${HOST_OUTPUT}/latest_checkpoint.pt" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT}/latest_checkpoint.pt")
fi

SYNC_INTERVAL_SECONDS="${AWS_SYNC_INTERVAL_SECONDS:-300}"
periodic_sync() {
  while sleep "${SYNC_INTERVAL_SECONDS}"; do
    sync_results || true
  done
}
periodic_sync &
SYNC_PID=$!
cleanup() {
  local status=$?
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  sync_results || true
  exit "${status}"
}
trap cleanup EXIT

docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.watcloud_trainer \
  --contract /opt/aorta/configs/aws_preop_v1_frozen.json \
  --canonical-root /data/canonical --freeze-manifest /state/aws_freeze_manifest.json \
  --output-dir "${OUTPUT}" --fold-index "${FOLD_INDEX}" --patch-nodes "${PATCH_NODES}" \
  "${RESUME_ARGS[@]}"

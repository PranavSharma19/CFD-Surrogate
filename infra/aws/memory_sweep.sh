#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_aws_environment
require_commands
verify_prepared_runtime

SWEEP_ROOT="${RUN_ROOT}/memory_sweep"
if [[ -e "${SWEEP_ROOT}/patch_selection.json" ]] || compgen -G "${SWEEP_ROOT}/patch_*" >/dev/null; then
  echo "Memory-sweep output already exists; archive it explicitly before a new sweep: ${SWEEP_ROOT}" >&2
  exit 8
fi
mkdir -p "${SWEEP_ROOT}"

on_exit() {
  local status=$?
  sync_results || true
  exit "${status}"
}
trap on_exit EXIT

mapfile -t DOCKER_ARGS < <(docker_base_args)
docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.verify_staged_runtime \
  --canonical-root /data/canonical --freeze-manifest /state/aws_freeze_manifest.json

for PATCH_NODES in 8192 12288 16384 20480 24576; do
  OUTPUT="/runs/memory_sweep/patch_${PATCH_NODES}"
  HOST_OUTPUT="${SWEEP_ROOT}/patch_${PATCH_NODES}"
  mkdir -p "${HOST_OUTPUT}"
  set +e
  docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.watcloud_trainer \
    --contract /opt/aorta/configs/aws_preop_v1_frozen.json \
    --canonical-root /data/canonical --freeze-manifest /state/aws_freeze_manifest.json \
    --output-dir "${OUTPUT}" --fold-index 0 --patch-nodes "${PATCH_NODES}" \
    --smoke-steps 2 --stop-after-step 1
  FIRST_STATUS=$?
  if [[ ${FIRST_STATUS} -eq 0 ]]; then
    docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.watcloud_trainer \
      --contract /opt/aorta/configs/aws_preop_v1_frozen.json \
      --canonical-root /data/canonical --freeze-manifest /state/aws_freeze_manifest.json \
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
  sync_results
done

docker run "${DOCKER_ARGS[@]}" python -m aorta_surrogate.training.select_watcloud_patch \
  --contract /opt/aorta/configs/aws_preop_v1_frozen.json \
  --sweep-root /runs/memory_sweep
echo "AWS patch selection: ${SWEEP_ROOT}/patch_selection.json"

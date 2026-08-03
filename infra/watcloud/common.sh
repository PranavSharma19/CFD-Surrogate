#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="watcloud-preop-aaa-v1"
IMAGE_TAG="aorta-surrogate:watcloud-preop-v1"
DATA_BUNDLE_NAME="watcloud-preop-aaa-v1-data.tar.gz"
SOURCE_BUNDLE_NAME="watcloud-preop-aaa-v1-source.tar.gz"
IMAGE_ARCHIVE_NAME="watcloud-preop-aaa-v1-image.tar.gz"

require_persistent_root() {
  : "${WATCLOUD_PERSIST_ROOT:?Set WATCLOUD_PERSIST_ROOT to a directory under /mnt/wato-drive*.}"
  case "${WATCLOUD_PERSIST_ROOT}" in
    /mnt/wato-drive*) ;;
    *) echo "WATCLOUD_PERSIST_ROOT must be under /mnt/wato-drive*: ${WATCLOUD_PERSIST_ROOT}" >&2; exit 2 ;;
  esac
  export EXPERIMENT_ROOT="${WATCLOUD_PERSIST_ROOT}/${EXPERIMENT_ID}"
  export BUNDLE_ROOT="${EXPERIMENT_ROOT}/bundles"
  export IMAGE_ROOT="${EXPERIMENT_ROOT}/images"
  export RUN_ROOT="${EXPERIMENT_ROOT}/runs"
  export LOG_ROOT="${EXPERIMENT_ROOT}/logs"
  mkdir -p "${BUNDLE_ROOT}" "${IMAGE_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"
}

verify_sidecar() {
  local path="$1"
  test -f "${path}" || { echo "Missing artifact: ${path}" >&2; exit 3; }
  test -f "${path}.sha256" || { echo "Missing checksum: ${path}.sha256" >&2; exit 3; }
  (cd "$(dirname "${path}")" && sha256sum --check "$(basename "${path}").sha256")
}

start_docker() {
  command -v slurm-start-dockerd.sh >/dev/null || { echo "slurm-start-dockerd.sh is unavailable" >&2; exit 4; }
  slurm-start-dockerd.sh
  docker info >/dev/null
}

load_training_image() {
  local archive="${IMAGE_ROOT}/${IMAGE_ARCHIVE_NAME}"
  verify_sidecar "${archive}"
  gzip --decompress --stdout "${archive}" | docker load
  docker image inspect "${IMAGE_TAG}" >/dev/null
}

stage_canonical_data() {
  local archive="${BUNDLE_ROOT}/${DATA_BUNDLE_NAME}"
  verify_sidecar "${archive}"
  rm -rf /tmp/aorta
  mkdir -p /tmp/aorta
  tar --extract --gzip --file "${archive}" --directory /tmp/aorta
  test -f /tmp/aorta/canonical/experiments/watcloud_preop_v1/freeze_manifest.json
}

docker_base_args() {
  printf '%s\n' \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --volume /tmp/aorta/canonical:/tmp/aorta/canonical:ro \
    --volume "${RUN_ROOT}:/runs" \
    "${IMAGE_TAG}"
}

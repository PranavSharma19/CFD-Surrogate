#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="aws-preop-aaa-v1"
IMAGE_TAG="aorta-surrogate:aws-preop-v1"
DATA_BUNDLE_NAME="watcloud-preop-aaa-v1-data.tar.gz"
EXPECTED_DATA_SHA256="7431ad11ab26706f90ddcd6f40be3b7b417cfd86c93e885fe3b34e3a1f538dcf"
RUNTIME_TREE_SHA256="43000bd90a93afad3fefc41b5cef8cd9b3042a341ea59485138ee44fc6b8f17a"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REPOSITORY_ROOT="${AWS_REPOSITORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
AWS_CONTRACT="${AWS_REPOSITORY_ROOT}/configs/aws_preop_v1_frozen.json"

require_aws_environment() {
  : "${AWS_PROJECT_ROOT:?Set AWS_PROJECT_ROOT to the encrypted EBS mount, for example /mnt/aorta.}"
  : "${AWS_S3_URI:?Set AWS_S3_URI to the private project prefix, for example s3://bucket/aorta-v1.}"
  case "${AWS_PROJECT_ROOT}" in
    /mnt/*) ;;
    *) echo "AWS_PROJECT_ROOT must be an absolute path under /mnt: ${AWS_PROJECT_ROOT}" >&2; exit 2 ;;
  esac
  case "${AWS_S3_URI}" in
    s3://*) ;;
    *) echo "AWS_S3_URI must start with s3://" >&2; exit 2 ;;
  esac
  AWS_S3_URI="${AWS_S3_URI%/}"
  export AWS_S3_URI
  export BUNDLE_ROOT="${AWS_PROJECT_ROOT}/bundles"
  export CANONICAL_ROOT="${AWS_PROJECT_ROOT}/canonical"
  export STATE_ROOT="${AWS_PROJECT_ROOT}/state"
  export RUN_ROOT="${AWS_PROJECT_ROOT}/runs"
  export LOG_ROOT="${AWS_PROJECT_ROOT}/logs"
  export AWS_FREEZE_MANIFEST="${STATE_ROOT}/aws_freeze_manifest.json"
  mkdir -p "${BUNDLE_ROOT}" "${STATE_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"
}

require_commands() {
  local missing=0
  for command_name in aws docker nvidia-smi python3 sha256sum tar; do
    if ! command -v "${command_name}" >/dev/null; then
      echo "Missing required command: ${command_name}" >&2
      missing=1
    fi
  done
  [[ ${missing} -eq 0 ]] || exit 3
}

verify_data_bundle() {
  local bundle="${BUNDLE_ROOT}/${DATA_BUNDLE_NAME}"
  test -f "${bundle}" || { echo "Missing data bundle: ${bundle}" >&2; exit 4; }
  test -f "${bundle}.sha256" || { echo "Missing data checksum sidecar" >&2; exit 4; }
  local actual
  actual="$(sha256sum "${bundle}" | awk '{print $1}')"
  [[ "${actual}" == "${EXPECTED_DATA_SHA256}" ]] || {
    echo "Data bundle does not match the registered SHA-256: ${actual}" >&2
    exit 4
  }
  (cd "${BUNDLE_ROOT}" && sha256sum --check "${DATA_BUNDLE_NAME}.sha256")
}

verify_l40s() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ "$(printf '%s\n' "${gpu_names}" | sed '/^$/d' | wc -l)" -eq 1 ]] || {
    echo "AWS V1 requires exactly one visible GPU; got: ${gpu_names}" >&2
    exit 5
  }
  grep -qi 'L40S' <<<"${gpu_names}" || {
    echo "AWS V1 requires the registered NVIDIA L40S; got: ${gpu_names}" >&2
    exit 5
  }
}

verify_prepared_runtime() {
  test -f "${AWS_CONTRACT}" || { echo "Missing AWS contract: ${AWS_CONTRACT}" >&2; exit 6; }
  test -f "${AWS_FREEZE_MANIFEST}" || { echo "Missing derived freeze manifest: ${AWS_FREEZE_MANIFEST}" >&2; exit 6; }
  test -d "${CANONICAL_ROOT}" || { echo "Missing canonical runtime: ${CANONICAL_ROOT}" >&2; exit 6; }
  docker image inspect "${IMAGE_TAG}" >/dev/null
  verify_l40s
}

docker_base_args() {
  printf '%s\n' \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --volume "${CANONICAL_ROOT}:/data/canonical:ro" \
    --volume "${STATE_ROOT}:/state:ro" \
    --volume "${RUN_ROOT}:/runs" \
    "${IMAGE_TAG}"
}

sync_results() {
  aws s3 sync "${RUN_ROOT}/" "${AWS_S3_URI}/runs/" --only-show-errors
  aws s3 sync "${STATE_ROOT}/" "${AWS_S3_URI}/state/" --only-show-errors
}

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_aws_environment
require_commands
verify_l40s
docker info >/dev/null

aws s3 sync "${AWS_S3_URI}/bundles/" "${BUNDLE_ROOT}/" --only-show-errors
verify_data_bundle

if [[ -d "${CANONICAL_ROOT}" ]]; then
  echo "Canonical runtime already exists; it will be verified without replacement."
else
  STAGING_ROOT="${AWS_PROJECT_ROOT}/canonical-staging-$(date +%Y%m%dT%H%M%S)"
  mkdir -p "${STAGING_ROOT}"
  tar --extract --gzip --file "${BUNDLE_ROOT}/${DATA_BUNDLE_NAME}" --directory "${STAGING_ROOT}"
  test -d "${STAGING_ROOT}/canonical" || { echo "Bundle did not contain canonical/" >&2; exit 7; }
  mv "${STAGING_ROOT}/canonical" "${CANONICAL_ROOT}"
  rmdir "${STAGING_ROOT}"
fi

CONTRACT_SHA256="$(sha256sum "${AWS_CONTRACT}" | awk '{print $1}')"
docker build --pull=false \
  --build-arg "CONTRACT_SHA256=${CONTRACT_SHA256}" \
  --tag "${IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  "${AWS_REPOSITORY_ROOT}"

docker run --rm "${IMAGE_TAG}" python -m pytest
docker run --rm \
  --volume "${BUNDLE_ROOT}:/bundles:ro" \
  "${IMAGE_TAG}" \
  python -m aorta_surrogate.training.verify_watcloud_bundle \
  --data-bundle "/bundles/${DATA_BUNDLE_NAME}"

BASE_FREEZE="${CANONICAL_ROOT}/experiments/watcloud_preop_v1/freeze_manifest.json"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "${CANONICAL_ROOT}:/data/canonical:ro" \
  --volume "${STATE_ROOT}:/state" \
  "${IMAGE_TAG}" \
  python -m aorta_surrogate.training.derive_freeze_manifest \
  --canonical-root /data/canonical \
  --base-manifest /data/canonical/experiments/watcloud_preop_v1/freeze_manifest.json \
  --contract /opt/aorta/configs/aws_preop_v1_frozen.json \
  --output /state/aws_freeze_manifest.json

docker run --rm \
  --volume "${CANONICAL_ROOT}:/data/canonical:ro" \
  --volume "${STATE_ROOT}:/state:ro" \
  "${IMAGE_TAG}" \
  python -m aorta_surrogate.training.verify_staged_runtime \
  --canonical-root /data/canonical \
  --freeze-manifest /state/aws_freeze_manifest.json

docker image inspect "${IMAGE_TAG}" > "${STATE_ROOT}/aws_image_inspect.json"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "${STATE_ROOT}/aws_gpu_preflight.csv"
sync_results
echo "AWS runtime prepared and verified under ${AWS_PROJECT_ROOT}."

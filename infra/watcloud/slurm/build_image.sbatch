#!/usr/bin/env bash
#SBATCH --job-name=aorta-image
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=tmpdisk:30720
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j-%x.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
require_persistent_root
start_docker

SOURCE_ARCHIVE="${BUNDLE_ROOT}/${SOURCE_BUNDLE_NAME}"
verify_sidecar "${SOURCE_ARCHIVE}"
BUILD_ROOT=/tmp/aorta-image-build
rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}"
tar --extract --gzip --file "${SOURCE_ARCHIVE}" --directory "${BUILD_ROOT}"

docker build --pull=false --tag "${IMAGE_TAG}" --file "${BUILD_ROOT}/infra/watcloud/Dockerfile" "${BUILD_ROOT}"
docker run --rm "${IMAGE_TAG}" python -m pytest
docker run --rm "${IMAGE_TAG}" python -m aorta_surrogate.training.watcloud_trainer --help >/dev/null

IMAGE_ARCHIVE="${IMAGE_ROOT}/${IMAGE_ARCHIVE_NAME}"
TEMP_ARCHIVE="${IMAGE_ARCHIVE}.partial.${SLURM_JOB_ID}"
docker save "${IMAGE_TAG}" | gzip --stdout > "${TEMP_ARCHIVE}"
mv "${TEMP_ARCHIVE}" "${IMAGE_ARCHIVE}"
(cd "${IMAGE_ROOT}" && sha256sum "${IMAGE_ARCHIVE_NAME}" > "${IMAGE_ARCHIVE_NAME}.sha256")
docker image inspect "${IMAGE_TAG}" > "${IMAGE_ROOT}/watcloud-preop-aaa-v1-image-inspect.json"
echo "Image ready: ${IMAGE_ARCHIVE}"

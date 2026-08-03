#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_persistent_root

NODE_DESCRIPTION="$(scontrol show node trpro-slurm2)"
grep -Eq 'Gres=.*gpu:rtx_4090:' <<<"${NODE_DESCRIPTION}" || {
  echo "trpro-slurm2 does not currently advertise gpu:rtx_4090; inspect: scontrol show node trpro-slurm2" >&2
  exit 7
}
verify_sidecar "${BUNDLE_ROOT}/${DATA_BUNDLE_NAME}"
verify_sidecar "${BUNDLE_ROOT}/${SOURCE_BUNDLE_NAME}"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
BUILD_JOB="$(sbatch --parsable slurm/build_image.sbatch)"
SWEEP_JOB="$(sbatch --parsable --dependency="afterok:${BUILD_JOB}" slurm/memory_sweep.sbatch)"
printf 'Submitted image build %s and dependent memory sweep %s.\n' "${BUILD_JOB}" "${SWEEP_JOB}"
printf 'Full training was not submitted. Review %s first.\n' "${RUN_ROOT}/memory_sweep/patch_selection.json"

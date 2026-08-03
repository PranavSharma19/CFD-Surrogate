#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_persistent_root

SELECTION="${RUN_ROOT}/memory_sweep/patch_selection.json"
test -f "${SELECTION}" || { echo "Missing patch selection: ${SELECTION}" >&2; exit 6; }
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="patch_selected"; print("Selected patch:",x["selected_patch_nodes"])' "${SELECTION}"
NODE_DESCRIPTION="$(scontrol show node trpro-slurm2)"
grep -Eq 'Gres=.*gpu:rtx_4090:' <<<"${NODE_DESCRIPTION}" || { echo "RTX 4090 GRES is unavailable" >&2; exit 7; }
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
ARRAY_SPEC="${1:-0}"
JOB_ID="$(sbatch --parsable --array="${ARRAY_SPEC}" slurm/train_folds.sbatch)"
printf 'Submitted training job %s with array spec %s.\n' "${JOB_ID}" "${ARRAY_SPEC}"

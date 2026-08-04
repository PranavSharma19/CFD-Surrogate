#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_aws_environment
require_commands
sync_results
echo "AWS results and state synchronized to ${AWS_S3_URI}."

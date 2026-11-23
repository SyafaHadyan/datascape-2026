#!/usr/bin/env bash
# Start a training run on Modal with the run tag also used as the app name in the dashboard.
# Usage, scripts/train.sh RUN_TAG [options for train, such as --model-name or --img-size 288]
set -euo pipefail
tag="${1:?Usage, scripts/train.sh RUN_TAG [train options]}"
shift
cd "$(dirname "$0")"
exec modal run -n "$tag" modal_train.py::train --run-tag "$tag" "$@"

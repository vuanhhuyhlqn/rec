#!/usr/bin/env bash
# Baseline: PAAC fine-tuning WITHOUT any pre-training (random-init backbone).
# Useful as the reference point for the pre-train scenarios.
set -euo pipefail
cd "$(dirname "$0")/.."          # -> rec/
source "$(dirname "$0")/_env.sh"

$PY -m newsrec.scripts.run_finetune \
    --config newsrec/config/finetune/paac.yaml \
    run_name=finetune_baseline "$@"

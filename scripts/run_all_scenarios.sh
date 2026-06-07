#!/usr/bin/env bash
# Run every pre-train scenario (pretrain -> PAAC finetune) sequentially.
# Forwards extra overrides to all runs, e.g.:  bash scripts/run_all_scenarios.sh device=cuda
set -euo pipefail
cd "$(dirname "$0")"

SCENARIOS=(full item_level sequence_level attribute aap_only mip_only map_only sp_only bsm_only)
for s in "${SCENARIOS[@]}"; do
    echo "########## SCENARIO: ${s} ##########"
    bash "pretrain_${s}.sh" "$@"
done
echo "All scenarios complete."

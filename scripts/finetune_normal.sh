#!/usr/bin/env bash
# Normal fine-tuning: plain recommender WITHOUT PAAC (no L_sa / L_cl).
# Trains L_rec(BPR) + L2 only. Checkpoints are interchangeable with PAAC runs.
#   bash scripts/finetune_normal.sh device=cuda
# Forward a pretrained backbone the same way as the scenario scripts, e.g.:
#   bash scripts/finetune_normal.sh device=cuda \
#       finetune.pretrained_ckpt=checkpoints/pretrain_full/pretrain/best
set -euo pipefail
cd "$(dirname "$0")/.."          # -> rec/
source "$(dirname "$0")/_env.sh"

$PY -m newsrec.scripts.run_finetune \
    --config newsrec/config/finetune/normal.yaml \
    run_name=finetune_normal "$@"

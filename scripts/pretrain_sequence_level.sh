#!/usr/bin/env bash
# Scenario: sequence_level
# Pre-train with the 'sequence_level' task combination, then PAAC fine-tune from that
# checkpoint. Extra CLI overrides are forwarded to BOTH stages, e.g.:
#   bash scripts/pretrain_sequence_level.sh device=cuda
set -euo pipefail
cd "$(dirname "$0")/.."          # -> rec/

SCENARIO="sequence_level"
PRETRAIN_RUN="pretrain_${SCENARIO}"

echo "[${SCENARIO}] === Pre-training ==="
python -m newsrec.scripts.run_pretrain \
    --config newsrec/config/pretrain/${SCENARIO}.yaml "$@"

echo "[${SCENARIO}] === PAAC fine-tuning (loading pretrained backbone) ==="
python -m newsrec.scripts.run_finetune \
    --config newsrec/config/finetune/paac.yaml \
    run_name=finetune_${SCENARIO} \
    finetune.pretrained_ckpt=checkpoints/${PRETRAIN_RUN}/pretrain/best "$@"

echo "[${SCENARIO}] === Done ==="

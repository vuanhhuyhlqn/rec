#!/usr/bin/env bash
# Push the local MIND splits to a HuggingFace dataset repo.
# Token is read from rec/.env (HUGGINGFACE_TOKEN / HF_TOKEN).
# Repo id defaults to <your-hf-username>/mind-small.
#
# Examples:
#   bash scripts/push_dataset.sh
#   bash scripts/push_dataset.sh --repo-id myuser/mind-small --private
#   bash scripts/push_dataset.sh --include-embeddings
set -euo pipefail
cd "$(dirname "$0")/.."          # -> rec/

python -m newsrec.scripts.push_dataset \
    --train dataset/train --dev dataset/dev "$@"

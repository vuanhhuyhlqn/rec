#!/usr/bin/env bash
# =============================================================================
# setup.sh — bootstrap a fresh machine to run the newsrec pipeline.
#
# Idempotent: safe to re-run. It will
#   1. install `uv` if it is missing,
#   2. `uv sync` the project (installs the cu128 torch build pinned in
#      pyproject.toml — compatible with CUDA 12.8 drivers and Blackwell GPUs),
#   3. verify torch + CUDA,
#   4. (optionally) write a .env with your HuggingFace token,
#   5. (optionally) pre-download the MIND dataset into dataset/.
#
# Usage:
#   bash setup.sh                       # install deps + verify GPU
#   HUGGINGFACE_TOKEN=hf_xxx bash setup.sh   # also writes .env
#   bash setup.sh --with-data           # also pre-fetch the dataset
#   bash setup.sh --token hf_xxx --with-data
#
# Run training afterwards (see docs/guide.md), e.g. on a 16 GB GPU:
#   tmux new -s train
#   source .venv/bin/activate
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     python -m newsrec.scripts.run_finetune \
#       --config newsrec/config/finetune/paac.yaml \
#       run_name=finetune_baseline device=cuda \
#       finetune.batch_size=8 data.max_history=30
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"          # repo root (rec/)
REPO_ROOT="$(pwd)"

WITH_DATA=0
TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

while [ $# -gt 0 ]; do
  case "$1" in
    --with-data) WITH_DATA=1; shift ;;
    --token)     TOKEN="${2:-}"; shift 2 ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[setup:warn]\033[0m %s\n' "$*"; }

# --- 1. uv ------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # make uv available in this shell session
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { warn "uv not on PATH; open a new shell or add ~/.local/bin to PATH"; exit 1; }
log "uv: $(uv --version)"

# --- 2. dependencies --------------------------------------------------------
log "Syncing dependencies (this installs the cu128 torch build) ..."
uv sync

# --- 3. verify torch + CUDA -------------------------------------------------
log "Verifying torch / CUDA ..."
.venv/bin/python - <<'PY'
import torch
print(f"  torch        : {torch.__version__}")
ok = torch.cuda.is_available()
print(f"  cuda.is_available: {ok}")
if ok:
    print(f"  gpu          : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  gpu memory   : {total/1e9:.1f} GB total")
    x = torch.randn(2048, 2048, device="cuda"); _ = (x @ x).sum().item()
    print("  gpu matmul   : OK")
else:
    print("  (no GPU visible — training will run on CPU and is likely to be very slow / OOM)")
PY

# --- 4. .env token ----------------------------------------------------------
if [ -n "$TOKEN" ]; then
  if [ -f .env ] && grep -q "HUGGINGFACE_TOKEN=" .env; then
    log ".env already contains a HUGGINGFACE_TOKEN; leaving it unchanged"
  else
    log "Writing HuggingFace token to .env"
    printf 'HUGGINGFACE_TOKEN=%s\n' "$TOKEN" >> .env
  fi
else
  warn "No HF token provided. The public dataset still downloads, but set one to"
  warn "avoid rate limits / push checkpoints:  HUGGINGFACE_TOKEN=hf_xxx bash setup.sh"
fi

# --- 5. optional data prefetch ---------------------------------------------
if [ "$WITH_DATA" -eq 1 ]; then
  if [ -f dataset/train/news.tsv ] && [ -f dataset/dev/news.tsv ]; then
    log "Dataset already present in dataset/ — skipping download"
  else
    log "Pre-downloading MIND dataset into dataset/ ..."
    .venv/bin/python -m newsrec.scripts.prepare_data || \
      warn "prepare_data failed; data will be fetched lazily on the first training run"
  fi
fi

log "Setup complete. Repo: $REPO_ROOT"
log "Next: see docs/guide.md (run inside tmux on a 16 GB GPU with batch_size=8 / max_history=30)."

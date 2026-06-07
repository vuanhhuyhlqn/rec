# newsrec — Popularity-Debiased News Recommender (MIND)

A news recommendation model for the [MIND](https://msnews.github.io/) dataset
that tackles popularity (long-tail) bias through two complementary mechanisms:

1. **Self-supervised pre-training** — a configurable subset of five S3Rec-style
   tasks (AAP, MIP, MAP, SP, BSM), all InfoNCE-based.
2. **PAAC fine-tuning** — Popularity-Aware Alignment and Contrast:
   `L_total = L_rec + λ1·L_sa + λ2·L_cl + λ3·‖Θ‖²`.

## Architecture

```
news text (title + abstract)
        │
        ▼
Module 0:  DistilBERT (distilbert-base-uncased) + LoRA  ── word embeddings [B, L, D]
        │                                        (gradual layer unfreezing)
        ▼
Module 1:  Fastformer news encoder            ── news vector  h_i  [B, D]
        │
        ▼  (history of news vectors)
Module 2:  Fastformer user encoder            ── states [B, S, D]
        │  + additive attention pooler         ── user vector z_u [B, D]
        ▼
score(z_u, candidate h_i) = cosine similarity
```

The same Modules 0/1/2 are shared between the pre-training and fine-tuning
stages; a pre-trained checkpoint can be loaded as the fine-tuning backbone.
The PLM encoder is architecture-aware — it auto-detects LoRA target modules and
layer names for both BERT and DistilBERT families.

## Layout

```
newsrec/
  config/      base.yaml + pretrain/ + finetune/ + smoke/ presets
  data/        MIND parsing, vocab, popularity p(i), datasets, token table,
               download (HF auto-download)
  models/      plm_encoder, fastformer, news_encoder, user_encoder,
               attention_pooler, rec_model, pretrain_model
  losses/      infonce, pretrain_losses (AAP/MIP/MAP/SP/BSM), paac_losses
               (BPR / L_sa / L_cl)
  eval/        metrics (AUC/MRR/nDCG@5/@10), impression evaluator
  training/    lora_schedule, finetuner, pretrainer, checkpoint, hub_uploader,
               batch_finder (auto batch size)
  utils/       config, logging, seed, env (dotenv + HF token)
  scripts/     prepare_data, run_pretrain, run_finetune, push_dataset, common
scripts/       *.sh runners (setup.sh, pretrain_<scenario>.sh, ...)
dataset/       train/ + dev/  (local or auto-downloaded; git-ignored)
docs/          guide.md + architecture.md
tests/         pytest suite (shape/dimension + smoke + batch finder)
```

## Setup (uv)

On a fresh machine, the bootstrap script installs `uv`, syncs deps (the CUDA
12.8 torch build), verifies the GPU, and optionally pre-downloads the data:

```bash
cd rec
bash setup.sh                                   # deps + GPU check
HUGGINGFACE_TOKEN=hf_xxx bash setup.sh --with-data   # also writes .env + fetches data
```

Or manually:

```bash
cd rec
uv venv
uv pip install -e ".[dev]"
```

> torch is pinned to the **cu128** wheel index in `pyproject.toml` so a fresh
> `uv sync` produces a GPU build compatible with CUDA 12.8 drivers and Blackwell
> GPUs (the default PyPI wheels ship a cu13 build that fails on those drivers).

The MIND small split lives in `rec/dataset/train` and `rec/dataset/dev` (present
locally, or auto-downloaded — see below).

Inspect the data:

```bash
python -m newsrec.scripts.prepare_data
```

## Dataset on the Hub (auto-download)

The MIND splits are published as a HuggingFace **dataset** repo so a fresh
machine needs no manual data copy. The default config points at
`huyva/mind-small` with `auto_download: true`; the splits are fetched only when
the local `dataset/train` / `dataset/dev` directories are missing, and land in a
project-local `dataset/` directory (not the global HF cache).

Re-publish (or push to your own repo):

```bash
bash scripts/push_dataset.sh                         # -> <your-username>/mind-small
bash scripts/push_dataset.sh --repo-id me/mind-small --private --include-embeddings
```

Point a run at a different repo (or disable remote fetch):

```yaml
data:
  hf_dataset_repo: your-username/mind-small   # or null to disable
  auto_download: true
  download_dir: dataset
```

## Pre-train scenarios (task combinations)

Each scenario is a config under `newsrec/config/pretrain/` plus a `.sh` runner
in `scripts/` that pre-trains with that task combination and then PAAC
fine-tunes from the resulting checkpoint:

| Scenario | Tasks | Script |
|---|---|---|
| full | AAP+MIP+MAP+SP+BSM | `scripts/pretrain_full.sh` |
| item_level | AAP+MIP+MAP | `scripts/pretrain_item_level.sh` |
| sequence_level | MIP+SP+BSM | `scripts/pretrain_sequence_level.sh` |
| attribute | AAP+MAP | `scripts/pretrain_attribute.sh` |
| aap_only / mip_only / map_only / sp_only / bsm_only | single task | `scripts/pretrain_<task>_only.sh` |

```bash
bash scripts/pretrain_full.sh                 # one scenario (pretrain + finetune)
bash scripts/pretrain_full.sh device=cuda     # overrides forwarded to both stages
bash scripts/finetune_baseline.sh             # PAAC with no pre-training (baseline)
bash scripts/run_all_scenarios.sh device=cuda # every scenario sequentially
```

Checkpoints are namespaced by `run_name`, e.g.
`checkpoints/pretrain_item_level/pretrain/best`, so scenarios never overwrite
each other.

## Training (run off-machine / on GPU)

`device` defaults to `auto` (uses CUDA if a GPU is visible, else CPU). Configs
support inheritance via `_base_` and dotted `key=value` CLI overrides.

Pre-train with all five tasks, then PAAC fine-tune from that checkpoint:

```bash
# 1) Pre-train (writes checkpoints/pretrain_full/pretrain/{epochN,best})
python -m newsrec.scripts.run_pretrain --config newsrec/config/pretrain/full.yaml

# 2) Fine-tune with PAAC, loading the pre-trained backbone
python -m newsrec.scripts.run_finetune --config newsrec/config/finetune/paac.yaml \
    finetune.pretrained_ckpt=checkpoints/pretrain_full/pretrain/best
```

Ablations (single pre-train task) — use a preset or override the `tasks` list:

```bash
python -m newsrec.scripts.run_pretrain --config newsrec/config/pretrain/mip_only.yaml
# or override inline:
python -m newsrec.scripts.run_pretrain --config newsrec/config/pretrain/full.yaml \
    pretrain.tasks='[aap, bsm]'
```

### Performance & memory

Several optimizations keep training fast and OOM-safe:

* **bf16 mixed precision** (`amp: true`, GPU only) — ~1.5–2× faster, lower
  memory.
* **Padding-skip encoding** — the news encoder runs BERT only on real history
  items, not padded slots (~2× on short histories).
* **Auto batch size** — set `finetune.batch_size: auto` (default) to probe the
  GPU for the largest batch that fits (no OOM). It probes the *worst-case*
  state (longest histories **and** the schedule's maximum layer unfreeze) and
  applies a `batch_safety` margin (0.95). Pin an integer to disable.
* On a **shared** GPU prefer pinning the batch — co-tenant memory fluctuates and
  can make the finder under/over-shoot.

```bash
python -m newsrec.scripts.run_finetune --config newsrec/config/finetune/paac.yaml \
    device=cuda data.max_history=30 finetune.batch_size=auto
```

Run long jobs inside `tmux` and detach (`Ctrl-b d`) so they survive disconnects.

### Choosing pre-train tasks

`pretrain.tasks` accepts either a list or a per-task mapping with weights:

```yaml
pretrain:
  tasks:
    aap: {enabled: true,  weight: 1.0}
    mip: {enabled: true,  weight: 1.0}
    map: {enabled: false}
    sp:  {enabled: true,  weight: 0.5}
    bsm: {enabled: true,  weight: 1.0}
```

## Logging

Every run writes to both the console and a timestamped file
`logs/{stage}_{run_name}_{timestamp}.log`. A **tqdm progress bar** shows
per-epoch progress (total batches, it/s, ETA) on the console; the detailed
**per-step loss breakdown** (`L_rec`, `L_sa`, `L_cl`, and each pre-train task)
is logged at DEBUG to the **file only**, so it doesn't break the progress bar.
Epoch summaries, LoRA-unfreeze events, and dev metrics (AUC / MRR / nDCG@5 /
nDCG@10) appear on both.

## HuggingFace checkpointing

Checkpoints are saved locally and (optionally) pushed to the HuggingFace Hub.
Uploads run **asynchronously** on a background thread so training is never
blocked on network I/O. Both stages push periodic checkpoints plus a separate
`best/` checkpoint (best dev-AUC for fine-tuning; lowest loss for pre-training).

Enable in any config and provide a token via the environment:

```yaml
hub:
  push_to_hub: true
  hub_repo_id: your-username/newsrec-mind
  hub_private: false
```

```bash
# Either export it...
export HF_TOKEN=hf_xxx          # HUGGINGFACE_TOKEN also accepted
# ...or put it in rec/.env (auto-loaded, and git-ignored):
echo "HUGGINGFACE_TOKEN=hf_xxx" > .env
```

The token is read from `HF_TOKEN`, `HUGGINGFACE_TOKEN`, or
`HUGGINGFACEHUB_API_TOKEN` (in that precedence). A `rec/.env` file is loaded
automatically at the start of each run and is excluded from git via
`.gitignore`. If `push_to_hub` is false or no token is found, the run falls
back to local-only checkpointing (with a logged warning).

A checkpoint directory contains: `model.pt` (Modules 0/1/2 + pooler),
`config.yaml`, `lora/` (LoRA adapters via peft), `tokenizer/`, and `extra.pt`
(pre-training heads).

## Tests

The suite asserts the output dimensions of every module and loss and runs a
tiny end-to-end pipeline on CPU (no GPU / no network required):

```bash
cd rec
python -m pytest            # full suite
python -m pytest tests/test_smoke.py   # end-to-end smoke test only
```

## Default hyper-parameters

| Component | Value |
|---|---|
| PLM | distilbert-base-uncased + LoRA (r=8, α=16) |
| Model dim | 256 |
| Fastformer | 2 layers, 16 heads |
| History length | 50 (use 30 for faster/lighter runs) |
| Title+abstract tokens | 64 |
| InfoNCE temperature τ | 0.1 |
| MIP mask ratio | 0.15 |
| PAAC popular split (x% / sa_ratio) | 80% / 0.8 |
| PAAC β / γ | 1.0 / 0.5 |
| λ1 / λ2 / λ3 | 0.1 / 0.1 / 1e-4 |
| Mixed precision | bf16 (`amp: true`, GPU) |
| Batch size | `auto` (probed) + 0.95 safety |
| Device | `auto` (cuda if available) |

All are overridable via YAML or CLI.

## References

* S3Rec (CIKM 2020) — self-supervised pre-training tasks.
* PAAC (2024) — popularity-aware alignment & contrast.
* Fastformer (2021) — additive-attention encoder.

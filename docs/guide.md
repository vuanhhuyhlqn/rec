# Guide: running the newsrec pipeline

A practical, step-by-step guide to setting up, training, evaluating, and
checkpointing the popularity-debiased MIND news recommender. For *what the code
is*, read `architecture.md` alongside this.

All commands assume you are in the `rec/` directory unless stated otherwise.

---

## 0. TL;DR

```bash
cd rec

# environment (fresh machine): installs uv + cu128 torch, verifies GPU, fetches data
bash setup.sh --with-data
source .venv/bin/activate

# sanity-check the data + the whole pipeline
python -m newsrec.scripts.prepare_data
python -m pytest tests/test_smoke.py

# run one scenario end-to-end (pretrain -> PAAC finetune); batch size auto-detected
bash scripts/pretrain_full.sh device=cuda
```

---

## 1. Environment setup

Requirements: Python ≥ 3.10, `uv`. **Easiest path — `setup.sh`** (idempotent):

```bash
cd rec
bash setup.sh                                    # uv + deps + GPU verify
HUGGINGFACE_TOKEN=hf_xxx bash setup.sh --with-data   # also write .env + prefetch data
```

Or manually:

```bash
cd rec
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"     # installs torch(cu128), transformers, peft, hf_hub, sklearn, pytest, ...
```

Dependencies (from `pyproject.toml`): `torch>=2.7` (pinned to the **cu128** wheel
index), `transformers`, `peft`, `huggingface_hub`, `scikit-learn`, `numpy`,
`pandas`, `pyyaml`, `tqdm`; dev extra adds `pytest`.

> **torch / CUDA**: the cu128 pin makes `uv sync` produce a GPU build compatible
> with CUDA-12.8 drivers and Blackwell GPUs. The default PyPI cu13 wheels report
> `cuda.is_available() == False` on those drivers → a silent CPU run → host OOM.
> The first run downloads `distilbert-base-uncased` weights + tokenizer (cached).

---

## 2. Data

### Already-present local data
The MIND-small split lives in `rec/dataset/train/` and `rec/dataset/dev/`
(`news.tsv`, `behaviors.tsv`, and `popularity_data.csv` in train).

Inspect it:
```bash
python -m newsrec.scripts.prepare_data
# prints #news, #impressions, #users, #categories (~17), avg history length, vocab sizes
```

### Auto-download on a fresh machine
The default config (`newsrec/config/base.yaml`) sets:
```yaml
data:
  hf_dataset_repo: huyva/mind-small
  auto_download: true
  download_dir: dataset        # downloaded splits land in rec/dataset/<split>/
```
`load_news_and_tokens` calls `ensure_mind_split`, which **only downloads when
the local split is missing**. So on this machine local files are used; on a new
machine the splits are fetched from the Hub automatically (a token is needed
only if the dataset repo is private — see §6).

Downloaded files are written to a **project-local** `rec/dataset/` directory
(e.g. `dataset/train/news.tsv`, `dataset/dev/news.tsv`) — *not* the global
`~/.cache/huggingface/hub` cache — and reused on subsequent runs. `dataset/` is
git-ignored. Change the location with `data.download_dir`.

To disable remote fetch: set `data.hf_dataset_repo: null` or
`data.auto_download: false`.

### Re-publishing the dataset
```bash
bash scripts/push_dataset.sh                              # -> <your-username>/mind-small
bash scripts/push_dataset.sh --repo-id me/mind --private  # custom repo, private
bash scripts/push_dataset.sh --include-embeddings         # also push *.vec files
```
This reads the token from `.env` (see §6), resolves your username via
`whoami()`, creates the dataset repo, and uploads `train/` + `dev/` subfolders.

---

## 3. The two training stages

### Stage 1 — pre-training (self-supervised)
```bash
python -m newsrec.scripts.run_pretrain --config newsrec/config/pretrain/full.yaml
```
Writes checkpoints to `checkpoints/<run_name>/pretrain/{epochN, best}` (run_name
= `pretrain_full` here). `best` tracks the lowest average loss.

### Stage 2 — PAAC fine-tuning
```bash
python -m newsrec.scripts.run_finetune --config newsrec/config/finetune/paac.yaml \
    device=cuda data.max_history=30 \
    finetune.pretrained_ckpt=checkpoints/pretrain_full/pretrain/best
```
Trains BPR + L_sa + L_cl, evaluates on the dev split each epoch (AUC / MRR /
nDCG@5 / nDCG@10), and saves `checkpoints/<run_name>/finetune/{epochN, best}`
(`best` = best dev AUC). `device` defaults to `auto`; `batch_size` defaults to
`auto` (probed — see §5.1). Use the **same `max_history`** in both stages.

Omit `finetune.pretrained_ckpt` (or leave it `null`) to fine-tune from scratch
(the baseline). Run long jobs inside `tmux` (detach with `Ctrl-b d`).

---

## 4. Pre-train scenarios (task combinations)

Each scenario has a config in `newsrec/config/pretrain/` and a one-shot shell
runner in `scripts/` that does **pretrain → finetune**:

| Scenario | Tasks | Command |
|---|---|---|
| full | AAP+MIP+MAP+SP+BSM | `bash scripts/pretrain_full.sh` |
| item_level | AAP+MIP+MAP | `bash scripts/pretrain_item_level.sh` |
| sequence_level | MIP+SP+BSM | `bash scripts/pretrain_sequence_level.sh` |
| attribute | AAP+MAP | `bash scripts/pretrain_attribute.sh` |
| aap_only | AAP | `bash scripts/pretrain_aap_only.sh` |
| mip_only | MIP | `bash scripts/pretrain_mip_only.sh` |
| map_only | MAP | `bash scripts/pretrain_map_only.sh` |
| sp_only | SP | `bash scripts/pretrain_sp_only.sh` |
| bsm_only | BSM | `bash scripts/pretrain_bsm_only.sh` |

Extra arguments are **forwarded to both stages**:
```bash
bash scripts/pretrain_item_level.sh device=cuda
bash scripts/pretrain_full.sh device=cuda pretrain.training.epochs=20 finetune.epochs=12
```

Other runners:
```bash
bash scripts/finetune_baseline.sh device=cuda     # PAAC, no pre-training (reference)
bash scripts/run_all_scenarios.sh device=cuda     # every scenario sequentially
```

Each scenario's checkpoints are namespaced by `run_name`
(`checkpoints/pretrain_<scenario>/...` and `checkpoints/finetune_<scenario>/...`),
so they never overwrite one another.

### Defining a custom combination on the fly
```bash
python -m newsrec.scripts.run_pretrain --config newsrec/config/pretrain/full.yaml \
    "pretrain.tasks=[aap, mip, bsm]" run_name=pretrain_custom
```
Or with per-task weights, edit a YAML:
```yaml
pretrain:
  tasks:
    aap: {enabled: true, weight: 1.0}
    mip: {enabled: true, weight: 2.0}
    sp:  {enabled: false}
```

---

## 5. Configuration system

Configs are YAML with two power features:

1. **Inheritance** via `_base_` (path relative to the file):
   ```yaml
   _base_: ../base.yaml
   run_name: my_run
   ```
2. **CLI overrides** as trailing `dotted.key=value` args (scalars are coerced
   to int/float/bool/None):
   ```bash
   python -m newsrec.scripts.run_finetune --config .../paac.yaml \
       device=cuda finetune.lambda2=0.2 model.model_dim=128 seed=7
   ```

Precedence (low → high): `_base_` files → current file → `overrides` dict →
`cli_overrides`.

Key knobs (full schema in `architecture.md` §9):
- `device`: `auto` (default; cuda if available else cpu), `cuda`, or `cpu`.
- `data.max_title_len`, `data.max_history`, `data.mask_prob`.
- `model.model_dim`, `model.plm.{model_name, lora_r, lora_alpha}`, `model.score.temperature`.
- `model.news_encoder.num_heads` / `model.user_encoder.num_heads` (default 16).
- `finetune.{lambda1, lambda2, lambda3, cl_beta, cl_gamma, cl_tau, sa_ratio, cl_x_percent}`.
- `finetune.{batch_size, max_batch_size, batch_safety, amp}` (see §5.1).
- Training is **LoRA-only**: the DistilBERT base stays frozen and only the LoRA
  adapters (+ the Fastformer encoders/heads) train. Tune capacity via
  `model.plm.lora_r` / `lora_alpha`.

Defaults (overridable): τ=0.1, mask ratio=0.15, **PAAC popular split x=80% /
sa_ratio=0.8**, β=1.0, γ=0.5, λ1=λ2=0.1, λ3=1e-4, model_dim=256, **Fastformer 2
layers × 16 heads**, history=50, title+abstract tokens=64, **PLM =
distilbert-base-uncased** (6 layers), bf16 AMP on, batch_size=auto.

### 5.1 Auto batch size & mixed precision

- `finetune.batch_size: auto` (default) probes the GPU for the largest batch
  that fits and applies `batch_safety` (0.95). It probes the **worst case** —
  the longest histories — so the chosen batch stays safe for every batch.
  Pin an integer to disable; on CPU it uses `fallback_batch_size` (8).
- `max_batch_size` caps the search (default 256).
- `amp: true` enables **bf16 mixed precision** on GPU (~1.5–2× faster, lower
  memory; no-op on CPU).
- ⚠️ On a **shared** GPU the finder can be unstable (a co-tenant's memory use
  fluctuates) — prefer pinning `finetune.batch_size` to a value you've verified.

---

## 6. HuggingFace token & checkpointing

### Token
Provide a token via **either** an environment variable **or** `rec/.env`
(auto-loaded, git-ignored):
```bash
export HF_TOKEN=hf_xxx
# or
echo "HUGGINGFACE_TOKEN=hf_xxx" > .env
```
Accepted names (precedence): `HF_TOKEN`, `HUGGINGFACE_TOKEN`,
`HUGGINGFACEHUB_API_TOKEN`. Needed for pushing checkpoints/datasets and for
downloading private repos.

### Pushing checkpoints to the Hub
Enable in any config:
```yaml
hub:
  push_to_hub: true
  hub_repo_id: your-username/newsrec-mind
  hub_private: false
```
Uploads run **asynchronously** on a background thread (training never blocks on
network). Both stages push periodic checkpoints plus a separate `best/`
revision. If `push_to_hub` is false or no token is found, the run falls back to
local-only checkpointing with a logged warning.

A checkpoint directory contains: `model.pt` (Modules 0/1/2 + pooler),
`config.yaml`, `lora/` (peft adapters), `tokenizer/`, `extra.pt` (pre-train
heads).

### Resuming / using a checkpoint
- Load a pre-trained backbone into fine-tuning:
  `finetune.pretrained_ckpt=checkpoints/<run>/pretrain/best`.
- In code: `Finetuner.load_pretrained(ckpt_dir)` →
  `load_backbone_weights(model, ckpt_dir, strict=False)`.

---

## 7. Logging & outputs

Every run writes to console and a file:
```
rec/logs/{stage}_{run_name}_{timestamp}.log
```
The **console** shows a tqdm progress bar (total batches, it/s, ETA) plus epoch
summaries and dev metrics. The detailed **per-step loss
breakdown** (`L_rec`, `L_sa`, `L_cl`, and each pre-train task) is written at
DEBUG to the **file only**, so it never breaks the progress bar. Set
`logging.level=DEBUG` to also show step lines on the console.

Outputs land in (all git-ignored):
- `rec/logs/` — run logs.
- `rec/checkpoints/<run_name>/{pretrain,finetune}/{epochN,best}/` — checkpoints.

---

## 8. Evaluation

Evaluation runs automatically during fine-tuning (`finetune.eval_every` epochs)
on the dev split, reporting AUC / MRR / nDCG@5 / nDCG@10 averaged per
impression. To cap cost, set `finetune.max_eval_impressions`.

Programmatic evaluation:
```python
from newsrec.eval.evaluator import ImpressionEvaluator
ev = ImpressionEvaluator(model, device="cuda")
news_vectors = ev.encode_news_table(news_tokens.tokens)   # {nid: {input_ids, attention_mask}}
metrics = ev.evaluate(dev_impressions, news_vectors, max_history=50)
print(metrics)  # {'auc':..., 'mrr':..., 'ndcg@5':..., 'ndcg@10':...}
```

> ⚠️ Keep `max_history` ≤ `model.max_history_len` (the user Fastformer's
> position capacity). The built-in evaluator caps this automatically.

---

## 9. Tests

```bash
cd rec
python -m pytest                      # full suite (~109 tests, ~35s)
python -m pytest tests/test_smoke.py  # end-to-end tiny pipeline only
python -m pytest -k paac              # filter by name
```
Tests are offline-safe (tiny random BERT, fake HF API). Real-data tests skip if
`dataset/train` is absent. After any change, run the full suite (in default
order) before committing.

---

## 10. Typical workflows

### A. Reproduce the headline run
```bash
bash scripts/pretrain_full.sh device=cuda
# -> checkpoints/pretrain_full/... then checkpoints/finetune_full/... + dev metrics in logs
```

### B. Ablate a single task vs the baseline
```bash
bash scripts/finetune_baseline.sh device=cuda      # no pretraining
bash scripts/pretrain_bsm_only.sh device=cuda      # BSM-only pretraining
# compare dev AUC/nDCG in the two finetune logs
```

### C. New machine, private dataset, push results
```bash
echo "HUGGINGFACE_TOKEN=hf_xxx" > .env
bash scripts/pretrain_full.sh device=cuda \
    hub.push_to_hub=true hub.hub_repo_id=me/newsrec-full
# data auto-downloads; checkpoints push asynchronously to the Hub
```

### D. Quick CPU smoke before a big run
```bash
python -m pytest tests/test_smoke.py
```

---

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Process **`Killed`** (no traceback), runs on CPU | torch can't see the GPU — usually the wrong CUDA build. Check `.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`. Reinstall the cu128 build (`uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128`) or run `setup.sh`. |
| `torch.cuda.OutOfMemoryError` | Batch too large for the card. With `batch_size: auto` this is rare; if pinned, lower `finetune.batch_size`, lower `batch_safety`, or reduce `data.max_history`. (Pretrain configs pin `batch_size: 32` — set it to `auto` or a smaller int on small GPUs.) |
| Auto batch picks `1` / wildly varying sizes | **Shared GPU** — a co-tenant's memory fluctuates during the probe. Pin `finetune.batch_size` to a verified value. |
| Epoch ETA absurdly long (tqdm total == #triplets) | `batch_size` resolved to 1. See the two rows above; pin the batch. |
| `IndexError: index out of range in self` during forward/eval | Sequence longer than Fastformer position capacity. Increase `model.max_title_len` / `model.max_history_len`, or lower `data.max_history`. |
| `Index put requires ... dtypes match` in `encode_news` | The pad-skip buffer dtype must match the encoder output (autocast bf16). Already handled; don't revert that. |
| `FileNotFoundError: news.tsv not found` | Local split missing and auto-download disabled/misconfigured. Set `data.hf_dataset_repo` + `data.auto_download: true` + `data.download_dir`, or fix `data.train_dir`. |
| HF upload silently does nothing | `hub.push_to_hub` false or no token. Check the log for the device/token line. |
| `peft` UserWarning about missing config when saving | Benign; LoRA adapters still save correctly. |
| Pretrained weights don't seem to load | Confirm `finetune.pretrained_ckpt` points at a dir containing `model.pt`; the log prints `missing=/unexpected=` counts (should be ~0). |
| Out of memory on GPU (up front) | Lower `finetune.batch_size` / `data.max_history` / `model.model_dim`, keep AMP on, or reduce `model.plm.lora_r`. |
| Two scenarios overwrote each other's checkpoints | They shared a `run_name`. Give each run a unique `run_name`. |

---

## 12. Where to look next

- `docs/architecture.md` — module-by-module internals, tensor contracts, data
  flow per step, and extension recipes.
- `README.md` — condensed quick start.
- `tests/` — every public contract has an example of intended use.

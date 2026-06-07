# Architecture

This document describes the structure and internals of the `newsrec` codebase
(a popularity-debiased MIND news recommender). It is written for engineers and
AI agents who need to extend or debug the system in later sessions. For *how to
run* things, see `guide.md`.

---

## 1. High-level overview

The project trains a **two-tower news recommender** on the MIND dataset and
fights long-tail popularity bias with two mechanisms:

1. **Self-supervised pre-training** (Stage 1) — a configurable subset of five
   S3Rec-style tasks (AAP, MIP, MAP, SP, BSM), all InfoNCE-based.
2. **PAAC fine-tuning** (Stage 2) — Popularity-Aware Alignment and Contrast:
   `L_total = L_rec + λ1·L_sa + λ2·L_cl + λ3·‖Θ‖²`.

Both stages share the same three encoder modules. Stage 1 writes a checkpoint;
Stage 2 can load it as the backbone.

### The three modules (+ scoring)

```
news text (title + abstract, tokenised)
        │
        ▼
Module 0  PLMEncoder      BERT (bert-base-uncased) + LoRA   → word embeddings [B, L, D_plm=768]
        │                 (gradual layer unfreezing)
        ▼
Module 1  NewsEncoder     Fastformer(D_plm) + attention pool + Linear(D_plm→D)
        │                 → news vector  h_i  [B, D]        (D = model_dim, default 256)
        ▼  (history of news vectors [B, S, D])
Module 2  UserEncoder     Fastformer(D) → states [B, S, D]
        │                 + AdditiveAttentionPooling → user vector z_u [B, D]
        ▼
score(z_u, candidate h_i) = cosine similarity / temperature
```

Key dimension convention: **D_plm = 768** (BERT hidden) feeds Module 1, which
projects down to the shared **model_dim D** (default 256). Module 2 and all
losses operate at D. News vectors `h_i` and the user vector `z_u` are the
central objects every loss consumes.

---

## 2. Directory layout

```
rec/
├── pyproject.toml            # uv-managed; deps: torch, transformers, peft,
│                             #   huggingface_hub, scikit-learn, numpy, pandas, pyyaml, tqdm
├── README.md                 # quick start
├── .env                      # HUGGINGFACE_TOKEN=... (git-ignored)
├── .gitignore                # ignores .env, logs/, checkpoints/, *.pt, caches
├── docs/                     # this folder
│   ├── architecture.md
│   └── guide.md
├── MINDsmall_train/          # data: news.tsv, behaviors.tsv, popularity_data.csv, *.vec
├── MINDsmall_dev/            # data: news.tsv, behaviors.tsv, *.vec
├── scripts/                  # SHELL runners (.sh) — not python
│   ├── pretrain_<scenario>.sh    # 9 scenarios: pretrain + finetune end-to-end
│   ├── run_all_scenarios.sh
│   ├── finetune_baseline.sh
│   └── push_dataset.sh
├── newsrec/                  # the python package
│   ├── config/               # YAML presets (see §8)
│   │   ├── base.yaml
│   │   ├── pretrain/{full,item_level,sequence_level,attribute,
│   │   │             aap_only,mip_only,map_only,sp_only,bsm_only}.yaml
│   │   ├── finetune/paac.yaml
│   │   └── smoke/{pretrain,finetune}.yaml
│   ├── data/                 # parsing, vocab, popularity, datasets, download
│   ├── models/               # the encoders + two-tower + pretrain wrapper
│   ├── losses/               # infonce, pretrain_losses, paac_losses
│   ├── eval/                 # metrics + impression evaluator
│   ├── training/             # trainers, scheduler, checkpoint, hub uploader
│   ├── utils/                # config, logging, seed, env
│   └── scripts/              # PYTHON entry points (run_pretrain, etc.)
└── tests/                    # pytest suite (~101 tests; all offline-safe)
```

> ⚠️ There are **two** `scripts` locations: top-level `rec/scripts/*.sh`
> (shell) and `rec/newsrec/scripts/*.py` (python CLI modules). The shell
> scripts call the python modules via `python -m newsrec.scripts.<name>`.

---

## 3. `newsrec/data/` — data layer

| File | Key symbols | Notes |
|---|---|---|
| `mind_parser.py` | `NewsItem`, `Impression`, `parse_news`, `parse_behaviors`, `MindData`, `load_mind_split(dir)` | Tokenizer-free TSV parsing. `NewsItem.text` = `title + " " + abstract`. `Impression.clicked` / `.non_clicked` derive from candidates. `MindData.stats()` returns counts. |
| `vocab.py` | `Vocab`, `build_news_vocab`, `build_category_vocab`, `NewsTextEncoder` | `Vocab` has special tokens (PAD/UNK/MASK) at fixed ids; `.index()` falls back to UNK. `NewsTextEncoder` wraps a HF `AutoTokenizer` (loads lazily in `__init__`) → `encode`/`encode_batch` produce padded `input_ids`/`attention_mask`. |
| `popularity.py` | `ItemPopularity`, `build_popularity` | `p(i)` from click frequency. `from_impressions(impressions, include_history=True, include_clicked_candidates=True)`. Methods: `count`, `prob`, `top_percent_split(items, x)` (batch-level positional indices), `user_pop_unpop_split(history, ratio)` (per-user, ordering constraint), `long_tail_summary()`. CSV fallback `from_csv`. |
| `news_tokens.py` | `NewsTokenTable` | `{nid: {input_ids, attention_mask}}` + zero PAD entry. `.build(news, encoder, batch_size)`, `.get(nid)`, `.has(nid)`, `.max_len`. Datasets only need a mapping with `get` + `max_len`, so tests pass plain dicts. |
| `finetune_dataset.py` | `FinetuneTripletDataset` | BPR triplets `(history, i_pos, j_neg)`, in-impression negatives; skips impressions without both a click and a non-click. Optional `popularity` adds per-position pop values. |
| `pretrain_dataset.py` | `build_user_sequences`, `PretrainDataset` | Per-user longest-history sequences; emits masking / segment / BSM tensors. |
| `collate.py` | `stack_collate` | Per-key `torch.stack` (all samples are fixed-shape). |
| `download.py` | `ensure_mind_split` | Returns local dir if `news.tsv` present, else `snapshot_download`s `{subfolder}/*` from a HF dataset repo **into a project-local `download_dir` (default `dataset/`)** via `local_dir=`, returning `{download_dir}/{subfolder}`. Reuses an existing `dataset/{subfolder}` if already downloaded. Not the global `~/.cache/huggingface/hub`. |

### Batch tensor contracts (memorise these)

**FinetuneTripletDataset `__getitem__`** (S = `max_history`, L = `max_title_len`):
```
history_input_ids        [S, L] long
history_attention_mask   [S, L] long
history_mask             [S]    float   (1 = real click, 0 = pad)
history_pop              [S]    float   (global count per history item; 0 if no popularity)
pos_input_ids            [L]    long
pos_attention_mask       [L]    long
neg_input_ids            [L]    long
neg_attention_mask       [L]    long
pos_pop                  scalar float
neg_pop                  scalar float
```

**PretrainDataset `__getitem__`** (S = `max_seq_len`, L = `max_title_len`):
```
input_ids        [S, L] long
attention_mask   [S, L] long
seq_mask         [S]    float   (1 = real item, 0 = pad)
mip_mask         [S]    float   (1 = position masked for MIP/MAP; ≥1 always)
category         [S]    long    (category vocab index; 0 at pad)
segment_mask     [S]    float   (1 = inside the SP contiguous segment)
context_mask     [S]    float   (= seq_mask * (1 - segment_mask); SP context)
bsm_a_mask       [S]    float   (first temporal half; disjoint from b)
bsm_b_mask       [S]    float   (second temporal half)
```

After `stack_collate` every key gains a leading batch dim `[B, ...]`.

---

## 4. `newsrec/models/` — encoders & towers

| File | Symbol | Contract |
|---|---|---|
| `plm_encoder.py` | `PLMEncoder` | BERT + LoRA. `forward(input_ids[B,L], attn[B,L]) → (last_hidden_state[B,L,D_plm], mask)`. `set_trainable_layers(n)` opens top-n BERT layers' base weights (LoRA always trainable); `frozen_layers`, `num_trainable_parameters()`, `output_dim`. `pretrained=False` + `small_config` builds a tiny random BERT for offline tests. LoRA targets `["query","key","value","dense"]`. |
| `fastformer.py` | `FastformerConfig`, `FastformerEncoder` | Vendored/adapted Fastformer. **Returns per-position states `[B,S,D]`** (not pooled). Uses `BertIntermediate/BertOutput/BertSelfOutput` from `transformers`. Additive mask `(1-mask)*-1e4`, learned position embeddings + LayerNorm + dropout. |
| `attention_pooler.py` | `AdditiveAttentionPooling` | `forward(x[B,S,D], mask[B,S]) → (pooled[B,D], alpha[B,S])`. Masked-fill `-inf` then softmax + `nan_to_num` (all-pad rows → 0 vector). |
| `news_encoder.py` | `NewsEncoder` | Module 1. `forward(word_emb[B,L,D_plm], mask[B,L]) → h_i[B,D]`. = Fastformer(D_plm) → pool → `Linear(D_plm→D)` (`Identity` if equal). |
| `user_encoder.py` | `UserEncoder` | Module 2. `encode_sequence(vecs[B,S,D], mask) → [B,S,D]`; `pool(states, mask) → [B,D]`; `forward(...) → (sequence, z_u)`. The pooler is reused as `self.pooler`. |
| `rec_model.py` | `NewsRecModel`, `build_rec_model(config)` | Two-tower. `encode_news(ids[...,L], attn) → [...,D]` (flattens leading dims, runs PLM→NewsEncoder, reshapes back). `encode_user(hist_ids[B,S,L], hist_attn, hist_mask[B,S]) → (states[B,S,D], z_u[B,D])`. `score(z_u[B,D], cand[B,K,D] or [B,D]) → [B,K] or [B]` (cosine/dot ÷ temperature). `model_dim` property. |
| `pretrain_model.py` | `PretrainModule` | Wraps a `NewsRecModel` + adds `category_embeddings: nn.Embedding(C, D)` and `mask_token: nn.Parameter(D)`. `compute_losses(batch) → {task: loss, ..., "total": ...}`. |

### `build_rec_model(config)` config keys (merged over `DEFAULT_MODEL_CONFIG`)
```
plm: {model_name, pretrained, use_lora, lora_r, lora_alpha, lora_dropout,
      lora_target_modules?, small_config?}
model_dim: 256
news_encoder: {num_layers, num_heads, dropout}
user_encoder: {num_layers, num_heads, dropout}
score: {type: cosine|dot, temperature}
max_title_len  (→ news Fastformer max position embeddings)
max_history_len(→ user Fastformer max position embeddings)
```
> ⚠️ The Fastformer position-embedding capacity is set from `max_title_len`
> (news) and `max_history_len` (user). If a sequence is longer than this
> capacity you get `IndexError: index out of range in self`. The evaluator caps
> eval history to the user encoder's capacity for safety.

---

## 5. `newsrec/losses/` — losses

### `infonce.py`
- `info_nce_inbatch(query[N,D], key[N,D], tau)` — symmetric in-batch InfoNCE
  (positives on the diagonal). L2-normalises inputs. Returns 0 if `N<2`.
- `info_nce_against_table(anchor[N,D], labels[N], table[C,D], tau)` —
  classify each anchor against a candidate table (used for category prediction).

### `pretrain_losses.py` (Stage 1)
Pure functions over encoded tensors, plus the task registry:
- `aap_loss(news_vecs, categories, category_table, tau)` — item-level, AAP.
- `mip_loss(context_states, target_vecs, tau)` — in-batch, MIP.
- `map_loss(context_states, categories, category_table, tau)` — MAP.
- `sp_loss(context_repr, segment_repr, tau)` — in-batch, SP.
- `bsm_loss(user_repr_a, user_repr_b, tau)` — in-batch, BSM.
- `PRETRAIN_TASKS = ["aap","mip","map","sp","bsm"]`.
- `select_enabled_tasks(config)` — accepts a **list** `["aap","mip"]` or a
  **mapping** `{"aap": {"enabled": true, "weight": 1.0}, ...}`; raises on
  unknown task names.
- `task_weights(config, enabled)` — per-task weights (default 1.0).

### `paac_losses.py` (Stage 2)
- `bpr_loss(pos_scores[B], neg_scores[B])` = `-logsigmoid(pos-neg).mean()` (L_rec).
- `l2_regularization(params)` — sum of squared norms of trainable params (L2 term).
- `supervised_alignment_loss(history_vecs[B,S,D], history_mask[B,S], history_pop[B,S], ratio)`
  — `L_sa`: per-user split top-`ratio` by popularity, mean pairwise L2 (cdist)
  between popular and unpopular **news vectors**, divided by `|I_u|`, averaged
  over users.
- `augment_views(h, dropout_p, noise_std)` — two augmented views (feature
  dropout + Gaussian noise).
- `reweighting_contrastive_loss(item_vecs[N,D], pop_values[N], x_percent, beta,
  gamma, tau, dropout_p, noise_std)` — `L_cl`: two views, top-`x%` popular split,
  β-reweighted InfoNCE with popular-anchor and unpopular-anchor terms (logsumexp
  with `+log(beta)` for the cross-group negatives), combined as
  `γ·L_pop + (1-γ)·L_unpop`.

---

## 6. `newsrec/eval/` — metrics & evaluation

### `metrics.py` (per-impression, then averaged; NaN impressions skipped)
- `auc_score(scores, labels)` — Mann-Whitney rank AUC with tie handling.
- `mrr_score(scores, labels)` — **official MIND MRR** = `Σ(y_i / rank_i) / Σy`
  (averages reciprocal rank over *all* positives; note 2 positives perfectly
  ranked → 0.75, not 1.0).
- `ndcg_score(scores, labels, k)` — nDCG@k.
- `compute_impression_metrics(scores_list, labels_list, metrics)` — averages,
  dropping impressions where a metric is undefined (e.g. all-pos/all-neg AUC).
  Metric keys: `"auc"`, `"mrr"`, `"ndcg@5"`, `"ndcg@10"`.

### `evaluator.py` — `ImpressionEvaluator(model, device)`
Two phases (mirrors representation caching):
1. `encode_news_table({nid: {input_ids, attention_mask}}, batch_size) → {nid: vec}`
   — runs PLM+NewsEncoder once over the catalogue.
2. `evaluate(impressions, news_vectors, max_history, batch_size,
   max_impressions, metrics)` — mini-batches the **user encoder + cosine
   scoring** using cached vectors, ranks within each impression, returns
   averaged metrics. The scoring path is decoupled from BERT so tests can inject
   arbitrary `news_vectors`.

---

## 7. `newsrec/training/` — trainers & checkpointing

| File | Symbol | Role |
|---|---|---|
| `lora_schedule.py` | `LoRAUnfreezeScheduler(plm, schedule=[[epoch, n_layers], ...])` | `.step(epoch) → (changed, n)` calls `plm.set_trainable_layers(n)` only when `n` changes. |
| `finetuner.py` | `Finetuner` | Stage 2. `compute_losses(batch)` builds `{L_rec, L_sa?, L_cl?, L_reg?, total}`; `_encode_triplet` returns `(z_u, pos_vec, neg_vec, history_vecs[B,S,D], history_mask)` (history_vecs are Module-1 outputs, needed by L_sa); `train_step`, `train(loader, dev_impressions, news_tokens, epochs)`, `evaluate(...)`, `load_pretrained(ckpt)`. Optimiser = Adam over **all** params (so gradual unfreeze needs no rebuild). |
| `pretrainer.py` | `Pretrainer(module, config, device, logger, checkpoint_manager)` | Stage 1. Joint weighted multi-task loop; saves per `save_every` with `metric=avg_loss` (`higher_is_better=False`). |
| `checkpoint.py` | `save_checkpoint`, `load_backbone_weights`, `CheckpointManager` | A checkpoint dir = `model.pt` (NewsRecModel state) + `config.yaml` + `lora/` (peft adapters) + `tokenizer/` + `extra.pt`. `load_backbone_weights(model, dir_or_file, strict=False)` loads shared Modules 0/1/2. `CheckpointManager.save(model, tag, extra_state, metric, higher_is_better)` writes `{local_dir}/{tag}` + enqueues HF upload + `maybe_save_best → {local_dir}/best`. |
| `hub_uploader.py` | `HubUploader(repo_id, token, private, enabled, logger, api)` | **Async** background-thread queue. `enabled` only if `repo_id` and a token (or injected `api`). Graceful fallback (logs, `enqueue→False`) when disabled. `enqueue(local_dir, path_in_repo)`, `close(wait=True)` flushes. Token resolved via `_resolve_token()` → `utils.env.get_hf_token`. |

### Config keys consumed by the trainers
`DEFAULT_FINETUNE_CONFIG`: `lr, weight_decay, grad_clip, epochs, log_every,
eval_every, max_eval_impressions, max_history, lambda1 (L_sa), lambda2 (L_cl),
lambda3 (L2), sa_ratio, cl_x_percent, cl_beta, cl_gamma, cl_tau, cl_dropout,
cl_noise`.
`DEFAULT_PRETRAIN_CONFIG`: `lr, weight_decay, grad_clip, epochs, log_every,
save_every`.

---

## 8. `newsrec/utils/` — cross-cutting

| File | Symbols |
|---|---|
| `config.py` | `Config` (attribute + dotted `get`/`set` + dict access), `load_config(path, overrides, cli_overrides)` (supports `_base_` inheritance + dotted `key=value` CLI overrides with scalar coercion), `save_config`, `deep_merge`. |
| `logging.py` | `setup_logger(name, log_dir, stage, run_name, level, ...)` → logger writing to console + `logs/{stage}_{run_name}_{timestamp}.log`; `format_metrics(dict, prefix)`; `build_log_path`. |
| `seed.py` | `set_seed(seed, deterministic)`. |
| `env.py` | `load_dotenv(path, override=False)` (minimal `KEY=VALUE` parser), `get_hf_token()` (checks `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`). |

---

## 9. `newsrec/scripts/` — python entry points

| Module | What it does |
|---|---|
| `prepare_data.py` | Loads MIND, prints stats + vocab sizes. |
| `common.py` | Shared builders: `build_logger`, `build_checkpoint_manager` (wires async `HubUploader` + namespaces ckpt dir by `run_name`), `load_news_and_tokens` (loads train+dev with optional `max_*_impressions` slicing, auto-download via `ensure_mind_split`, builds `NewsTokenTable`, `category_ids`, `ItemPopularity`; returns `(train, dev, news_tokens, category_ids, cat_vocab, popularity, encoder)`). |
| `run_pretrain.py` | `run_pretrain(cfg)` + `main`. Builds `PretrainDataset` + `PretrainModule` + `Pretrainer`; selects tasks from `pretrain.tasks`. |
| `run_finetune.py` | `run_finetune(cfg)` + `main`. Builds `FinetuneTripletDataset` + `Finetuner`; optional `LoRAUnfreezeScheduler` from `finetune.lora_schedule`; loads `finetune.pretrained_ckpt` if set; evaluates on dev. |
| `push_dataset.py` | Uploads MIND splits to a HF **dataset** repo (`train/`, `dev/` subfolders). |

### Config schema (composed via `_base_: ../base.yaml`)
```yaml
run_name: <unique per run; namespaces logs + checkpoints>
seed: 42
device: cpu|cuda
data:
  train_dir / dev_dir
  hf_dataset_repo / auto_download / train_subfolder / dev_subfolder
  max_title_len / max_history / min_seq_len / mask_prob
  max_train_impressions / max_dev_impressions   # optional slicing (smoke)
model: { ...build_rec_model keys... }
logging: { log_dir, level }
checkpoint: { dir }            # final path = {dir}/{run_name}/{stage}
hub: { push_to_hub, hub_repo_id, hub_private }
pretrain: { tau, tasks, training: {...DEFAULT_PRETRAIN_CONFIG...} }   # stage 1
finetune: { pretrained_ckpt, negatives_per_pos, batch_size, num_workers,
            lambda1/2/3, sa_*/cl_*, lr, epochs, eval_every, lora_schedule } # stage 2
```

---

## 10. Data & control flow per training step

**Pre-training step** (`PretrainModule.compute_losses`):
1. `h_seq = model.encode_news(input_ids, attention_mask)` → `[B,S,D]`.
2. AAP: `aap_loss(h_seq[valid], category[valid], category_table)`.
3. MIP/MAP: replace `mip_mask` positions in `h_seq` with `mask_token`,
   `ctx = user_encoder.encode_sequence(masked, seq_mask)`, read `ctx[mip_mask]`;
   MIP matches it to `h_seq[mip_mask]`, MAP classifies its category.
4. SP: context = pool over `context_mask` of a segment-masked sequence; segment
   = pool over `segment_mask` of the clean sequence; `sp_loss(context, segment)`.
5. BSM: pool the two `bsm_a/bsm_b` sub-sequences → `sp_loss`-style match.
6. `total = Σ weight[t] · loss[t]`.

**Fine-tuning step** (`Finetuner.compute_losses`):
1. `history_vecs = encode_news(history)`; `z_u = user_encoder(history_vecs, mask)[1]`.
2. `pos_vec`, `neg_vec = encode_news(pos/neg)`.
3. `L_rec = bpr_loss(score(z_u,pos_vec), score(z_u,neg_vec))`.
4. If `lambda1`: `L_sa(history_vecs, history_mask, history_pop)`.
5. If `lambda2`: `L_cl(pos_vec as batch items, pos_pop)`.
6. `total = L_rec + λ1·L_sa + λ2·L_cl + λ3·L_reg`.

---

## 11. Testing strategy

- All model/loss tests use a **tiny random BERT** (`pretrained=False` +
  `small_config`) so they run fast and offline.
- Real-data tests are guarded by data presence (`pytest.skip` otherwise).
- `tests/test_smoke.py::test_smoke_end_to_end` runs the full pipeline
  (pretrain → save → finetune-loading-checkpoint → dev eval) on a 300-impression
  slice with the tiny model — no GPU, no network.
- HF uploads are tested with an injected fake API (`_FakeApi`), never hitting
  the network.
- Run: `cd rec && python -m pytest` (≈101 tests).

---

## 12. Extension recipes

- **Add a pre-train task**: add a loss fn in `losses/pretrain_losses.py`, append
  to `PRETRAIN_TASKS`, emit any needed masks from `PretrainDataset`, and handle
  it in `PretrainModule.compute_losses`. Enable via `pretrain.tasks`.
- **Add a scenario**: drop a YAML in `config/pretrain/` (unique `run_name`!) and
  a `scripts/pretrain_<name>.sh` (copy an existing one).
- **Swap the PLM**: set `model.plm.model_name`; LoRA targets may need adjusting
  in `PLMEncoder.DEFAULT_LORA_TARGETS` for non-BERT architectures.
- **Change scoring**: `model.score.type` = `cosine` or `dot`; extend
  `NewsRecModel.score` for new variants.
- **New metric**: add to `eval/metrics.py::METRIC_FNS`.

---

## 13. Known gotchas

- Fastformer position capacity = `max_title_len` / `max_history_len`; exceeding
  it raises `IndexError: index out of range in self`.
- `peft` emits a benign `UserWarning` when saving LoRA adapters for a model with
  no tokenizer config — harmless.
- Checkpoints are namespaced `{checkpoint.dir}/{run_name}/{stage}`; **reusing a
  `run_name` overwrites** the previous run's checkpoints.
- The MIND MRR is the competition variant (`Σ y/rank ÷ Σ y`), not first-hit RR.
- `optimizer` is built over all params on purpose; frozen params simply receive
  no gradient until the LoRA scheduler unfreezes them.

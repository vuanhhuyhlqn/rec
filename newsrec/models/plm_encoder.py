"""
plm_encoder.py — Module 0 (word embedder)
==========================================

A pretrained language model (PLM) wrapped with LoRA adapters that turns news
``input_ids`` into contextual *word* embeddings ``[B, L, D]`` (the news encoder,
Module 1, pools these into a single news vector).

Supports BERT- and DistilBERT-family encoders; the LoRA target modules and the
layer-name marker used for gradual unfreezing are auto-detected from the model
architecture (so ``bert-base-uncased`` and ``distilbert-base-uncased`` both work
without manual wiring).

Key features
------------
* LoRA (via ``peft``) on the attention / FFN projection matrices.  By default
  the PLM backbone is frozen and only the LoRA adapters train.
* :meth:`set_trainable_layers` implements **gradual unfreezing**: it makes the
  *base* weights of the top-``n`` transformer layers trainable (LoRA adapters
  always stay trainable).  ``n = 0`` → LoRA-only; ``n = num_layers`` → full
  fine-tune.
* A ``pretrained=False`` path builds a small randomly-initialised BERT from an
  explicit config — used by the unit tests so they run fast and offline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn


# LoRA target projection names per architecture family.
DEFAULT_LORA_TARGETS = ["query", "key", "value", "dense"]            # BERT
DISTILBERT_LORA_TARGETS = ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]

# Marker substrings used to locate the transformer-layer index in a parameter
# name, per architecture family.
_LAYER_MARKERS = ("encoder.layer.", "transformer.layer.")


def _default_lora_targets(model_type: str) -> List[str]:
    if model_type == "distilbert":
        return DISTILBERT_LORA_TARGETS
    return DEFAULT_LORA_TARGETS


class PLMEncoder(nn.Module):
    """Pretrained LM (BERT / DistilBERT) + LoRA word embedder."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pretrained: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        small_config: Optional[dict] = None,
    ):
        super().__init__()
        from transformers import AutoModel, BertConfig, BertModel

        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            cfg_kwargs = dict(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=128,
                max_position_embeddings=128,
                vocab_size=30522,
            )
            if small_config:
                cfg_kwargs.update(small_config)
            self.bert = BertModel(BertConfig(**cfg_kwargs))

        self.hidden_size = self.bert.config.hidden_size
        self.num_layers = self.bert.config.num_hidden_layers
        self.use_lora = use_lora

        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            # Targets must match the *actual* built architecture (e.g. the
            # offline test path always builds a BertModel).
            model_type = getattr(self.bert.config, "model_type", "bert")
            targets = lora_target_modules or _default_lora_targets(model_type)
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                bias="none",
            )
            self.bert = get_peft_model(self.bert, lora_cfg)

        # Start frozen-backbone (LoRA-only). Gradual unfreeze opens base layers.
        self._frozen_layers = self.num_layers
        self.set_trainable_layers(0)

    # ------------------------------------------------------------------ #
    # Gradual unfreezing                                                 #
    # ------------------------------------------------------------------ #
    def _is_lora_param(self, name: str) -> bool:
        return "lora_" in name

    def _layer_index_of(self, name: str) -> Optional[int]:
        """Return the transformer-layer index encoded in a parameter name, else None.

        Handles both BERT (``encoder.layer.N``) and DistilBERT
        (``transformer.layer.N``) naming.
        """
        for marker in _LAYER_MARKERS:
            pos = name.find(marker)
            if pos == -1:
                continue
            rest = name[pos + len(marker):]
            num = rest.split(".", 1)[0]
            return int(num) if num.isdigit() else None
        return None

    def set_trainable_layers(self, n: int) -> int:
        """
        Make the top-``n`` BERT layers' *base* weights trainable.

        LoRA adapters (if present) are always trainable.  When LoRA is
        disabled, this directly controls which base layers train.

        Returns the number of currently-frozen layers.
        """
        n = max(0, min(self.num_layers, n))
        unfreeze_from = self.num_layers - n  # layers with index >= this unfreeze

        for name, param in self.bert.named_parameters():
            if self.use_lora and self._is_lora_param(name):
                param.requires_grad = True
                continue

            layer_idx = self._layer_index_of(name)
            if layer_idx is None:
                # Embeddings / pooler: trainable only on (near) full unfreeze.
                param.requires_grad = (not self.use_lora) and (n >= self.num_layers)
            else:
                param.requires_grad = layer_idx >= unfreeze_from

        self._frozen_layers = self.num_layers - n
        return self._frozen_layers

    @property
    def frozen_layers(self) -> int:
        return self._frozen_layers

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``[B, L]`` token ids → (``[B, L, D]`` embeddings, ``[B, L]`` mask)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask

    @property
    def output_dim(self) -> int:
        return self.hidden_size

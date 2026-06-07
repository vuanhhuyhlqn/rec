"""
fastformer.py
=============

A self-contained Fastformer encoder (Wu et al., 2021) adapted from the
Legommenders implementation.

The crucial difference from the upstream version is that
:class:`FastformerEncoder` returns the **full sequence of contextual hidden
states** ``[B, S, D]`` instead of a single pooled vector.  This is required so
that:

* Module 1 (news encoder) can pool word states however it likes, and
* the masked pre-training tasks (MIP / MAP) can read per-position states.

Pooling is provided separately by
:class:`newsrec.models.attention_pooler.AdditiveAttentionPooling`.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertOutput,
    BertSelfOutput,
)


class FastformerConfig:
    """Minimal config object compatible with the borrowed BERT sub-modules."""

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 8,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = "gelu",
        intermediate_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or hidden_size * 4


class FastSelfAttention(nn.Module):
    """Additive (linear-complexity) self attention from Fastformer."""

    def __init__(self, config: FastformerConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(_init_weights(config))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(hidden_states)

        # global query
        query_for_score = (
            self.query_att(mixed_query).transpose(1, 2) / self.attention_head_size ** 0.5
        )
        query_for_score = query_for_score + attention_mask
        query_weight = self.softmax(query_for_score).unsqueeze(2)
        query_layer = self.transpose_for_scores(mixed_query)
        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .view(-1, 1, self.num_attention_heads * self.attention_head_size)
        )
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)

        # global key
        mixed_query_key = mixed_key * pooled_query_repeat
        query_key_score = (
            self.key_att(mixed_query_key) / self.attention_head_size ** 0.5
        ).transpose(1, 2)
        query_key_score = query_key_score + attention_mask
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)
        key_layer = self.transpose_for_scores(mixed_query_key)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,)
        )
        weighted_value = self.transform(weighted_value) + mixed_query
        return weighted_value


class FastAttention(nn.Module):
    def __init__(self, config: FastformerConfig):
        super().__init__()
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self_output = self.self(input_tensor, attention_mask)
        return self.output(self_output, input_tensor)


class FastformerLayer(nn.Module):
    def __init__(self, config: FastformerConfig):
        super().__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class FastformerEncoder(nn.Module):
    """Stacked Fastformer layers returning per-position hidden states."""

    def __init__(self, config: FastformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FastformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(_init_weights(config))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, S, D]`` embeddings → ``[B, S, D]`` contextual states."""
        batch_size, seq_len, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        # [B, 1, S] additive mask in {0, -10000}
        ext_mask = attention_mask.unsqueeze(1).to(dtype=inputs_embeds.dtype)
        ext_mask = (1.0 - ext_mask) * -10000.0

        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        embeddings = inputs_embeds + self.position_embeddings(position_ids)
        embeddings = self.dropout(self.LayerNorm(embeddings))

        hidden = embeddings
        for layer in self.layers:
            hidden = layer(hidden, ext_mask)
        return hidden


def _init_weights(config: FastformerConfig):
    def fn(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    return fn

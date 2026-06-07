"""
news_tokens.py
==============

:class:`NewsTokenTable` maps each ``news_id`` to its tokenised
``title + abstract`` (``input_ids`` / ``attention_mask``).

It is built once from a parsed news dict + a
:class:`~newsrec.data.vocab.NewsTextEncoder` and then consumed by the
fine-tune and pre-train datasets.  Datasets only need a mapping-like object
exposing ``get(nid)`` and ``max_len``, so unit tests can substitute a plain
synthetic dict without loading a tokenizer.
"""

from __future__ import annotations

from typing import Dict, List, Mapping


class NewsTokenTable:
    """``{news_id: {input_ids, attention_mask}}`` with a zero PAD entry."""

    def __init__(self, tokens: Dict[str, Dict[str, List[int]]], max_len: int):
        self.tokens = tokens
        self.max_len = max_len
        self.pad = {
            "input_ids": [0] * max_len,
            "attention_mask": [0] * max_len,
        }

    @classmethod
    def build(cls, news: Mapping[str, object], encoder, batch_size: int = 256) -> "NewsTokenTable":
        """Encode every news item's text with a :class:`NewsTextEncoder`."""
        nids = list(news.keys())
        tokens: Dict[str, Dict[str, List[int]]] = {}
        for start in range(0, len(nids), batch_size):
            chunk = nids[start:start + batch_size]
            texts = [news[n].text for n in chunk]  # type: ignore[attr-defined]
            out = encoder.encode_batch(texts)
            for i, nid in enumerate(chunk):
                tokens[nid] = {
                    "input_ids": out["input_ids"][i],
                    "attention_mask": out["attention_mask"][i],
                }
        return cls(tokens, encoder.max_len)

    def get(self, nid: str) -> Dict[str, List[int]]:
        return self.tokens.get(nid, self.pad)

    def has(self, nid: str) -> bool:
        return nid in self.tokens

    def __contains__(self, nid: str) -> bool:
        return nid in self.tokens

    def __len__(self) -> int:
        return len(self.tokens)

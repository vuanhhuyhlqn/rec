"""
vocab.py
========

* :class:`Vocab` — a generic token <-> index map with configurable special
  tokens (PAD / UNK / MASK), used for news ids and categories.
* :func:`build_news_vocab` / :func:`build_category_vocab` — convenience
  builders over parsed :class:`~newsrec.data.mind_parser.MindData`.
* :class:`NewsTextEncoder` — wraps a HuggingFace tokenizer to turn news
  ``title + abstract`` into padded ``input_ids`` / ``attention_mask``.
  Loading the HF tokenizer is deferred to construction so the rest of the
  data stack stays import-safe (and unit-testable) without network access.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"


class Vocab:
    """Bidirectional token/index map with reserved special tokens."""

    def __init__(self, name: str, special_tokens: Sequence[str] = (PAD_TOKEN,)):
        self.name = name
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self.special_tokens = list(special_tokens)
        for token in self.special_tokens:
            self._add(token)

    # ---- construction ------------------------------------------------------ #
    def _add(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)
        return self.token2id[token]

    def add(self, token: str) -> int:
        return self._add(token)

    def build(self, tokens: Iterable[str]) -> "Vocab":
        for token in tokens:
            self._add(token)
        return self

    # ---- lookup ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.id2token)

    def __contains__(self, token: str) -> bool:
        return token in self.token2id

    def index(self, token: str) -> int:
        """Token -> id; falls back to UNK id when available, else raises."""
        if token in self.token2id:
            return self.token2id[token]
        if UNK_TOKEN in self.token2id:
            return self.token2id[UNK_TOKEN]
        raise KeyError(f"Token '{token}' not in vocab '{self.name}' and no [UNK].")

    def token(self, idx: int) -> str:
        return self.id2token[idx]

    # ---- reserved-token ids ------------------------------------------------ #
    @property
    def pad_id(self) -> int:
        return self.token2id.get(PAD_TOKEN, 0)

    @property
    def unk_id(self) -> Optional[int]:
        return self.token2id.get(UNK_TOKEN)

    @property
    def mask_id(self) -> Optional[int]:
        return self.token2id.get(MASK_TOKEN)

    # ---- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"name": self.name, "special_tokens": self.special_tokens,
                 "id2token": self.id2token},
                handle,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        vocab = cls(data["name"], special_tokens=())
        vocab.special_tokens = data["special_tokens"]
        vocab.id2token = list(data["id2token"])
        vocab.token2id = {tok: i for i, tok in enumerate(vocab.id2token)}
        return vocab


# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #
def build_news_vocab(
    news: Mapping[str, object],
    *extra_news: Mapping[str, object],
) -> Vocab:
    """News-id vocab with PAD / UNK / MASK reserved (ids 0 / 1 / 2)."""
    vocab = Vocab("news", special_tokens=(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN))
    for nid in news.keys():
        vocab.add(nid)
    for extra in extra_news:
        for nid in extra.keys():
            vocab.add(nid)
    return vocab


def build_category_vocab(
    news: Mapping[str, object],
    *extra_news: Mapping[str, object],
) -> Vocab:
    """Category vocab with PAD / UNK reserved (ids 0 / 1)."""
    vocab = Vocab("category", special_tokens=(PAD_TOKEN, UNK_TOKEN))
    cats = set()
    for item in news.values():
        cats.add(item.category)  # type: ignore[attr-defined]
    for extra in extra_news:
        for item in extra.values():
            cats.add(item.category)  # type: ignore[attr-defined]
    for cat in sorted(cats):
        vocab.add(cat)
    return vocab


# --------------------------------------------------------------------------- #
# Text encoder                                                                #
# --------------------------------------------------------------------------- #
class NewsTextEncoder:
    """Tokenise news text with a HuggingFace tokenizer to fixed-length ids."""

    def __init__(self, tokenizer_name: str = "distilbert-base-uncased", max_len: int = 64):
        from transformers import AutoTokenizer

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, text: str) -> Dict[str, List[int]]:
        out = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    def encode_batch(self, texts: Sequence[str]) -> Dict[str, List[List[int]]]:
        out = self.tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

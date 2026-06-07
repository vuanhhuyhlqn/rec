"""
prepare_data.py
===============

Demo / sanity script for Task 1.

Loads a MIND split, builds the news- and category-vocabs, and prints summary
statistics.  Defaults point at the bundled ``MINDsmall_train`` directory.

Usage::

    python -m newsrec.scripts.prepare_data --train rec/MINDsmall_train \\
        --dev rec/MINDsmall_dev
"""

from __future__ import annotations

import argparse
import os

from newsrec.data.download import ensure_mind_split
from newsrec.data.mind_parser import load_mind_split
from newsrec.data.vocab import build_category_vocab, build_news_vocab
from newsrec.utils.env import get_hf_token, load_dotenv


def _default(path: str) -> str:
    # Resolve relative to repo root (the directory containing this package).
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a MIND dataset split.")
    parser.add_argument("--train", default=_default("dataset/train"))
    parser.add_argument("--dev", default=_default("dataset/dev"))
    parser.add_argument("--repo", default="huyva/mind-small",
                        help="HF dataset repo to download from when a split is missing")
    parser.add_argument("--download-dir", default=_default("dataset"))
    parser.add_argument("--no-download", action="store_true",
                        help="do not auto-download; use local paths only")
    args = parser.parse_args()

    # Auto-download missing splits (mirrors the training entry points) so this
    # works on a fresh machine / `setup.sh --with-data`.
    if not args.no_download:
        load_dotenv(".env")
        token = get_hf_token()
        args.train = ensure_mind_split(args.train, args.repo, "train", True, token,
                                       download_dir=args.download_dir)
        args.dev = ensure_mind_split(args.dev, args.repo, "dev", True, token,
                                     download_dir=args.download_dir)

    print(f"Loading train split from: {args.train}")
    train = load_mind_split(args.train)
    print("Train statistics:")
    for key, value in train.stats().items():
        print(f"  {key:>20}: {value}")

    dev = None
    if os.path.isdir(args.dev):
        print(f"\nLoading dev split from: {args.dev}")
        dev = load_mind_split(args.dev)
        print("Dev statistics:")
        for key, value in dev.stats().items():
            print(f"  {key:>20}: {value}")

    extra = [dev.news] if dev is not None else []
    news_vocab = build_news_vocab(train.news, *extra)
    cat_vocab = build_category_vocab(train.news, *extra)

    print("\nVocab sizes (incl. special tokens):")
    print(f"  news vocab    : {len(news_vocab)} (PAD/UNK/MASK reserved)")
    print(f"  category vocab: {len(cat_vocab)} (PAD/UNK reserved)")
    print(f"  categories    : {train.categories}")


if __name__ == "__main__":
    main()

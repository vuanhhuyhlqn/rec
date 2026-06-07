"""
push_dataset.py
===============

Upload the local MIND splits to a HuggingFace **dataset** repo so they can be
auto-downloaded on other machines (see :mod:`newsrec.data.download`).

Repo layout produced::

    <repo_id>/
        train/ news.tsv behaviors.tsv [popularity_data.csv] [*.vec]
        dev/   news.tsv behaviors.tsv

Usage::

    python -m newsrec.scripts.push_dataset \\
        --train MINDsmall_train --dev MINDsmall_dev \\
        [--repo-id user/mind-small] [--private] [--include-embeddings]

The token is read from .env / HF_TOKEN / HUGGINGFACE_TOKEN.  If ``--repo-id``
is omitted it defaults to ``<your-hf-username>/mind-small``.
"""

from __future__ import annotations

import argparse
import os

from newsrec.utils.env import get_hf_token, load_dotenv

# Files uploaded per split (core MIND recommender inputs).
CORE_FILES = ["news.tsv", "behaviors.tsv", "popularity_data.csv"]
EMBEDDING_FILES = ["entity_embedding.vec", "relation_embedding.vec"]


def _upload_split(api, repo_id: str, local_dir: str, subfolder: str, include_embeddings: bool):
    from huggingface_hub import CommitOperationAdd

    files = list(CORE_FILES)
    if include_embeddings:
        files += EMBEDDING_FILES

    operations = []
    for fname in files:
        fpath = os.path.join(local_dir, fname)
        if os.path.exists(fpath):
            operations.append(
                CommitOperationAdd(path_in_repo=f"{subfolder}/{fname}", path_or_fileobj=fpath)
            )
    if not operations:
        print(f"  [skip] no files found in {local_dir}")
        return
    print(f"  uploading {len(operations)} file(s) -> {repo_id}/{subfolder}/ ...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add MIND {subfolder} split",
    )


def main():
    parser = argparse.ArgumentParser(description="Push MIND splits to the HF Hub.")
    parser.add_argument("--train", default="dataset/train")
    parser.add_argument("--dev", default="dataset/dev")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--include-embeddings", action="store_true",
                        help="also upload entity/relation .vec files (~26MB)")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    token = get_hf_token()
    if not token:
        raise SystemExit("No HF token found (set HUGGINGFACE_TOKEN/HF_TOKEN or .env).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = args.repo_id or f"{username}/mind-small"
    print(f"Authenticated as: {username}")
    print(f"Target dataset repo: {repo_id} (private={args.private})")

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print("Train split:")
    _upload_split(api, repo_id, args.train, "train", args.include_embeddings)
    if os.path.isdir(args.dev):
        print("Dev split:")
        _upload_split(api, repo_id, args.dev, "dev", args.include_embeddings)

    print(f"\nDone. Set this in your config to enable auto-download:\n"
          f"  data:\n    hf_dataset_repo: {repo_id}\n    auto_download: true")


if __name__ == "__main__":
    main()

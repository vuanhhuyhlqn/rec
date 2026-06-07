"""
hub_uploader.py
===============

Asynchronous HuggingFace Hub uploader.

Checkpoint uploads run on a **background thread** consuming a queue, so the
training loop is never blocked on network I/O.  If uploading is disabled or no
token is available, the uploader degrades gracefully to a no-op (with a logged
warning) — this keeps the offline smoke test free of any network dependency.

The HfApi object can be injected (``api=...``) so unit tests can assert the
upload calls without touching the network.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class _UploadJob:
    local_dir: str
    path_in_repo: str


def _resolve_token() -> Optional[str]:
    """Resolve the HF token, accepting HF_TOKEN / HUGGINGFACE_TOKEN."""
    try:
        from newsrec.utils.env import get_hf_token

        return get_hf_token()
    except Exception:  # pragma: no cover - defensive
        return os.environ.get("HF_TOKEN")


class HubUploader:
    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False,
        enabled: bool = True,
        logger=None,
        api=None,
    ):
        self.repo_id = repo_id
        self.private = private
        self.logger = logger
        self._queue: "queue.Queue[Optional[_UploadJob]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._repo_ready = False

        token = token or _resolve_token()
        self.enabled = bool(enabled and repo_id and (token or api is not None))
        if enabled and not self.enabled:
            self._log(
                "HF upload requested but disabled: missing repo_id or HF_TOKEN. "
                "Falling back to local-only checkpointing."
            )

        self._api = api
        if self.enabled and self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=token)

    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _ensure_repo(self) -> None:
        if self._repo_ready:
            return
        try:
            self._api.create_repo(repo_id=self.repo_id, private=self.private, exist_ok=True)
        except Exception as exc:  # pragma: no cover - network/credential issues
            self._log(f"create_repo failed (continuing): {exc}")
        self._repo_ready = True

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                self._ensure_repo()
                self._api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=job.local_dir,
                    path_in_repo=job.path_in_repo,
                )
                self._log(f"Uploaded {job.local_dir} -> {self.repo_id}/{job.path_in_repo}")
            except Exception as exc:  # pragma: no cover - network/credential issues
                self._log(f"Upload failed for {job.path_in_repo}: {exc}")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    def enqueue(self, local_dir: str, path_in_repo: str) -> bool:
        """Queue a folder for async upload.  Returns False if disabled."""
        if not self.enabled:
            return False
        if self._thread is None:
            self.start()
        self._queue.put(_UploadJob(local_dir=local_dir, path_in_repo=path_in_repo))
        return True

    def close(self, wait: bool = True) -> None:
        """Flush the queue and stop the worker thread."""
        if not self.enabled or self._thread is None:
            return
        self._queue.put(None)
        if wait:
            self._thread.join()
        self._thread = None

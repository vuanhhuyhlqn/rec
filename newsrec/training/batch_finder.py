"""
batch_finder.py
===============

Auto-detect the largest training batch size that fits in GPU memory without
triggering a CUDA out-of-memory error.

The memory bottleneck of this model is encoding ``batch_size * max_history``
news articles through BERT each step, so the right batch size is very
device/config dependent. :func:`find_max_batch_size` probes a real
forward+backward step (the backward pass is where activation memory peaks) at
exponentially increasing batch sizes, then binary-refines between the largest
size that fit and the first that OOMed, and finally applies a ``safety``
multiplier for headroom (memory fragmentation + optimizer state).

It is a no-op (returns ``None``) on non-CUDA devices: "max batch size" only has
a well-defined, catchable meaning for GPU OOM. Host-RAM OOM on CPU is a hard
SIGKILL we cannot probe safely.
"""

from __future__ import annotations

import gc
from typing import Callable, Dict, Optional

import torch


def search_largest_fit(
    can_fit: Callable[[int], bool],
    min_batch: int = 1,
    max_batch: int = 256,
    logger=None,
) -> int:
    """
    Return the largest ``b`` in ``[min_batch, max_batch]`` for which
    ``can_fit(b)`` is True, assuming ``can_fit`` is monotonic (if ``b`` fits,
    every smaller size fits too).

    Strategy: exponential growth (min_batch, 2x, 4x, ...) to bracket the first
    failure, then binary search to refine. Pure / side-effect free, so it is
    unit-testable without a GPU.
    """
    bs = max(1, min_batch)
    last_ok = 0
    first_fail: Optional[int] = None
    while bs <= max_batch:
        if logger:
            logger.info(f"[batch-finder] trying batch_size={bs} ...")
        if can_fit(bs):
            last_ok = bs
            if bs == max_batch:
                break
            bs = min(bs * 2, max_batch)
        else:
            first_fail = bs
            break

    if last_ok == 0:
        raise RuntimeError(
            f"[batch-finder] batch_size={min_batch} does not fit; "
            "reduce data.max_history / model.max_title_len or model_dim."
        )

    if first_fail is not None and first_fail - last_ok > 1:
        lo, hi = last_ok, first_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if logger:
                logger.info(f"[batch-finder] refining batch_size={mid} ...")
            if can_fit(mid):
                lo = mid
            else:
                hi = mid
        last_ok = lo
    return last_ok


def find_max_batch_size(
    build_batch: Callable[[int], Dict[str, torch.Tensor]],
    probe_step: Callable[[Dict[str, torch.Tensor]], None],
    *,
    device,
    min_batch: int = 1,
    max_batch: int = 256,
    safety: float = 0.95,
    logger=None,
) -> Optional[int]:
    """
    Return the largest batch size (after a ``safety`` margin) that runs one
    fwd+bwd step without CUDA OOM, or ``None`` if ``device`` is not CUDA.

    Parameters
    ----------
    build_batch : callable(bs) -> batch dict
        Produce a representative (collated) batch of the given size.
    probe_step : callable(batch) -> None
        Run a single forward + backward on the batch. Must clear gradients
        afterwards and must NOT call ``optimizer.step`` (so probing does not
        corrupt model weights).
    min_batch, max_batch : int
        Search bounds (inclusive). ``max_batch`` should already be capped to the
        dataset size by the caller.
    safety : float
        Multiplier applied to the largest fitting size for headroom.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return None

    def _can_fit(bs: int) -> bool:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = None
        try:
            batch = build_batch(bs)
            probe_step(batch)
            torch.cuda.synchronize()
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            del batch
            gc.collect()
            torch.cuda.empty_cache()

    last_ok = search_largest_fit(_can_fit, min_batch, max_batch, logger)
    final = max(min_batch, int(last_ok * safety))
    if logger:
        logger.info(
            f"[batch-finder] largest fitting batch_size={last_ok}; "
            f"using {final} after safety={safety}"
        )
    gc.collect()
    torch.cuda.empty_cache()
    return final


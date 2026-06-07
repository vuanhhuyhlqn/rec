"""
lora_schedule.py
================

Gradual unfreezing schedule for the BERT backbone during fine-tuning.

The schedule maps an *epoch* to the number of top BERT layers whose base
weights should be trainable (LoRA adapters are always trainable).  At each
epoch boundary the trainer calls :meth:`LoRAUnfreezeScheduler.step` which
applies the appropriate ``set_trainable_layers`` to the PLM encoder and
reports whether the number of open layers changed (so the trainer can log it).

Example schedule (list of ``[epoch, num_layers]``)::

    [[0, 0], [2, 2], [4, 6], [6, 12]]

→ epochs 0-1: LoRA only; 2-3: top-2 layers; 4-5: top-6; 6+: all 12.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


class LoRAUnfreezeScheduler:
    def __init__(self, plm_encoder, schedule: Sequence[Sequence[int]] | None = None):
        self.plm = plm_encoder
        # Default: LoRA-only the whole time.
        sched = schedule if schedule else [[0, 0]]
        # Normalise to sorted list of (epoch, n_layers).
        self.schedule: List[Tuple[int, int]] = sorted(
            (int(e), int(n)) for e, n in sched
        )
        self._current_n: int | None = None

    def num_layers_for_epoch(self, epoch: int) -> int:
        n = self.schedule[0][1]
        for sched_epoch, sched_n in self.schedule:
            if epoch >= sched_epoch:
                n = sched_n
            else:
                break
        return n

    def step(self, epoch: int) -> Tuple[bool, int]:
        """
        Apply the schedule for ``epoch``.

        Returns ``(changed, num_open_layers)`` where ``changed`` is ``True``
        when the number of unfrozen layers differs from the previous step.
        """
        n = self.num_layers_for_epoch(epoch)
        changed = n != self._current_n
        if changed:
            self.plm.set_trainable_layers(n)
            self._current_n = n
        return changed, n

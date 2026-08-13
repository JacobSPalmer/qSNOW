"""Reusable sinter sampling for experiments.

Sampling a large batch of circuits well takes more than one ``sinter.collect``
call. This module owns that policy so experiments only have to supply tasks rather
than worry about the nuances of optimizing the sampling process.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import sinter

from qsnow.experiments.progress import PhasedProgress

logger = logging.getLogger(__name__)

# Default number of tasks handed to a single `sinter.collect` call. Sinter recomputes an O(n) status update
# (where n is number of circuits to sample) each time a task is completed, causing a pretty nasty scaling as
# chip sizes get larger. Batching it caps that recompute to O(batch_size), which drastically increases the
# performance on large-scale simulations. There's probably a more optimal batch size, but with some basic testing
# it seems that ~1024 is the sweet spot.
DEFAULT_SAMPLING_BATCH_SIZE = 1024

# The top-up pass needs a shot ceiling (sinter has no unbounded mode as `max_shots`
# is mandatory with no equivalent `min_shots`), so by default it scales with the caller's baseline shot count.
DEFAULT_TOPUP_SHOT_MULTIPLIER = 20


def _chunked(items: Sequence, size: int) -> Iterator[List]:
    """Yield `items` in consecutive lists of at most `size` (size >= 1)."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _metadata_key(metadata: Any) -> str:
    """A stable, hashable, controlled identity for a task, derived from its ``json_metadata``.

    Stats cannot be matched back to their tasks by ``strong_id`` in all sc
    """
    return json.dumps(metadata, sort_keys=True, default=str)


class _TaskProgress:
    """Advances a progress phase in fractions of a task as shots stream in.

    A phase counts tasks, but a single batch of long-running circuits can take hours,
    leaving the bar frozen between batch completions. sinter reports partial
    statistics many times per task, so each task can instead earn credit in
    proportion to the shots it has taken, and is trued up to a whole task the moment
    it completes -- which may be early, when `max_errors` is hit before `max_shots`.

    Credit only ever moves forward: partial updates are capped at one whole task, so
    the bar can never overshoot its total.
    """

    def __init__(self, phase, shots_per_task: int):
        self._phase = phase
        self._shots_per_task = max(shots_per_task, 1)
        self._shots: Dict[str, int] = {}
        self._credited: Dict[str, float] = {}

    def on_progress(self, update: "sinter.Progress") -> None:
        """`progress_callback` for `sinter.collect`; new_stats are shot deltas."""
        for stat in update.new_stats:
            key = _metadata_key(stat.json_metadata)
            self._shots[key] = self._shots.get(key, 0) + stat.shots
            self._credit(key, min(1.0, self._shots[key] / self._shots_per_task))

    def complete(self, key: str) -> None:
        self._credit(key, 1.0)

    def finish(self, completed: float) -> None:
        """Snap the phase to an exact count.

        Credit accumulates as a sum of floats, which lands a hair under a whole
        number of tasks (5.999999999999999 for six). `_Row.finished` compares
        against the integer total, so without this the phase reads as unfinished
        and gets its total rewritten to the short value.
        """
        if self._phase is not None:
            self._phase.update(completed=completed)

    def _credit(self, key: str, fraction: float) -> None:
        earned = fraction - self._credited.get(key, 0.0)
        if earned <= 0:
            return
        self._credited[key] = fraction
        if self._phase is not None:
            self._phase.advance(earned)


@dataclass(frozen=True)
class SamplingRun:
    """The outcome of one `ErrorFloorSampler.collect`.

    `stalled` holds the `json_metadata` of every task still at zero errors after
    the top-up ceiling was spent.
    """

    stats: List[sinter.TaskStats]
    timings: Dict[str, float] = field(default_factory=dict)
    stalled: List[Any] = field(default_factory=list)


@dataclass
class ErrorFloorSampler:
    """Samples sinter tasks in batches, then tops up any that saw too few errors.

    A single `sinter.collect` stops at whichever of `max_shots`/`max_errors``
    comes first, so a quiet circuit can spend its whole shot budget and still record
    zero errors -- reported as a logical error rate of exactly 0. This sampler runs a
    second pass over exactly those tasks, sampling until each has seen at least
    `min_errors` errors (or until `max_topup_shots` is exhausted).

    `collect_fn` is injected so the logic can be tested without
    spawning worker processes.
    """

    decoder: str = "pymatching"
    shots: int = 50_000
    max_errors: Optional[int] = 5_000
    min_errors: int = 1
    max_topup_shots: Optional[int] = None
    batch_size: int = DEFAULT_SAMPLING_BATCH_SIZE
    max_workers: int = 32
    collect_fn: Callable[..., List[sinter.TaskStats]] = sinter.collect

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch size must be >= 1, got {self.batch_size}")
        if self.shots < 1:
            raise ValueError(f"shots must be >= 1, got {self.shots}")
        if self.min_errors < 0:
            raise ValueError(f"min_errors must be >= 0, got {self.min_errors}")
        if self.max_topup_shots is not None and self.max_topup_shots < 1:
            raise ValueError(
                f"max_topup_shots must be >= 1, got {self.max_topup_shots}"
            )

    # ------------------------------------------------------------------
    # Derived settings
    # ------------------------------------------------------------------

    @property
    def num_workers(self) -> int:
        return min(self.max_workers, os.cpu_count() or 1)

    @property
    def tops_up(self) -> bool:
        return self.min_errors > 0

    @property
    def topup_shot_cap(self) -> int:
        """Shot ceiling for the top-up pass; sinter requires one on every call."""
        if self.max_topup_shots is not None:
            return self.max_topup_shots
        return self.shots * DEFAULT_TOPUP_SHOT_MULTIPLIER

    @property
    def phases(self) -> int:
        """Number of progress phases `collect` will open."""
        return 2 if self.tops_up else 1

    def run_config(self) -> Dict[str, Any]:
        """The run parameters, for recording in an experiment's config."""
        return {
            "shots": self.shots,
            "max_errors": self.max_errors,
            "min_errors": self.min_errors,
            "max_topup_shots": self.topup_shot_cap if self.tops_up else None,
            "decoder": self.decoder,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        tasks: Sequence[sinter.Task],
        progress: Optional[PhasedProgress] = None,
    ) -> SamplingRun:
        """Sample every task, then top up the ones below the error floor."""
        keys = [_metadata_key(t.json_metadata) for t in tasks]
        self._reject_duplicates(tasks, keys)

        t_start = perf_counter()
        stats_by_key = self._sample(
            tasks,
            max_shots=self.shots,
            max_errors=self.max_errors,
            description="Sampling circuits",
            progress=progress,
        )
        timings = {"sampling": perf_counter() - t_start}

        if self.tops_up:
            needy = [
                task
                for task, key in zip(tasks, keys)
                if stats_by_key[key].errors < self.min_errors
            ]
            t_start = perf_counter()
            topped_up = self._sample(
                needy,
                max_shots=self.topup_shot_cap,
                max_errors=self.min_errors,
                description="Topping up low error circuits",
                progress=progress,
                # a top-up task stops at the first error, not at the shot ceiling its
                # progress is measured against, so credit arrives in coarse jumps and
                # an ETA extrapolated from it would swing wildly
                show_eta=False,
            )
            timings["topup"] = perf_counter() - t_start
            # TaskStats.__add__ validates that the strong ids match, so this
            # doubles as a check that the top-up hit the same tasks.
            for key, extra in topped_up.items():
                stats_by_key[key] = stats_by_key[key] + extra

        stalled = [s.json_metadata for s in stats_by_key.values() if s.errors == 0]
        if stalled:
            logger.warning(
                f"{len(stalled)} of {len(tasks)} tasks recorded no errors after "
                f"{self.topup_shot_cap} top-up shots; their error rates are upper "
                "bounds, not measurements."
            )

        return SamplingRun(
            stats=list(stats_by_key.values()), timings=timings, stalled=stalled
        )

    def _sample(
        self,
        tasks: Sequence[sinter.Task],
        *,
        max_shots: int,
        max_errors: Optional[int],
        description: str,
        progress: Optional[PhasedProgress],
        show_eta: bool = True,
    ) -> Dict[str, sinter.TaskStats]:
        """Run one batched collection pass, keyed by task metadata."""
        # No phase is opened for an empty pass: a bar that can only ever read 0/0
        # is noise, and the common case is that nothing needs topping up at all.
        if not tasks:
            return {}

        stats_by_key: Dict[str, sinter.TaskStats] = {}
        with self._phase(progress, description, total=len(tasks), show_eta=show_eta) as phase:
            tracker = _TaskProgress(phase, max_shots)
            for batch in _chunked(tasks, self.batch_size):
                batch_stats = self.collect_fn(
                    num_workers=self.num_workers,
                    tasks=batch,
                    decoders=[self.decoder],
                    max_shots=max_shots,
                    max_errors=max_errors,
                    # streams partial results so the bar moves *within* a batch,
                    # which may otherwise run for hours
                    progress_callback=tracker.on_progress if phase is not None else None,
                )
                for stat in batch_stats:
                    key = _metadata_key(stat.json_metadata)
                    stats_by_key[key] = stat
                    tracker.complete(key)
            tracker.finish(len(stats_by_key))
        return stats_by_key

    @staticmethod
    def _phase(
        progress: Optional[PhasedProgress],
        description: str,
        total: int,
        show_eta: bool = True,
    ):
        """Open a progress phase, or a no-op context when there is no display."""
        from contextlib import nullcontext

        if progress is None:
            return nullcontext(None)
        return progress.phase(description, total=total, show_eta=show_eta)

    @staticmethod
    def _reject_duplicates(tasks: Sequence[sinter.Task], keys: Sequence[str]) -> None:
        seen: Dict[str, int] = {}
        for i, key in enumerate(keys):
            if key in seen:
                raise ValueError(
                    "every task needs unique json_metadata to be matched with its "
                    f"stats, but tasks {seen[key]} and {i} share "
                    f"{tasks[i].json_metadata!r}"
                )
            seen[key] = i

"""Shared rich-progress plumbing for experiments.

`PhasedProgress` renders a single live display with one overall row plus one
row per sequential phase::

    Experiment: SquarePackingExp ━━━━━━╸━━━━━  50% / 0:00:12
      Generating circuits       ━━━━━━━━━━━━ 100% / 0:00:03
      Sampling circuits         ━━━╸━━━━━━━━  30% / 0:01:40

Phase rows are added lazily as each phase begins, so an experiment declares
only how many phases it has (for the overall bar) and opens them one at a
time with `phase()` or `track()`.
"""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")


class PhasedTimeColumn(ProgressColumn):
    """Elapsed time for the overall row, remaining→elapsed for phase rows.

    The overall row only advances once per phase, so rich can never form a
    speed estimate for it and a remaining-time column would render `-:--:--`
    for the whole run; its elapsed time (the running sum of all phases) is
    the meaningful number. Rows are told apart via the `overall` task field.
    """

    def __init__(self) -> None:
        self._elapsed = TimeElapsedColumn()
        self._remaining = TimeRemainingColumn(elapsed_when_finished=True)
        super().__init__()

    def render(self, task: Task):
        if task.fields.get("overall"):
            return self._elapsed.render(task)
        return self._remaining.render(task)


def default_columns() -> Sequence[ProgressColumn]:
    """Column set shared by all experiment progress displays."""
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("/"),
        PhasedTimeColumn(),
    )


class Phase:
    """Handle for advancing a single phase row (e.g. from a worker callback)."""

    def __init__(self, progress: Progress, task_id: TaskID):
        self._progress = progress
        self.task_id = task_id

    def advance(self, step: float = 1.0) -> None:
        self._progress.update(self.task_id, advance=step)

    def update(self, **kwargs: Any) -> None:
        self._progress.update(self.task_id, **kwargs)


class PhasedProgress:
    """A live progress display for a job made of sequential sub-phases.

    Use as a context manager; open each phase with `phase()` (or `track()` to
    wrap an iterable). Completing a phase advances the overall row by one.
    A phase that stops early with no error (e.g. sinter hitting `max_errors`
    before `max_shots`) is shrunk to its completed count so it renders as
    finished rather than stalled.
    """

    def __init__(
        self,
        title: str,
        phases: Optional[int] = None,
        columns: Optional[Sequence[ProgressColumn]] = None,
        **progress_kwargs: Any,
    ):
        self._progress = Progress(*(columns or default_columns()), **progress_kwargs)
        self._title = title
        self._phases = phases
        self._overall: Optional[TaskID] = None

    def __enter__(self) -> "PhasedProgress":
        self._progress.start()
        self._overall = self._progress.add_task(
            f"[bold]{self._title}", total=self._phases, overall=True
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None and self._overall is not None:
            (overall,) = (t for t in self._progress.tasks if t.id == self._overall)
            if not overall.finished:
                self._progress.update(self._overall, total=overall.completed)
        # Jupyter's Live.stop() skips the final refresh, which would leave the
        # last frame stale (e.g. the overall row's final advance never shown)
        self._progress.refresh()
        self._progress.stop()
        return False

    @property
    def tasks(self):
        """Snapshot of all task rows (overall first), mainly for inspection."""
        return self._progress.tasks

    @contextmanager
    def phase(self, description: str, total: Optional[float] = None) -> Iterator[Phase]:
        """Open the next sequential phase as a new indented row."""
        task_id = self._progress.add_task(f"  {description}", total=total)
        yield Phase(self._progress, task_id)
        (task,) = (t for t in self._progress.tasks if t.id == task_id)
        if not task.finished:
            self._progress.update(task_id, total=task.completed)
        if self._overall is not None:
            self._progress.update(self._overall, advance=1)

    def track(
        self,
        iterable: Iterable[T],
        description: str,
        total: Optional[float] = None,
    ) -> Iterator[T]:
        """Iterate `iterable` as a phase of its own, advancing once per item."""
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        with self.phase(description, total=total) as ph:
            for item in iterable:
                yield item
                ph.advance()

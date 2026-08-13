"""Progress display for experiments, backed by tqdm.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

# tqdm's default bar layout, minus the ``<{remaining}`` ETA estimate. Used for a
# phase opened with ``show_eta=False`` (one whose completion advances in coarse,
# bursty steps that would make a remaining-time estimate jump around).
# A phase may advance in fractions of an item (e.g. a task credited by the shots it
# has taken so far), and tqdm's `{n_fmt}` renders those at full float precision --
# "24.05666666666667/41". Formatting the count explicitly keeps the bar readable.
_COUNT = "{n:.0f}/{total:.0f}"
_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| " + _COUNT + " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
_BAR_FORMAT_NO_ETA = (
    "{desc}: {percentage:3.0f}%|{bar}| " + _COUNT + " [{elapsed}, {rate_fmt}{postfix}]"
)


class _Row:
    """Bookkeeping for one phase, decoupled from the tqdm bar that displays it.

    Completion state stays inspectable (and testable) even when the visible bar
    is disabled -- the bar is treated as a pure view over this state.
    """

    def __init__(self, description: str, total: Optional[float]):
        self.description = description
        self.total = total
        self.completed: float = 0
        self.fields: dict[str, Any] = {}

    @property
    def finished(self) -> bool:
        return self.total is not None and self.completed >= self.total


class Phase:
    """Handle for advancing a single phase (e.g. from a worker callback)."""

    def __init__(self, row: _Row, bar: Optional[tqdm]):
        self._row = row
        self._bar = bar

    def advance(self, step: float = 1) -> None:
        self._row.completed += step
        if self._bar is not None:
            self._bar.update(step)

    def update(
        self,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        **_ignored: Any,
    ) -> None:
        if total is not None:
            self._row.total = total
            if self._bar is not None:
                self._bar.total = total
        if completed is not None:
            self._row.completed = completed
            if self._bar is not None:
                self._bar.n = completed
        if self._bar is not None:
            self._bar.refresh()


class PhasedProgress:
    """A progress display for a job made of sequential phases.

    Use as a context manager and open each phase with ``phase()`` (or ``track()``
    to wrap an iterable)::

        with self.progress(phases=2) as prog:
            for item in prog.track(items, "Generating circuits"):
                ...
            with prog.phase("Sampling circuits", total=n, show_eta=False) as ph:
                ...  # ph.advance(k) from a callback

    ``phases`` is used only to label each bar ``[i/phases]`` for a sense of
    overall position. A phase that stops early with no error (e.g. sinter hitting
    ``max_errors`` before ``max_shots``) is shrunk to its completed count so it
    reads as finished rather than stalled.
    """

    def __init__(
        self,
        title: str = "",
        phases: Optional[int] = None,
        disable: bool = False,
        **_ignored: Any,
    ):
        self._title = title
        self._phases = phases
        self._disable = disable
        self._rows: list[_Row] = []
        self._index = 0

    def __enter__(self) -> PhasedProgress:
        if self._title and not self._disable:
            tqdm.write(self._title)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    @property
    def tasks(self) -> list[_Row]:
        """The phase rows opened so far (in order), for inspection/testing."""
        return list(self._rows)

    @contextmanager
    def phase(
        self,
        description: str,
        total: Optional[float] = None,
        show_eta: bool = True,
    ) -> Iterator[Phase]:
        """Open the next sequential phase as its own bar.

        ``show_eta=False`` hides the remaining-time estimate (useful when the
        phase advances in coarse, bursty steps that make an ETA jump around).
        """
        self._index += 1
        label = f"[{self._index}/{self._phases}] {description}" if self._phases else description

        row = _Row(label, total)
        self._rows.append(row)

        bar: Optional[tqdm] = None
        if not self._disable:
            #TODO - fix so that each bar has same width and x anchor
            bar = tqdm(
                total=total,
                desc=label,
                leave=True,
                dynamic_ncols=True,
                # an explicit count needs a known total to format against
                bar_format=(
                    None
                    if total is None
                    else (_BAR_FORMAT if show_eta else _BAR_FORMAT_NO_ETA)
                ),
            )
        try:
            yield Phase(row, bar)
        except BaseException:
            if bar is not None:
                bar.close()
            raise

        if not row.finished:
            row.total = row.completed
            if bar is not None:
                bar.total = row.completed
                bar.refresh()
        if bar is not None:
            bar.close()

    def track(
        self,
        iterable: Iterable,
        description: str,
        total: Optional[float] = None,
    ) -> Iterator:
        """Iterate ``iterable`` as a phase of its own, advancing once per item."""
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        with self.phase(description, total=total) as ph:
            for item in iterable:
                yield item
                ph.advance()

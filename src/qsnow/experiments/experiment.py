from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from qsnow.experiments.progress import PhasedProgress
from qsnow.interface.models import Tag
from qsnow.visualize.visualize import VisualizationStyle


class Experiment:
    """
    Base class for all experiments.

    Stores the configuration object and general properties shared by every
    experiment; subclass-specific state lives on the subclass itself.

    Shared properties:
      - `tag`: generic annotation (name/desc/metadata); `desc` is the freeform
        text description for recording what the experiment is and any context
        or details not tracked elsewhere.
      - `config`: general configuration values (including run parameters).
      - `results`: output of the latest `run()`.
      - `source`: path this experiment was last saved to / loaded from.
      - `results_refs`: every results file this experiment generated, each named
        relative to the setup file's own directory (see `serialize.flake_ref`), so
        the setup and its results stay linked when the data folder moves as a unit.

    Workflow: implement `run()` in the subclass, then use the inherited
    `save()` / `save_results()` to persist the setup and each run's results as
    separate artifacts (see qsnow.helpers.serialize).
    """

    def __init__(self, desc: str = "", **config):
        self.tag: Tag = Tag(desc=desc)
        self.config: Dict = config
        self.results: Dict = {}
        self.source: Optional[Path] = None
        self.results_refs: List[str] = []

    @property
    def desc(self) -> Optional[str]:
        """Freeform description; convenience accessor for `self.tag.desc`."""
        return self.tag.desc

    @desc.setter
    def desc(self, value: Optional[str]) -> None:
        self.tag.desc = value

    def progress(
        self, phases: Optional[int] = None, title: Optional[str] = None, **kwargs
    ) -> PhasedProgress:
        """
        Create the shared phased progress display for this experiment's `run()`.

        `phases` is the number of sequential phases, used to label each bar
        `[i/phases]`. Subclasses open each phase via `PhasedProgress.phase()` /
        `PhasedProgress.track()`::

            with self.progress(phases=2) as prog:
                for item in prog.track(items, "Generating circuits"):
                    ...
                with prog.phase("Sampling circuits", total=shots) as ph:
                    ...  # ph.advance(n) from a callback
        """
        return PhasedProgress(
            title or f"Experiment: {type(self).__name__}", phases=phases, **kwargs
        )

    def run(self, *args, **kwargs) -> Dict:
        """Execute the experiment, populating and returning `self.results`.

        Implementations should record run parameters in `self.config` so they
        ride along with the saved results record, and should leave the setup
        (chip, tiles) as they found it, so a second `run()` or a `save()` after
        the run does not see sweep state.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement run().")

    def _results_record(
        self, results: Optional["ResultsLike"] = None
    ) -> "ExperimentResults":
        """
        `results` as an `ExperimentResults` record, whichever form it arrived in: a
        record already (returned as is), the bare dict `run()` returns, or (None)
        this experiment's own latest results. A dict is paired with this
        experiment's `config` as its run config, which is what a `save_results()`
        of it would have written.

        Every display surface reads results through here so that `exp.show()`,
        `exp.show(exp.results)` and `exp.show(import_flake(results_path))` all mean
        the same thing.
        """
        if isinstance(results, ExperimentResults):
            return results
        return ExperimentResults(
            experiment_ref=self.source.name if self.source is not None else None,
            run_config=self.config,
            results=self.results if results is None else results,
            desc=self.desc or "",
        )

    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        label: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> Path:
        """
        Export this experiment's setup as a flake (see `serialize.export_flake`).
        `desc` updates the experiment's freeform description before saving.
        """
        # deferred import: qsnow.helpers.serialize imports this module
        from qsnow.helpers import serialize

        if desc is not None:
            self.desc = desc
        self.source = serialize.export_flake(self, path, label=label)
        return self.source

    def save_results(
        self, path: Optional[Union[str, Path]] = None, *, label: Optional[str] = None
    ) -> Path:
        """
        Export `self.results` as a separate results file referencing this
        experiment, and record the back-link in `results_refs`. If the setup
        has been saved, its file is re-exported so the on-disk refs stay current.
        """
        # deferred import: qsnow.helpers.serialize imports this module
        from qsnow.helpers import serialize

        results_path = serialize.export_results(self, path, label=label)
        self.results_refs.append(serialize.flake_ref(results_path, self.source))
        if self.source is not None:
            serialize.export_flake(self, self.source)
        return results_path

    def summary(self) -> Dict[str, object]:
        """Compact facts describing the experiment, for display surfaces
        (visualization headers, HTML exports, reprs)."""
        return {}

    def show(
        self,
        results: Optional["ResultsLike"] = None,
        *,
        extra_styles: Optional[Mapping[str, VisualizationStyle]] = None,
    ):
        """Display an interactive visualization of a run's results. This must be called from the experiment
        in order to expose the underlying chip noise model to the visualization module.

        `results` is a persisted `ExperimentResults` record or the dict `run()`
        returns; with none given, the experiment's own latest results are shown.
        `extra_styles` (name -> VisualizationStyle) extends the views the
        subclass bundles by default.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement show().")


@dataclass(frozen=True)
class ExperimentResults:
    """A persisted experiment run: the results plus the setup/experiment that produced them."""

    experiment_ref: Optional[str]
    run_config: Dict
    results: Dict
    desc: str = ""


# What a display surface accepts as "the results": the persisted record, or the
# live dict `Experiment.run()` returns and stores on `Experiment.results`.
ResultsLike = Union[ExperimentResults, Dict]

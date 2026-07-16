from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from qsnow.experiments.progress import PhasedProgress
from qsnow.interface.models import Tag


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
      - `results_refs`: filenames of every results file this experiment generated.

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

        `phases` sizes the overall bar (one unit per phase; omit for an
        indeterminate overall row). Subclasses open each sequential phase via
        `PhasedProgress.phase()` / `PhasedProgress.track()`::

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
        ride along with the saved results record.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement run().")

    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        label: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> Path:
        """
        Export this experiment's setup as JSON (see `serialize.export_json`).
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
        self.results_refs.append(results_path.name)
        if self.source is not None:
            serialize.export_flake(self, self.source)
        return results_path

    def show(self, results, style_fn):
        pass


@dataclass(frozen=True)
class ExperimentResults:
    """A persisted experiment run: the results plus the setup/experiment that produced them."""

    experiment_ref: Optional[str]
    run_config: Dict
    results: Dict
    desc: str = ""

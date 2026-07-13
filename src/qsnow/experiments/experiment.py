from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


class Experiment:
    """
    Base class for all experiments.

    Stores the configuration object and general properties shared by every
    experiment; subclass-specific state lives on the subclass itself.

    Shared properties:
      - `desc`: freeform text description for recording what the experiment is
        and any context or details not tracked elsewhere.
      - `config`: general configuration values (including run parameters).
      - `results`: output of the latest `run()`.
      - `source`: path this experiment was last saved to / loaded from.
      - `results_refs`: filenames of every results file this experiment generated.

    Workflow: implement `run()` in the subclass, then use the inherited
    `save()` / `save_results()` to persist the setup and each run's results as
    separate artifacts (see qsnow.helpers.serialize).
    """

    def __init__(self, desc: str = "", **config):
        self.desc: str = desc
        self.config: Dict = config
        self.results: Dict = {}
        self.source: Optional[Path] = None
        self.results_refs: List[str] = []

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
        self.source = serialize.export_json(self, path, label=label)
        return self.source

    def save_results(self, path: Optional[Union[str, Path]] = None, *, label: Optional[str] = None) -> Path:
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
            serialize.export_json(self, self.source)
        return results_path


@dataclass(frozen=True)
class ExperimentResults:
    """A persisted experiment run: the results plus the setup that produced them."""

    experiment_ref: Optional[str]
    run_config: Dict
    results: Dict
    desc: str = ""

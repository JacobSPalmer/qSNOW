import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import sinter
from stim import Circuit

from qsnow.experiments.experiment import Experiment, ResultsLike
from qsnow.experiments.sampling import (  # noqa: F401  (re-exported for callers)
    DEFAULT_SAMPLING_BATCH_SIZE,
    ErrorFloorSampler,
)
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.models import Coord
from qsnow.visualize import (
    VisualizationStyle,
    custom_heatmap_style,
    noise_heatmap_style,
    packing_profile_style,
    visualize_interactive,
)

logger = logging.getLogger(__name__)


@dataclass
class SquarePackingExp(Experiment):
    chip: Chip
    tile: LogicalTile
    profile: List[Coord] = field(default_factory=list)
    results: Dict = field(default_factory=dict)

    def __post_init__(self):
        # NOTE - probably a better way to do this, but  safest to always create a copy so as to not accidentally work on the same chip
        self.tile = self.tile.copy()
        self.chip = self.chip.copy()
        super().__init__()
        if not self.profile:
            self.profile = self._generate_profile(self.chip, self.tile)

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def _generate_profile(self, chip: Chip, tile: LogicalTile) -> List[Coord]:
        return chip.candidate_placements(tile)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _circuit_for_profile_loc(self, loc: Coord) -> Circuit:
        """The tile's noise-injected circuit at `loc`, moving or placing the tile there."""
        if self.tile.initialized() and self.tile.chip is self.chip:
            self.tile.shift_to(loc)
        elif not self.chip.add_tile(self.tile, loc):
            # `shift_to` raises on a bad move; a rejected first placement has to as
            # well, or the sweep would sample whatever circuit the tile last held
            raise ValueError(
                f"Profile placement {loc} is not valid on the chip in its current "
                "state, so no circuit can be generated for it."
            )

        return self.tile.circuit

    def _release_tile(self) -> None:
        """Take the sweep tile back off the chip, so a run leaves the setup as it found it."""
        # membership, not `initialized()`: this runs from a `finally`, and a placement
        # that failed midway must not turn into a second error that hides the first
        if any(placed is self.tile for placed in self.chip.tiles):
            self.chip.remove_tile(self.tile)

    def run(
        self,
        shots: int = 50_000,
        max_errors: Optional[int] = 5_000,
        min_errors: int = 1,
        max_topup_shots: Optional[int] = None,
        decoder: str = "pymatching",  # TODO - move default decoder to global config or experiment specific config file at some point
        batch_size: int = DEFAULT_SAMPLING_BATCH_SIZE,
        max_workers: int = 32,
    ):
        """Sample the logical error rate of every candidate placement.

        `min_errors` is the error floor: a placement that records fewer than this
        many errors within `shots` is resampled until it clears the floor, so no
        placement is reported at exactly 0% LER just because it was undersampled.
        Pass `min_errors=0` to skip that pass.
        """
        sampler = ErrorFloorSampler(
            decoder=decoder,
            shots=shots,
            max_errors=max_errors,
            min_errors=min_errors,
            max_topup_shots=max_topup_shots,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        logger.info(f"{len(self.profile)} valid placements to sample.")

        logger.info(
            f"Beginning run with {sampler.num_workers} workers with max batch size of "
            f"{batch_size} and {shots} shots per sample."
        )

        with self.progress(phases=1 + sampler.phases) as prog:
            t_start = perf_counter()
            try:
                tasks = [
                    sinter.Task(
                        circuit=self._circuit_for_profile_loc(loc),
                        json_metadata={"loc": loc},
                    )
                    for loc in prog.track(self.profile, "Generating circuits")
                ]
            finally:
                # the circuits are captured, so the tile has no business staying on
                # the chip: a save() after the run would otherwise embed it, and a
                # second run() would find its first placement occupied
                self._release_tile()
            t_generation = perf_counter() - t_start

            run = sampler.collect(tasks, prog)

            # sinter round-trips json_metadata through JSON, so 'loc' comes back as a list
            self.results = {
                tuple(s.json_metadata["loc"]): {
                    "strong_id": s.strong_id,
                    "shots": s.shots,
                    "errors": s.errors,
                    "ler": s.errors / s.shots,
                }
                for s in run.stats
            }
            # TODO - the stats is stored in config rn but will need to be moved to it's own subdictionary, which will likely require a migration
            # TODO - create minor versioning in the serialize code
            self.config.update(
                **sampler.run_config(),
                stats={
                    "runtime": {
                        "generation": f"{t_generation}",
                        **{k: f"{v}" for k, v in run.timings.items()},
                    }
                },
            )

        return self.results

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        """Compact facts describing the experiment, for display surfaces
        (visualization headers, HTML exports, reprs)."""
        return {
            "chip": self.chip.summary(),
            "tile": self.tile.summary(),
            "placements": len(self.profile),
        }

    def _footprint_for(self, origin: Coord) -> Tuple[Coord, Coord]:
        return self.chip.footprint_for(origin, self.tile.length, self.tile.height)

    def _average_per_for_candidate_placements(self) -> Dict[Coord, float]:
        from statistics import mean

        return {
            o: mean(
                [
                    q.noise.p
                    for q in self.chip.select_rect(
                        *o, *self._footprint_for(o)[1]
                    ).values()
                ]
            )
            for o in self.profile
        }

    def _bounds_for_candidate_placements(self) -> Dict[Coord, Coord]:
        return {o: self._footprint_for(o)[1] for o in self.profile}

    def _interactive_styles(
        self, results: Optional[ResultsLike] = None
    ) -> Dict[str, VisualizationStyle]:
        """The view bundle for `show`: chip-level views plus LER/placement results."""
        ler_map = {
            k: v["ler"] for k, v in self._results_record(results).results.items()
        }
        bounds_map = self._bounds_for_candidate_placements()
        avg_per_map = self._average_per_for_candidate_placements()
        base_map = {
            k: f"{k} → {bounds_map.get(k) if bounds_map.get(k) else k}"
            for k in self.profile
        }

        profile = {
            loc: {
                "base": f"{loc} → {bounds_map.get(loc) if bounds_map.get(loc) else loc}",
                "ler": ler_map.get(loc),
                "bound": bounds_map.get(loc),
            }
            for loc in self.profile
        }

        styles = {
            "PER": noise_heatmap_style(
                self.chip, desc="Heatmap of each qubit's physical error rate (PER)."
            ),
            "Candidate Placements": packing_profile_style(
                self.chip,
                profile,
                desc="Placements, or sites, where tiles could be validly placed, anchored by the tile's origin (upper-leftmost qubit).",
            ),
            "Avg. PER": custom_heatmap_style(
                self.chip,
                avg_per_map,
                "Avg. PER",
                additional_hovertext={"base": base_map, "LER": ler_map},
                desc="Average physical error rate across the tile footprint for each candidate site.",
            ),
            "LER": custom_heatmap_style(
                self.chip,
                ler_map,
                "LER",
                additional_hovertext={"base": base_map, "Avg. PER": avg_per_map},
                desc="Sampled logical error rate (LER) if the tile's origin were placed at each candidate site.",
            ),
        }

        return styles

    def show(
        self,
        results: Optional[ResultsLike] = None,
        *,
        extra_styles: Optional[Mapping[str, VisualizationStyle]] = None,
    ):
        styles = self._interactive_styles(results)
        styles.update(extra_styles or {})
        visualize_interactive(
            self.chip,
            styles,
            active="LER",
            title=self.tag.name or type(self).__name__,
            subtitle=f"chip: {self.chip.unit_dims[0]} x {self.chip.unit_dims[1]} · tile: {self.tile.spec.distance} {self.tile.tag.name} · {len(self.profile)} placements",
            show=True,
        )

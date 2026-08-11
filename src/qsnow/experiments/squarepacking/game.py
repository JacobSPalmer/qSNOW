import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypeVar, TYPE_CHECKING
from stim import Circuit
from time import perf_counter

import sinter

logger = logging.getLogger(__name__)

# Default number of tasks handed to a single ``sinter.collect`` call. Sinter recomputes an O(n) status update 
# (where n is number of circuits to sample) each time a task is completed, causing a pretty nasty scaling as 
# chip sizes get larger. Batching it caps that recompute to O(batch_size), which drastically increases the 
# performance on large-scale simulations. There's probably a more optimal batch size, but with some basic testing
# it seems that ~1024 is the sweet spot.
DEFAULT_SAMPLING_BATCH_SIZE = 1024

def _chunked(items: List, size: int) -> Iterator[List]:
    """Yield ``items`` in consecutive lists of at most ``size`` (size >= 1)."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]

from qsnow.experiments.experiment import Experiment, ExperimentResults
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.models import Coord
from qsnow.visualize import (
    VisualizationStyle,
    custom_heatmap_style,
    noise_heatmap_style,
    packing_profile_style,
    visualize_interactive,
)

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
        profile = []
        for i in range(chip.length - 1):
            for j in range(chip.height - 1):
                origin = (i, j)
                bound = (i + tile.length, j + tile.height)
                if chip.is_valid_tile_placement(origin, bound):
                    profile.append(origin)

        logger.info(f"{len(profile)} valid placements to sample.")
        return profile


    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    
    def _circuit_for_profile_loc(self, loc: Coord) -> Circuit:
        if self.tile.initialized() and self.tile.chip == self.chip:
            self.tile.shift_to(loc)
        else:
            self.chip.add_tile(self.tile, loc)

        return self.tile.circuit

    #TODO - the bulk of the logic here needs to be moved to a shared runner class since this logic is reusable and not unique to this SPP
    def run(
        self,
        shots: int = 50_000,
        max_errors: Optional[int] = 5_000,
        decoder: str = "pymatching",
        batch_size: int = DEFAULT_SAMPLING_BATCH_SIZE,
        max_workers: int = 32
    ):
        num_workers =  min(max_workers, os.cpu_count() or 1)
        logger.info(f"Beginning run with {num_workers} workers with max batch size of {batch_size} and {shots} shots per sample.")

        #TODO - most if not all of the logic for sampling mass experiments should be extracted to a dedicated reusable class.
        with self.progress(phases=2) as prog:
            t_start = perf_counter()
            tasks = [
                sinter.Task(
                    circuit=self._circuit_for_profile_loc(loc),
                    json_metadata={"loc": loc},
                )
                for loc in prog.track(self.profile, "Generating circuits")
            ]
            t_generation = perf_counter() - t_start

            t_start = perf_counter()
            
            collected_stats: List[sinter.TaskStats] = []
            with prog.phase(
                "Sampling circuits", total=len(tasks), show_eta=True
            ) as phase:
                for batch in _chunked(tasks, batch_size):
                    batch_stats = sinter.collect(
                        num_workers=num_workers,
                        tasks=batch,
                        decoders=[
                            decoder
                        ],  # TODO - move default decoder to global config or experiment specific config file at some point
                        max_shots=shots,
                        max_errors=max_errors,
                    )
                    collected_stats.extend(batch_stats)
                    phase.advance(len(batch_stats))

            t_sampling = perf_counter() - t_start

            # sinter round-trips json_metadata through JSON, so 'loc' comes back as a list
            self.results = {
                tuple(s.json_metadata["loc"]): {
                    "strong_id": s.strong_id,
                    "shots": s.shots,
                    "errors": s.errors,
                    "ler": s.errors / s.shots,
                }
                for s in collected_stats
            }
            #TODO - the stats is stored in config rn but will need to be moved to it's own subdictionary, which will likely require a migration
            #TODO - create minor versioning in the serialize code
            self.config.update(
                shots=shots,
                max_errors=max_errors,
                decoder=decoder,
                batch_size=batch_size,
                max_workers=max_workers,
                stats = {
                    'runtime': {
                        'generation': f"{t_generation}",
                        'sampling': f"{t_sampling}",
                    }
                }
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
    
    def _average_per_for_candidate_placements(self) -> Dict[Coord, float]:
        from statistics import mean
        return {o: mean([q.noise.p for q in self.chip.select_rect(*o, o[0] +  self.tile.length - 1, o[1] + self.tile.height - 1).values()]) for o in self.profile}

    def _bounds_for_candidate_placements(self) -> Dict[Coord, Coord]:
        return {o: (o[0] +  self.tile.length - 1, o[1] + self.tile.height - 1) for o in self.profile}

    def _interactive_styles(
        self, results: ExperimentResults
    ) -> Dict[str, VisualizationStyle]:
        """The view bundle for `show`: chip-level views plus LER/placement results."""
        ler_map = {k: v["ler"] for k, v in results.results.items()}
        bounds_map = self._bounds_for_candidate_placements()
        avg_per_map = self._average_per_for_candidate_placements()
        base_map = {k: f'{k} → {bounds_map.get(k) if bounds_map.get(k) else k}' for k in self.profile}

        profile = {loc: {"base": f'{loc} → {bounds_map.get(loc) if bounds_map.get(loc) else loc}',
                         "ler": ler_map.get(loc), 
                         "bound": bounds_map.get(loc)} for loc in self.profile}

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
                'Avg. PER',
                additional_hovertext={'base': base_map, 'LER': ler_map},
                desc="Average physical error rate across the tile footprint for each candidate site.",
            ),
            "LER": custom_heatmap_style(
                self.chip,
                ler_map,
                'LER',
                additional_hovertext={'base': base_map, 'Avg. PER': avg_per_map},
                desc="Sampled logical error rate (LER) if the tile's origin were placed at each candidate site.",
            )
        }

        return styles

    def show(
        self,
        results: ExperimentResults,
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
            subtitle=f"chip: {self.chip.length} x {self.chip.height} grid · tile: {self.tile.spec.distance} {self.tile.tag.name} · {len(self.profile)} placements",
            show=True,
        )

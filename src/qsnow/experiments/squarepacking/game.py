from __future__ import annotations

from typing import TYPE_CHECKING

from qsnow.interface.models import Coord

if TYPE_CHECKING:
    from qsnow.interface.circuit import Circuit

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sinter

logger = logging.getLogger(__name__)

from qsnow.experiments.experiment import Experiment, ExperimentResults
from qsnow.experiments.progress import Phase
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
        # TODO - probably a better way to do this, but safest to always create a copy so as to not accidentally work on the same chip
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

        logger.info(f"{len(profile)} valid placements to sample")
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

    def run(
        self, shots: int = 50_000, max_errors: int = 5_000, decoder: str = "pymatching"
    ):
        def _sinter_progress_callback(phase: Phase):
            def callback(progress_data: sinter.Progress):
                delta_shots = sum(stat.shots for stat in progress_data.new_stats)
                phase.advance(delta_shots)

            return callback

        with self.progress(phases=2) as prog:
            # TODO - modify progress class this bar to use "1/<circuits to sample" rather than percents
            tasks = [
                sinter.Task(
                    circuit=self._circuit_for_profile_loc(loc),
                    json_metadata={"loc": loc},
                )
                for loc in prog.track(self.profile, "Generating circuits")
            ]

            with prog.phase("Sampling circuits", total=shots) as phase:
                collected_stats: List[sinter.TaskStats] = sinter.collect(
                    num_workers=os.process_cpu_count() or os.cpu_count() or 1,
                    tasks=tasks,
                    decoders=[
                        decoder
                    ],  # TODO - move default decoder to global config or experiment specific config file at some point
                    max_shots=shots,
                    max_errors=max_errors,
                    progress_callback=_sinter_progress_callback(phase),
                )

            # sinter round-trips json_metadata through JSON, so 'loc' comes back as a list
            self.results = {
                tuple(s.json_metadata["loc"]): {
                    "strong_id": s.strong_id,
                    "shots": s.shots,
                    "errors": s.errors,
                    "ler": s.errors / s.shots,
                    "discards": s.discards,
                    "seconds": round(s.seconds, 3),
                }
                for s in collected_stats
            }
            self.config.update(shots=shots, max_errors=max_errors, decoder=decoder)

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

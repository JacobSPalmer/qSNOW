import os
from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Literal, Optional

import sinter

import logging
logger = logging.getLogger(__name__)

from qsnow.experiments.experiment import Experiment, ExperimentResults
from qsnow.experiments.progress import Phase
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.models import Coord, Tag
from qsnow.visualize import visualize, VisualizationStyle, custom_heatmap_style


@dataclass
class SquarePackingExp(Experiment):
    chip: Chip
    tile: LogicalTile
    profile: List[Coord] = field(default_factory=list)
    results: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        #TODO - probably a better way to do this, but safest to always create a copy so as to not accidentally work on the same chip
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
        
        logger.info(f'{len(profile)} valid placements to sample')
        return profile

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _circuit_for_profile_loc(self, loc: Coord):
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

    def show(self, results: ExperimentResults, style_fn: Optional[VisualizationStyle] = None):
        style = custom_heatmap_style(chip=self.chip, coord_float_map={k:v['ler'] for k, v in results.results.items()})
        visualize(chip=self.chip, style=style, show=True)
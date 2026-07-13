import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import sinter
from rich.progress import track, Progress

from qsnow.experiments.experiment import Experiment
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.models import Coord
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

@dataclass
class SquarePackingExp(Experiment):
    chip: Chip
    tile: LogicalTile
    bad: float = 0.005
    profile: Dict[Coord, Any] = field(default_factory=dict)
    results: Dict = field(default_factory=dict)

    def __post_init__(self):
        super().__init__()
        if not self.profile:
            self.profile = self._generate_profile(self.chip, self.tile)

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def _generate_profile(self, chip: Chip, tile: LogicalTile):
        profile = {}
        for i in range(chip.length - 1):
            for j in range(chip.height - 1):
                origin = (i, j)
                bound = (i + tile.length, j + tile.height)
                if chip.is_valid_tile_placement(origin, bound):
                    profile[(float(origin[0]), float(origin[1]))] = {
                        "origin": origin,
                        "bound": (bound[0] - 1, bound[1] - 1),
                        "circuit": None,
                    }
        print(f"{len(profile)} possible valid placements will be simulated!")
        return profile

    def create_profile(self):
        tile = self.tile.copy()

        self.chip.add_tile(tile, (0, 0))
        self.profile[(0, 0)]["circuit"] = tile.circuit

        for k, v in track(
            self.profile.items(), description=f"Generating {len(self.profile)} noise-injected circuits ..."
        ):
            tile.shift_to(k)
            self.profile[k]["circuit"] = tile.circuit.copy()

        self.chip.remove_tile(0)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self, shots=50_000, max_errors=5_000):
        tasks = [
            sinter.Task(
                circuit=p['circuit'],
                json_metadata={"loc": loc, "shots": shots, "max_errors": max_errors},
            )
            for loc, p in self.profile.items()
        ]

        def _create_sinter_progress_callback(progress: Progress, task_id):
            def callback(progress_data: sinter.Progress):
                delta_shots = sum(stat.shots for stat in progress_data.new_stats)
                progress.update(task_id, advance=delta_shots)
            return callback
        
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),  # Shows time spent so far
                TextColumn("/"),
                TimeRemainingColumn(), # Shows estimated time left
            ) as progress:

            task = progress.add_task("[cyan]Sampling...", total=shots)

            prog_callback = _create_sinter_progress_callback(progress, task)

            collected_stats: List[sinter.TaskStats] = sinter.collect(
                num_workers=os.cpu_count(),
                tasks=tasks,
                decoders=["pymatching"],
                max_shots=shots,
                max_errors=max_errors,
                progress_callback=prog_callback,
            )

        # sinter round-trips json_metadata through JSON, so 'loc' comes back as a list
        self.results = {tuple(s.json_metadata['loc']):
                    {
                        "strong_id": s.strong_id,
                        "decoder": s.decoder,
                        "json_metadata": s.json_metadata,
                        "shots": s.shots,
                        "ler": s.errors / s.shots,
                        "errors": s.errors,
                        "discards": s.discards,
                        "seconds": s.seconds,
                    } for s in collected_stats}

        self.config.update(shots=shots, max_errors=max_errors)
        return self.results

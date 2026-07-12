from typing import Dict, Any, List, Optional
from dataclasses import field, dataclass

from qsnow.interface.models import Qubit, Coord
from qsnow.interface.chip import LogicalTile, Chip

import os
import sinter

from rich.progress import track

import json
from datetime import datetime

# WIP
@dataclass
class SquarePackingExp():
    chip: Chip
    tile: LogicalTile
    bad: float = 0.005
    profile: Dict[Coord, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.profile:
            self.profile = self._generate_profile(self.chip, self.tile)

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    # i -> col, j -> row
    def _generate_profile(self, chip: Chip, tile: LogicalTile):
        profile = {}
        for i in range(chip.length - 1):
            for j in range(chip.height - 1):
                origin = (i, j)
                bound = (i + tile.length, j + tile.height)
                if chip.is_valid_tile_placement(origin, bound):
                    profile[(float(origin[0]), float(origin[1]))] = {'origin': origin, 'bound':(bound[0] - 1, bound[1] - 1), 'circuit': None, 'ler': 'N/A', 'time': None}

        print(f"{len(profile)} possible valid placements will be simulated!")
        return profile

    def create_profiles_via_deletion(self, chip: Chip, tile: LogicalTile):
        for k, v in self.profile.items():
            chip.add_tile(tile.copy(), k)
            v['circuit'] = chip.tiles[0].circuit
            del chip.tiles[0]

    def create_profile(self):
        tile = self.tile.copy()
        
        self.chip.add_tile(tile, (0,0))
        self.profile[(0,0)]['circuit'] = tile.circuit

        for k, v in track(self.profile.items(), description="Generating noise injected circuits..."):
            # print(f"Attempting shift from {tile.origin} to {k}")
            tile.shift_to(k)
            self.profile[k]['circuit'] = tile.circuit.copy()

        # clean up
        del self.chip.tiles[0]

    # ------------------------------------------------------------------
    # Import/export
    # ------------------------------------------------------------------

    def generate_filename(self, base_title="report", extension="txt"):
        # Get current timestamp
        now = datetime.now()
        
        # Format: YYYY-MM-DD_HH-MM-SS (Safe for Windows, Mac, and Linux)
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Combine into final title
        filename = f"{base_title}_{timestamp}.{extension}"
        return filename

    def run_simulation(self, shots = 50_000, max_errors = 5_000):
        tasks = [
            sinter.Task(
                circuit = p.circuit,
                json_metadata = {'loc': loc, 'shots': shots, 'max_errors': max_errors}
            )
            for loc, p in self.profile.items()
        ]

        collected_stats: List[sinter.TaskStats] = sinter.collect(
            num_workers = os.cpu_count(),
            tasks = tasks,
            decoders = ['pymatching'],
            max_shots = shots,
            max_errors = max_errors,
            print_progress= True
        )

        dump = {
            'profile': self.to_dict(),
            'results': collected_stats
        }

        with open(f"data/{self.generate_filename('run','json')}", "w") as file:
            json.dump(dump, file)

    # ------------------------------------------------------------------
    # Import/export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            'chip': self.chip.to_dict(),
            'tile_tag': self.tile.tag,
            'profile': self.profile 
        }

    @classmethod
    def from_json(cls, file):
        return NotImplemented

        

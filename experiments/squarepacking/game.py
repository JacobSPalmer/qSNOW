from typing import Dict, Any, List, Optional

from interface.models import Qubit, Coord
from interface.chip import LogicalTile, Chip

# WIP
class SquarePackingExp():

    def __init__(self,
                 chip: Chip,
                 tile: LogicalTile,
                 bad: float = 0.005):
        self.chip = chip
        self.tile = tile
        self.bad = bad
        self.profile = self.create_profile(chip, tile)

    # i -> col, j -> row
    def generate_profile(self, chip: Chip, tile: LogicalTile):
        profile = {}
        for i in range(chip.length - 1):
            for j in range(chip.height - 1):
                origin = (i, j)
                bound = (i + tile.length, j + tile.height)
                if chip.is_valid_tile_placement(origin, bound):
                    profile[(float(origin[0]), float(origin[1]))] = {'bound':(bound[0] - 1, bound[1] - 1), 'circuit': None, 'ler': 'N/A', 'time': None}

        print(f"{len(profile)} possible valid placements will be simulated!")
        return profile

    def create_profiles_via_deletion(self, chip: Chip, tile: LogicalTile):
        first = True
        profile = self.generate_profile(chip,tile)
        
        for k, v in profile.items():
            chip.add_tile(tile.copy(), k)
            v['circuit'] = chip.tiles[0].circuit
            del chip.tiles[0]
        return profile

    def create_profile(self, chip: Chip, tile: LogicalTile):
        profile = self.generate_profile(chip, tile)
        tile = tile.copy()
        
        chip.add_tile(tile, (0,0))
        profile[(0,0)]['circuit'] = tile.circuit

        for k, v in profile.items():
            # print(f"Attempting shift from {tile.origin} to {k}")
            tile.shift_to(k)
            profile[k]['circuit'] = tile.circuit

        return profile

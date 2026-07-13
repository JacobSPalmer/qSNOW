from qsnow.interface import Chip
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.models import CSSType


class TestSurfaceCodeTileConstruction:
    def test_construct_basic_tile(self, chip: Chip):
        tile = SCTile(3, rounds=3, task="memory_x")
        chip.add_tile(tile)

        memory_x = [(2, 0), (2, 4), (4, 2), (4, 6)]
        memory_z = [(2, 2), (0, 4), (6, 2), (4, 4)]
        data = [(1, 1), (3, 1), (1, 3), (3, 3), (5, 1), (5, 3), (1, 5), (3, 5), (5, 5)]
        buffer = [
            (0, 0),
            (0, 2),
            (4, 0),
            (6, 0),
            (7, 1),
            (7, 3),
            (6, 4),
            (0, 6),
            (2, 6),
            (1, 7),
            (3, 7),
            (7, 5),
            (6, 6),
            (5, 7),
            (7, 7),
        ]

        assert all(chip.loc(c).is_x_measure() for c in memory_x)
        assert all(chip.loc(c).is_z_measure() for c in memory_z)
        assert all(chip.loc(c).is_data() for c in data)
        assert all(chip.loc(c).type == CSSType.BUFFER for c in buffer)

from collections import Counter

import pytest

from qsnow.interface import Chip
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.models import CSSType


class TestSurfaceCodeTileConstruction:
    @pytest.mark.parametrize("task", ["memory_x", "memory_z"])
    def test_construct_basic_tile(self, chip: Chip, task):
        tile = SCTile(3, rounds=3, task=task)
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


class TestSurfaceCodeTileTyping:
    """A rotated surface code of distance d has d^2 data qubits and (d^2-1)/2
    stabilizers of each type, whatever the task or distance."""

    @pytest.mark.parametrize("distance", [3, 5])
    @pytest.mark.parametrize("task", ["memory_x", "memory_z"])
    def test_css_type_counts_match_the_code_distance(self, distance, task):
        chip = Chip(10, 10)
        tile = SCTile(distance, task=task)
        assert chip.add_tile(tile)

        counts = Counter(q.type for q in tile.qubits)

        assert counts[CSSType.DATA] == distance**2
        assert counts[CSSType.X_CHECK] == (distance**2 - 1) // 2
        assert counts[CSSType.Z_CHECK] == (distance**2 - 1) // 2

    @pytest.mark.parametrize("task", ["memory_x", "memory_z"])
    def test_typing_partitions_the_circuit_qubits(self, task):
        chip = Chip(10, 10)
        tile = SCTile(5, task=task)
        assert chip.add_tile(tile)

        typed = {c: chip.loc(c).type for c in tile._c2i}

        # every qubit the circuit uses is classified, and never as BUFFER
        assert CSSType.UNASSIGNED not in typed.values()
        assert CSSType.BUFFER not in typed.values()
        # ...while everything outside the circuit is BUFFER
        outside = set(tile.grid) - set(tile._c2i)
        assert all(chip.loc(c).type == CSSType.BUFFER for c in outside)

    @pytest.mark.parametrize("task", ["memory_x", "memory_z"])
    def test_measure_qubits_are_exactly_the_reset_measure_targets(self, task):
        chip = Chip(10, 10)
        tile = SCTile(5, task=task)
        assert chip.add_tile(tile)

        i2c = tile._circuit.get_final_qubit_coordinates()
        mr_targets = {
            (i2c[t.value][0], i2c[t.value][1])
            for instr in tile._circuit.flattened()
            if instr.name == "MR"
            for t in instr.targets_copy()
        }
        measures = {c for c in tile._c2i if chip.loc(c).is_measure()}

        assert measures == mr_targets

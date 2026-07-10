import stim
import pytest

from qsnow.interface.chip import Chip, LogicalTile


@pytest.fixture
def chip() -> Chip:
    return Chip(5, 5)


@pytest.fixture
def surface_code_circuit() -> stim.Circuit:
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z", distance=3, rounds=1
    )


@pytest.fixture
def logical_tile(surface_code_circuit) -> LogicalTile:
    return LogicalTile(surface_code_circuit)

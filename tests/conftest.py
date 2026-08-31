import pytest
import stim

from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.lattice import SQUARE


@pytest.fixture
def chip() -> Chip:
    return Chip(5, 5)


@pytest.fixture
def lg_chip() -> Chip:
    return Chip(10, 10)


@pytest.fixture
def square_chip() -> Chip:
    """A dense integer lattice: 5x5 coordinates, a qubit at every one of them."""
    return Chip(5, 5, lattice=SQUARE)


@pytest.fixture
def dense_circuit() -> stim.Circuit:
    """A circuit on adjacent integer coordinates - illegal on a checkerboard."""
    return stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(0, 1) 2
        QUBIT_COORDS(1, 1) 3
        R 0 1 2 3
        CX 0 1 2 3
        M 0 1 2 3
    """)


@pytest.fixture
def surface_code_circuit() -> stim.Circuit:
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z", distance=3, rounds=3
    )


@pytest.fixture
def logical_tile(surface_code_circuit) -> LogicalTile:
    return LogicalTile(surface_code_circuit)

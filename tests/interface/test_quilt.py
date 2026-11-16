import pytest

from qsnow.interface.chip import Chip
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.lattice import SQUARE
from qsnow.interface.noise import Uniform
from qsnow.interface.quilt import QuiltedChip


@pytest.fixture
def quilt() -> QuiltedChip:
    """Two 5x5 checkerboard chips side by side: the same extent as Chip(10, 5)."""
    return QuiltedChip({(0, 0): Chip(5, 5), (10, 0): Chip(5, 5)})


@pytest.fixture
def l_quilt() -> QuiltedChip:
    """Three 4x4 chips in an L, leaving the top-right 4x4 block unowned."""
    return QuiltedChip({(0, 0): Chip(4, 4), (0, 8): Chip(4, 4), (8, 8): Chip(4, 4)})


# ----------------------------------------------------------------------
# Stitching
# ----------------------------------------------------------------------


def test_stitched_quilt_matches_monolithic_chip(quilt):
    """Stitching abutting modules yields exactly the sites and couplers of one big chip."""
    whole = Chip(10, 5)
    assert quilt.unit_dims == whole.unit_dims
    assert set(quilt.grid) == set(whole.grid)
    assert set(quilt.coupler_map) == set(whole.coupler_map)


def test_seams_are_the_cross_module_edges_and_belong_to_no_module(quilt):
    seams = set(quilt.seam_map)
    crossing = {
        k
        for k in quilt.coupler_map
        if quilt.locate(k[0])[0] is not quilt.locate(k[1])[0]
    }
    assert seams == crossing and seams
    assert all(a[0] == 9 and b[0] == 10 for a, b in seams)
    for module in quilt.modules.values():
        assert not seams & set(module.coupler_map)


def test_locate_maps_quilt_coordinates_into_the_module_frame(quilt):
    module, local = quilt.locate((12, 2))
    assert module is quilt.modules[(10, 0)]
    assert local == (2, 2)
    with pytest.raises(KeyError):
        quilt.locate((100, 100))


def test_module_footprints_tile_the_quilt(quilt):
    assert quilt.module_footprints() == {
        (0, 0): ((0, 0), (9, 9)),
        (10, 0): ((10, 0), (19, 9)),
    }


# ----------------------------------------------------------------------
# Write-through between module and quilt
# ----------------------------------------------------------------------


def test_module_noise_is_visible_through_the_quilt(quilt):
    quilt.modules[(10, 0)].generate_uniform_noise(0.02)
    assert quilt.loc((12, 2)).noise.p == 0.02
    assert quilt.loc((2, 2)).noise.p == 0.0
    assert quilt.coupler((10, 0), (11, 1)).noise.p == 0.02


def test_quilt_writes_reach_the_modules(quilt):
    quilt.set_noise_map({(12, 2): 0.03})
    assert quilt.modules[(10, 0)].loc((2, 2)).noise.p == 0.03

    quilt.generate_uniform_noise(0.01)
    for module in quilt.modules.values():
        assert all(q.noise.p == 0.01 for q in module.qubits)
        assert all(c.noise.p == 0.01 for c in module.couplers)
    assert all(c.noise.p == 0.01 for c in quilt.seam_couplers)


# ----------------------------------------------------------------------
# Seam control
# ----------------------------------------------------------------------


def test_seams_derive_from_their_endpoints(quilt):
    quilt.modules[(0, 0)].generate_uniform_noise(0.01)
    quilt.modules[(10, 0)].generate_uniform_noise(0.03)
    quilt.derive_seam_noise()
    assert all(c.noise.p == pytest.approx(0.02) for c in quilt.seam_couplers)
    quilt.derive_seam_noise("max")
    assert all(c.noise.p == 0.03 for c in quilt.seam_couplers)


def test_generate_seam_noise_touches_only_seams(quilt):
    quilt.generate_uniform_noise(0.01)
    quilt.generate_seam_noise(Uniform(0.05))
    assert all(c.noise.p == 0.05 for c in quilt.seam_couplers)
    seams = set(quilt.seam_map)
    assert all(n.p == 0.01 for k, n in quilt.coupler_map.items() if k not in seams)


# ----------------------------------------------------------------------
# Tiles
# ----------------------------------------------------------------------


def test_tile_spans_the_seam(quilt):
    tile = SCTile(3)
    assert quilt.add_tile(tile, (8, 0))
    left, right = quilt.modules[(0, 0)], quilt.modules[(10, 0)]
    assert any(q.is_active() for q in left.qubits)
    assert any(q.is_active() for q in right.qubits)
    assert tile.circuit.num_qubits > 0

    quilt.remove_tile(tile)
    assert not any(q.is_active() for q in left.qubits + right.qubits)


def test_candidate_placements_match_monolithic_chip(quilt):
    tile = SCTile(3)
    assert quilt.candidate_placements(tile) == Chip(10, 5).candidate_placements(tile)


def test_l_quilt_has_nothing_in_its_hole(l_quilt):
    assert l_quilt.unit_dims == (8, 8)
    assert not l_quilt.select_rect(8, 0, 15, 7)
    assert all(not (a[0] >= 8 and a[1] < 8) for a, _ in l_quilt.coupler_map)


def test_placement_over_a_hole_is_rejected_without_raising(l_quilt):
    tile = SCTile(3)
    with pytest.warns(UserWarning, match="outside every module"):
        assert not l_quilt.add_tile(tile, (8, 0))
    assert all(o[0] < 8 or o[1] >= 8 for o in l_quilt.candidate_placements(tile))
    assert l_quilt.add_tile(tile, (8, 8))


# ----------------------------------------------------------------------
# Construction guards and copy
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "modules",
    [
        pytest.param({}, id="empty"),
        pytest.param(
            {(0, 0): Chip(2, 2), (4, 0): Chip(2, 2, lattice=SQUARE)}, id="lattices"
        ),
        pytest.param({(-1, 0): Chip(2, 2)}, id="negative-offset"),
    ],
)
def test_unstitchable_module_maps_raise(modules):
    with pytest.raises(ValueError):
        QuiltedChip(modules)


@pytest.mark.parametrize(
    "modules",
    [
        pytest.param({(0, 0): Chip(5, 5), (2, 2): Chip(5, 5)}, id="partial"),
        pytest.param({(0, 0): Chip(5, 5), (2, 2): Chip(2, 2)}, id="nested"),
    ],
)
def test_overlapping_modules_raise(modules):
    with pytest.raises(ValueError, match=r"\(0, 0\) and \(2, 2\) overlap"):
        QuiltedChip(modules)


def test_off_lattice_offset_raises_with_nearest_sites():
    with pytest.raises(ValueError, match=r"Nearest valid offsets: \(4, 0\), \(5, 1\)"):
        QuiltedChip({(0, 0): Chip(2, 2), (5, 0): Chip(2, 2)})


def test_odd_site_offset_is_a_valid_half_cell_shift():
    """A module may start on an odd checkerboard site, as a tile origin may."""
    quilt = QuiltedChip({(1, 1): Chip(2, 2)})

    assert quilt.module_footprints() == {(1, 1): ((1, 1), (4, 4))}
    assert set(quilt.grid) == {(x + 1, y + 1) for x, y in Chip(2, 2).grid}
    assert quilt.bound == (5, 5)


def test_reused_module_raises():
    chip = Chip(2, 2)
    with pytest.raises(ValueError, match="distinct"):
        QuiltedChip({(0, 0): chip, (4, 0): chip})


def test_module_with_tiles_raises():
    module = Chip(5, 5)
    module.add_tile(SCTile(3), (0, 0))
    with pytest.raises(ValueError, match="before any tiles"):
        QuiltedChip({(0, 0): module})


def test_copy_is_an_independent_quilt(quilt):
    quilt.generate_uniform_noise(0.01)
    quilt.generate_seam_noise(Uniform(0.04))
    quilt.add_tile(SCTile(3), (8, 0))

    clone = quilt.copy(copy_tiles=True)
    assert isinstance(clone, QuiltedChip)
    assert clone.seam_map == quilt.seam_map
    assert len(clone.tiles) == 1

    clone.modules[(0, 0)].generate_uniform_noise(0.02)
    assert quilt.loc((0, 0)).noise.p == 0.01

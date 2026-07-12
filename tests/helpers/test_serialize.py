import pytest

from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.codes.rsc import SCTile
from qsnow.helpers.serialize import (
    export_json,
    import_json,
    import_latest,
    list_exports,
    set_data_dir,
    to_dict,
    from_dict,
)


@pytest.fixture
def data_dir(tmp_path):
    set_data_dir(tmp_path / "data")
    yield tmp_path / "data"
    set_data_dir()  # restore repo-root default


class TestTileRoundTrip:
    def test_unplaced_tile_round_trips(self, logical_tile):
        restored = from_dict(to_dict(logical_tile))

        assert isinstance(restored, LogicalTile)
        assert str(restored.base_circuit) == str(logical_tile.base_circuit)
        assert restored.origin == logical_tile.origin
        assert restored.length == logical_tile.length
        assert restored.height == logical_tile.height

    def test_tag_round_trips(self, logical_tile):
        logical_tile.tag.name = "my_tile"
        logical_tile.tag.metadata = {"note": "hello"}

        restored = from_dict(to_dict(logical_tile))

        assert restored.tag.name == "my_tile"
        assert restored.tag.metadata == {"note": "hello"}

    def test_ruleset_round_trips(self):
        tile = SCTile(distance=3)
        restored = from_dict(to_dict(tile))

        assert repr(restored._ruleset) == repr(tile._ruleset)

    def test_sc_tile_restores_subclass(self):
        tile = SCTile(distance=3, rounds=2, task="memory_x")
        restored = from_dict(to_dict(tile))

        assert isinstance(restored, SCTile)
        assert restored.tag.distance == 3
        assert restored.tag.rounds == 2
        assert str(restored.base_circuit) == str(tile.base_circuit)


class TestChipRoundTrip:
    def test_empty_chip_round_trips(self, chip):
        chip.generate_random_noise()
        restored = from_dict(to_dict(chip))

        assert isinstance(restored, Chip)
        assert restored.length == chip.length
        assert restored.height == chip.height
        assert {c: q.noise.p for c, q in restored.grid.items()} == \
               {c: q.noise.p for c, q in chip.grid.items()}

    def test_chip_with_tile_restores_placement_and_statuses(self, chip):
        chip.generate_random_noise()
        tile = SCTile(distance=3)
        assert chip.add_tile(tile, (2, 2)) is True

        restored = from_dict(to_dict(chip))

        assert len(restored.tiles) == 1
        assert restored.tiles[0].origin == (2, 2)
        assert {c: (q.status, q.type) for c, q in restored.grid.items()} == \
               {c: (q.status, q.type) for c, q in chip.grid.items()}

    def test_placed_tile_circuit_matches_after_reimport(self, chip):
        chip.generate_random_noise()
        tile = SCTile(distance=3)
        chip.add_tile(tile, (0, 0))

        restored = from_dict(to_dict(chip))

        assert str(restored.tiles[0].circuit) == str(chip.tiles[0].circuit)


class TestExperimentRoundTrip:
    def test_square_packing_exp_round_trips(self, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        chip.generate_random_noise()
        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3), bad=0.01)

        restored = from_dict(to_dict(exp))

        assert isinstance(restored, SquarePackingExp)
        assert restored.bad == 0.01
        assert restored.profile.keys() == exp.profile.keys()
        assert {c: q.noise.p for c, q in restored.chip.grid.items()} == \
               {c: q.noise.p for c, q in chip.grid.items()}

    def test_generic_experiment_round_trips(self):
        from qsnow.experiments.experiment import Experiment

        exp = Experiment(shots=1000, decoder="pymatching")
        restored = from_dict(to_dict(exp))

        assert isinstance(restored, Experiment)
        assert restored.config == {"shots": 1000, "decoder": "pymatching"}


class TestJsonFileRoundTrip:
    def test_export_import_json_file(self, tmp_path, chip):
        chip.generate_random_noise()
        chip.add_tile(SCTile(distance=3), (2, 2))

        path = export_json(chip, tmp_path / "chip.json")
        restored = import_json(path)

        assert isinstance(restored, Chip)
        assert str(restored.tiles[0].circuit) == str(chip.tiles[0].circuit)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            from_dict({"not_qsnow": True})


class TestAutoOrganization:
    def test_auto_paths_by_kind(self, data_dir, chip):
        tile = SCTile(distance=3)

        chip_path = export_json(chip)
        tile_path = export_json(tile)

        assert chip_path.parent == data_dir / "chips"
        assert chip_path.name.startswith("chip_5x5_")
        assert tile_path.parent == data_dir / "tiles"
        assert tile_path.name.startswith("tile_rsc_memory_z_d3_")

    def test_custom_label_keeps_obj_name_prefix(self, data_dir, chip):
        path = export_json(chip, label="baseline")
        assert path.name.startswith("chip_baseline_")

    def test_import_latest_finds_newest(self, data_dir, chip):
        first = export_json(chip, label="run")
        chip.generate_random_noise()
        export_json(chip, label="run")
        import os
        os.utime(first, (0, 0))  # force distinct mtimes

        restored = import_latest("chip_run")

        assert {c: q.noise.p for c, q in restored.grid.items()} == \
               {c: q.noise.p for c, q in chip.grid.items()}

    def test_list_exports_filters_by_kind_and_pattern(self, data_dir, chip):
        export_json(chip)
        export_json(SCTile(distance=3))

        assert len(list_exports()) == 2
        assert len(list_exports(kind="tiles")) == 1
        assert len(list_exports("chip*")) == 1

    def test_import_latest_missing_raises(self, data_dir):
        with pytest.raises(FileNotFoundError):
            import_latest("nonexistent")

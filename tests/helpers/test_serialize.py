from pathlib import Path

import pytest

from qsnow.helpers.serialize import (
    FORMAT_VERSION,
    export_json,
    from_dict,
    import_json,
    import_latest,
    list_exports,
    set_data_dir,
    to_dict,
)
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.codes.rsc import SCTile


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
        assert restored.spec.distance == 3
        assert restored.spec.rounds == 2
        assert str(restored.base_circuit) == str(tile.base_circuit)


class TestChipRoundTrip:
    def test_empty_chip_round_trips(self, chip):
        chip.generate_random_noise()
        restored = from_dict(to_dict(chip))

        assert isinstance(restored, Chip)
        assert restored.length == chip.length
        assert restored.height == chip.height
        assert {c: q.noise.p for c, q in restored.grid.items()} == {
            c: q.noise.p for c, q in chip.grid.items()
        }

    def test_chip_with_tile_restores_placement_and_statuses(self, chip):
        chip.generate_random_noise()
        tile = SCTile(distance=3)
        assert chip.add_tile(tile, (2, 2)) is True

        restored = from_dict(to_dict(chip))

        assert len(restored.tiles) == 1
        assert restored.tiles[0].origin == (2, 2)
        assert {c: (q.status, q.type) for c, q in restored.grid.items()} == {
            c: (q.status, q.type) for c, q in chip.grid.items()
        }

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
        exp.desc = "square packing on a 5x5 chip"

        restored = from_dict(to_dict(exp))

        assert isinstance(restored, SquarePackingExp)
        assert restored.bad == 0.01
        assert restored.desc == "square packing on a 5x5 chip"
        assert restored.profile.keys() == exp.profile.keys()
        assert {c: q.noise.p for c, q in restored.chip.grid.items()} == {
            c: q.noise.p for c, q in chip.grid.items()
        }

    def test_square_packing_payload_nests_under_experiment_headings(self, chip):
        from qsnow.experiments.experiment import Experiment
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        assert isinstance(exp, Experiment)

        data = to_dict(exp)

        assert set(data.keys()) == {
            "__qsnow__",
            "format_version",
            "tag",
            "config",
            "results_refs",
            "exp",
        }
        assert set(data["exp"].keys()) == {"chip", "tile", "bad", "profile"}

    def test_generic_experiment_round_trips(self):
        from qsnow.experiments.experiment import Experiment

        exp = Experiment(desc="a note", shots=1000, decoder="pymatching")
        restored = from_dict(to_dict(exp))

        assert isinstance(restored, Experiment)
        assert restored.desc == "a note"
        assert restored.config == {"shots": 1000, "decoder": "pymatching"}


class TestTagAnnotations:
    def test_chip_tag_round_trips(self, chip):
        chip.tag.name = "baseline"
        chip.tag.desc = "5x5 chip for BAD sweeps"
        chip.tag.metadata = {"campaign": 1}

        restored = from_dict(to_dict(chip))

        assert restored.tag.name == "baseline"
        assert restored.tag.desc == "5x5 chip for BAD sweeps"
        assert restored.tag.metadata == {"campaign": 1}

    def test_tile_tag_desc_round_trips(self):
        tile = SCTile(distance=3)
        tile.tag.desc = "d3 memory-z tile"

        restored = from_dict(to_dict(tile))

        assert restored.tag.desc == "d3 memory-z tile"

    def test_export_json_desc_persists_through_reimport(self, data_dir, chip):
        path = export_json(chip, desc="written at export time")

        restored = import_json(path)

        assert chip.tag.desc == "written at export time"
        assert restored.tag.desc == "written at export time"

    def test_experiment_desc_lives_on_tag(self, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.desc = "via the property"

        assert exp.tag.desc == "via the property"
        assert from_dict(to_dict(exp)).desc == "via the property"

    def test_summarize_exports_maps_paths_to_descs(self, data_dir, chip):
        from qsnow.helpers.serialize import summarize_exports

        chip_path = export_json(chip, desc="a described chip")
        tile_path = export_json(SCTile(distance=3))

        summary = summarize_exports()

        assert summary[chip_path] == "a described chip"
        assert summary[tile_path] is None


class TestResultsFlow:
    def test_save_run_save_results_round_trip(self, data_dir, chip):
        from qsnow.experiments.experiment import ExperimentResults
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        setup_path = exp.save()
        assert exp.source == setup_path

        # reimport the setup, fake a run, save its results separately
        restored = import_json(setup_path)
        assert restored.source == setup_path
        restored.results = {(0.0, 0.0): {"ler": 0.001, "shots": 1000}}
        restored.config.update(shots=1000, max_errors=100)
        results_path = restored.save_results()

        record = import_json(results_path)

        assert isinstance(record, ExperimentResults)
        assert record.experiment_ref == setup_path.name
        assert record.results == {(0, 0): {"ler": 0.001, "shots": 1000}}
        assert record.run_config["shots"] == 1000
        # the setup file itself was never replaced by the results export
        assert results_path != setup_path

    def test_save_with_desc_updates_description(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        path = exp.save(desc="testing desc at save time")

        assert exp.desc == "testing desc at save time"
        assert import_json(path).desc == "testing desc at save time"

    def test_save_results_backlinks_saved_setup(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        setup_path = exp.save()
        exp.results = {(0.0, 0.0): {"ler": 0.001}}
        results_path = exp.save_results()

        assert exp.results_refs == [results_path.name]
        # the additive re-export refreshed the on-disk setup's refs
        assert import_json(setup_path).results_refs == [results_path.name]

    def test_save_results_without_saved_setup_warns(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.results = {(0.0, 0.0): {"ler": 0.001}}

        with pytest.warns(UserWarning, match="call exp.save"):
            results_path = exp.save_results()

        assert import_json(results_path).experiment_ref is None


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

        assert {c: q.noise.p for c, q in restored.grid.items()} == {
            c: q.noise.p for c, q in chip.grid.items()
        }

    def test_list_exports_filters_by_kind_and_pattern(self, data_dir, chip):
        export_json(chip)
        export_json(SCTile(distance=3))

        assert len(list_exports()) == 2
        assert len(list_exports(kind="tiles")) == 1
        assert len(list_exports("chip*")) == 1

    def test_label_only_pattern_falls_back_to_substring_match(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.save(label="d3-20x20chip")

        # filename starts with "experiment_", but the bare label still matches
        restored = import_latest("d3-20x20chip", kind="experiments")

        assert isinstance(restored, SquarePackingExp)

    def test_import_latest_missing_raises(self, data_dir):
        with pytest.raises(FileNotFoundError):
            import_latest("nonexistent")


class TestFormatVersioning:
    FIXTURES = Path(__file__).parent / "fixtures"

    def test_v1_chip_golden_file_imports(self):
        chip = import_json(self.FIXTURES / "chip_v1.json")

        assert isinstance(chip, Chip)
        assert (chip.length, chip.height) == (10, 10)
        assert len(chip.tiles) == 1
        assert chip.tiles[0].origin == (2, 2)

    def test_v1_tile_golden_file_imports(self):
        tile = import_json(self.FIXTURES / "tile_v1.json")

        assert isinstance(tile, SCTile)
        assert tile.spec.distance == 3
        assert tile.spec.rounds == 2

    def test_v1_experiment_golden_file_imports(self):
        exp = import_json(self.FIXTURES / "experiment_v1.json")

        assert exp.config == {"shots": 1000, "decoder": "pymatching"}

    def test_future_version_raises_clear_error(self, chip):
        data = to_dict(chip)
        data["format_version"] = FORMAT_VERSION + 1

        with pytest.raises(ValueError, match="Upgrade qsnow"):
            from_dict(data)

    def test_missing_version_defaults_to_v1(self, chip):
        data = to_dict(chip)
        del data["format_version"]

        restored = from_dict(data)

        assert isinstance(restored, Chip)

    def test_migration_steps_chain_to_current_version(self, monkeypatch, chip):
        import qsnow.helpers.serialize as serialize

        applied = []
        monkeypatch.setattr(serialize, "FORMAT_VERSION", FORMAT_VERSION + 2)
        monkeypatch.setitem(
            serialize._MIGRATIONS, FORMAT_VERSION, lambda d: applied.append(1) or d
        )
        monkeypatch.setitem(
            serialize._MIGRATIONS, FORMAT_VERSION + 1, lambda d: applied.append(2) or d
        )

        data = to_dict(chip)
        data["format_version"] = FORMAT_VERSION
        restored = from_dict(data)

        assert applied == [1, 2]
        assert isinstance(restored, Chip)

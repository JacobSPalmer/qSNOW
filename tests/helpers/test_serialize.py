from pathlib import Path
from statistics import mean

import pytest

from qsnow.helpers.serialize import (
    FORMAT_VERSION,
    export_flake,
    from_dict,
    import_flake,
    import_latest,
    list_exports,
    set_data_dir,
    to_dict,
)
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.lattice import CHECKERBOARD, SQUARE
from qsnow.interface.models import NoiseModelSpec


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

    def test_square_lattice_chip_round_trips(self, square_chip, dense_circuit):
        square_chip.generate_random_noise()
        tile = LogicalTile(dense_circuit, lattice=SQUARE)
        assert square_chip.add_tile(tile, (1, 0)) is True

        restored = from_dict(to_dict(square_chip))

        assert restored.lattice == SQUARE
        assert (restored.length, restored.height) == (5, 5)
        assert len(restored.qubits) == 25
        assert restored.tiles[0].origin == (1, 0)
        assert restored.tiles[0].lattice == SQUARE
        assert {c: q.noise.p for c, q in restored.grid.items()} == {
            c: q.noise.p for c, q in square_chip.grid.items()
        }


class TestExperimentRoundTrip:
    def test_square_packing_exp_round_trips(self, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        chip.generate_random_noise()
        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.desc = "square packing on a 5x5 chip"

        restored = from_dict(to_dict(exp))

        assert isinstance(restored, SquarePackingExp)
        assert restored.desc == "square packing on a 5x5 chip"
        assert restored.profile == exp.profile
        assert {c: q.noise.p for c, q in restored.chip.grid.items()} == {
            c: q.noise.p for c, q in chip.grid.items()
        }

    def test_square_packing_profile_survives_json_round_trip(self, chip):
        # to_dict/from_dict alone keeps tuples in memory; only a real JSON
        # round-trip (as in export_flake/import_flake) degrades Coords to lists
        import json

        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))

        restored = from_dict(json.loads(json.dumps(to_dict(exp))))

        assert restored.profile == exp.profile
        assert all(isinstance(loc, tuple) for loc in restored.profile)

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
        assert set(data["exp"].keys()) == {"chip", "tile", "profile"}

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
        path = export_flake(chip, desc="written at export time")

        restored = import_flake(path)

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

        chip_path = export_flake(chip, desc="a described chip")
        tile_path = export_flake(SCTile(distance=3))

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
        restored = import_flake(setup_path)
        assert restored.source == setup_path
        restored.results = {(0.0, 0.0): {"ler": 0.001, "shots": 1000}}
        restored.config.update(shots=1000, max_errors=100)
        results_path = restored.save_results()

        record = import_flake(results_path)

        assert isinstance(record, ExperimentResults)
        assert record.experiment_ref == setup_path.name
        assert record.results == {(0, 0): {"ler": 0.001, "shots": 1000}}
        assert record.run_config["shots"] == 1000
        # the setup file itself was never replaced by the results export
        assert results_path != setup_path

    def test_sampling_settings_persist_to_both_flakes(self, data_dir, chip):
        """Every knob the sampler ran with -- including the error floor and the
        top-up ceiling -- has to survive into the setup flake *and* the results
        flake, so a stored error rate can be read back with the sampling effort
        that produced it."""
        import json

        from qsnow.experiments.sampling import ErrorFloorSampler
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        # sourced from the sampler itself, so this breaks if run_config drops a key
        exp.config.update(
            **ErrorFloorSampler(shots=1_000, min_errors=3, max_topup_shots=5_000).run_config()
        )
        exp.results = {(0, 0): {"ler": 0.001, "shots": 1000, "errors": 1}}

        setup_path = exp.save()
        results_path = exp.save_results()

        setup_config = json.loads(setup_path.read_text())["config"]
        results_config = json.loads(results_path.read_text())["run_config"]

        for config in (setup_config, results_config, import_flake(results_path).run_config):
            assert config["min_errors"] == 3
            assert config["max_topup_shots"] == 5_000
            assert config["shots"] == 1_000

    def test_disabled_error_floor_persists_as_no_ceiling(self, data_dir, chip):
        from qsnow.experiments.sampling import ErrorFloorSampler
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.config.update(**ErrorFloorSampler(shots=1_000, min_errors=0).run_config())
        exp.results = {(0, 0): {"ler": 0.0, "shots": 1000, "errors": 0}}
        exp.save()

        record = import_flake(exp.save_results())

        assert record.run_config["min_errors"] == 0
        assert record.run_config["max_topup_shots"] is None

    def test_save_with_desc_updates_description(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        path = exp.save(desc="testing desc at save time")

        assert exp.desc == "testing desc at save time"
        assert import_flake(path).desc == "testing desc at save time"

    def test_save_results_backlinks_saved_setup(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        setup_path = exp.save()
        exp.results = {(0.0, 0.0): {"ler": 0.001}}
        results_path = exp.save_results()

        assert exp.results_refs == [results_path.name]
        # the additive re-export refreshed the on-disk setup's refs
        assert import_flake(setup_path).results_refs == [results_path.name]

    def test_save_results_without_saved_setup_warns(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.results = {(0.0, 0.0): {"ler": 0.001}}

        with pytest.warns(UserWarning, match="call exp.save"):
            results_path = exp.save_results()

        assert import_flake(results_path).experiment_ref is None


class TestResultsLinkage:
    """A setup flake and its results flakes name each other relative to their own
    locations, so the link survives any working directory and a moved data folder."""

    @staticmethod
    def _saved_pair(chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.results = {(0, 0): {"ler": 0.001, "shots": 1000, "errors": 1}}
        setup_path = exp.save()
        results_path = exp.save_results()
        return exp, setup_path, results_path

    def test_reimported_setup_carries_its_results_from_any_cwd(
        self, data_dir, chip, tmp_path, monkeypatch
    ):
        # regression: refs were bare filenames resolved against the cwd, so a setup
        # imported from anywhere but the data folder silently came back with no results
        exp, setup_path, _ = self._saved_pair(chip)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        assert import_flake(setup_path).results == exp.results

    def test_moved_data_folder_keeps_the_pair_linked(self, data_dir, chip, tmp_path):
        import shutil

        exp, setup_path, _ = self._saved_pair(chip)
        moved = shutil.move(str(data_dir), str(tmp_path / "archive"))
        set_data_dir(tmp_path / "unrelated")

        restored = import_flake(Path(moved) / "experiments" / setup_path.name)

        assert restored.results == exp.results

    def test_newest_results_flake_wins(self, data_dir, chip):
        exp, setup_path, _ = self._saved_pair(chip)
        exp.results = {(0, 0): {"ler": 0.5, "shots": 10, "errors": 5}}
        exp.save_results(path=data_dir / "experiments" / "results_later.flake")

        assert import_flake(setup_path).results == exp.results

    def test_bare_filename_refs_from_older_flakes_resolve_beside_the_setup(
        self, data_dir, chip
    ):
        # pre-fix setups recorded only the filename; that is the sibling-relative
        # form, so it resolves without a migration
        exp, setup_path, results_path = self._saved_pair(chip)
        assert exp.results_refs == [results_path.name]

        assert import_flake(setup_path).results == exp.results

    def test_results_flake_names_its_setup_relative_to_itself(self, data_dir, chip):
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        exp.results = {(0, 0): {"ler": 0.001}}
        setup_path = exp.save()
        results_path = exp.save_results(path=data_dir / "experiments" / "runs" / "r.flake")

        record = import_flake(results_path)

        assert record.experiment_ref == f"../{setup_path.name}"
        assert exp.results_refs == ["runs/r.flake"]

    def test_missing_results_flake_leaves_results_empty(self, data_dir, chip):
        exp, setup_path, results_path = self._saved_pair(chip)
        results_path.unlink()

        assert import_flake(setup_path).results == {}

    def test_flake_ref_round_trips(self, tmp_path):
        from qsnow.helpers.serialize import flake_ref, resolve_flake_ref

        referrer = tmp_path / "experiments" / "setup.flake"
        target = tmp_path / "experiments" / "runs" / "r.flake"

        assert flake_ref(target, referrer) == "runs/r.flake"
        assert resolve_flake_ref(flake_ref(target, referrer), referrer) == target
        # nothing to be relative to: the absolute path is the only thing that resolves
        assert Path(flake_ref(target, None)).is_absolute()
        assert resolve_flake_ref(flake_ref(target, None), None) == target.resolve()


def test_get_timestamp_honours_its_format():
    from qsnow.helpers.serialize import get_timestamp

    # regression: the argument was accepted and ignored
    assert len(get_timestamp("%Y")) == 4


class TestJsonFileRoundTrip:
    def test_export_import_json_file(self, tmp_path, chip):
        chip.generate_random_noise()
        chip.add_tile(SCTile(distance=3), (2, 2))

        path = export_flake(chip, tmp_path / "chip.flake")
        restored = import_flake(path)

        assert isinstance(restored, Chip)
        assert str(restored.tiles[0].circuit) == str(chip.tiles[0].circuit)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            from_dict({"not_qsnow": True})


class TestAutoOrganization:
    def test_auto_paths_by_kind(self, data_dir, chip):
        tile = SCTile(distance=3)

        chip_path = export_flake(chip)
        tile_path = export_flake(tile)

        assert chip_path.parent == data_dir / "chips"
        assert chip_path.name.startswith("chip_5x5_")
        assert tile_path.parent == data_dir / "tiles"
        assert tile_path.name.startswith("tile_rsc_memory_z_d3_")

    def test_custom_label_keeps_obj_name_prefix(self, data_dir, chip):
        path = export_flake(chip, label="baseline")
        assert path.name.startswith("chip_baseline_")

    def test_import_latest_finds_newest(self, data_dir, chip):
        first = export_flake(chip, label="run")
        chip.generate_random_noise()
        export_flake(chip, label="run")
        import os

        os.utime(first, (0, 0))  # force distinct mtimes

        restored = import_latest("chip_run")

        assert {c: q.noise.p for c, q in restored.grid.items()} == {
            c: q.noise.p for c, q in chip.grid.items()
        }

    def test_list_exports_filters_by_kind_and_pattern(self, data_dir, chip):
        export_flake(chip)
        export_flake(SCTile(distance=3))

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
        chip = import_flake(self.FIXTURES / "chip_v1.flake")

        assert isinstance(chip, Chip)
        assert (chip.length, chip.height) == (10, 10)
        assert len(chip.tiles) == 1
        assert chip.tiles[0].origin == (2, 2)

    def test_v1_chip_predates_lattices_and_migrates_to_checkerboard(self):
        chip = import_flake(self.FIXTURES / "chip_v1.flake")

        assert chip.lattice == CHECKERBOARD
        assert chip.unit_dims == (5, 5)
        assert len(chip.qubits) == 50

    def test_v1_tile_golden_file_imports(self):
        tile = import_flake(self.FIXTURES / "tile_v1.flake")

        assert isinstance(tile, SCTile)
        assert tile.spec.distance == 3
        assert tile.spec.rounds == 2
        assert tile.lattice == CHECKERBOARD

    def test_nested_chip_and_tile_migrate_independently(self, chip):
        """Nested objects stamp and migrate their own format_version, so a v1
        experiment's chip/tile upgrade without the experiment migration touching them."""
        from qsnow.experiments.squarepacking.game import SquarePackingExp

        exp = SquarePackingExp(chip=chip, tile=SCTile(distance=3))
        data = to_dict(exp)

        # rewind the nested payloads to their pre-lattice v1 shape
        for nested in (data["exp"]["chip"], data["exp"]["tile"]):
            nested["format_version"] = 1
            del nested["lattice"]

        restored = from_dict(data)

        assert restored.chip.lattice == CHECKERBOARD
        assert restored.tile.lattice == CHECKERBOARD

    def test_v2_square_chip_golden_file_imports(self):
        chip = import_flake(self.FIXTURES / "chip_square_v2.flake")

        assert chip.lattice == SQUARE
        assert chip.unit_dims == (6, 6)
        assert (chip.length, chip.height) == (6, 6)
        assert len(chip.qubits) == 36
        assert chip.tiles[0].origin == (1, 2)
        assert chip.tiles[0].lattice == SQUARE

    def test_v1_experiment_golden_file_imports(self):
        exp = import_flake(self.FIXTURES / "experiment_v1.flake")

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


class TestCouplerRoundTrip:
    def test_coupler_rates_survive_a_round_trip(self, chip):
        chip.generate_random_noise()
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        restored = from_dict(to_dict(chip))

        assert {e: n.p for e, n in restored.coupler_map.items()} == {
            e: n.p for e, n in chip.coupler_map.items()
        }

    def test_a_hand_set_coupler_is_not_re_derived_on_import(self, chip):
        """The override must survive, not be recomputed from its endpoints."""
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        restored = from_dict(to_dict(chip))

        assert restored.coupler((0, 0), (1, 1)).noise.p == 0.2

    def test_channel_rule_source_round_trips(self, chip):
        chip.add_tile(SCTile(distance=3), (2, 2))

        restored = from_dict(to_dict(chip))
        cx_rule = next(
            r for r in restored.tiles[0].ruleset.rules if r.operation == "CX"
        )

        assert cx_rule.after[0].source == "coupler"


class TestChipSpecRoundTrip:
    def test_noise_model_and_coupler_mode_round_trip(self, chip):
        chip.generate_gaussian_noise(mean=0.01, deviation=0.002, seed=7)
        chip.derive_coupler_noise("max")

        restored = from_dict(to_dict(chip))

        assert restored.spec.noise_model == NoiseModelSpec(
            name="gaussian", seed=7, params={"mean": 0.01, "deviation": 0.002}
        )
        assert restored.spec.coupler_mode == "max"
        assert restored.has_independent_couplers is False

    def test_coupler_model_round_trips_with_its_rates(self, chip):
        chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3)
        chip.generate_coupler_noise(0.05, 0.01, 1.0, seed=4, correlation=0.6)

        restored = from_dict(to_dict(chip))

        assert restored.spec.coupler_model == chip.spec.coupler_model
        assert {e: n.p for e, n in restored.coupler_map.items()} == {
            e: n.p for e, n in chip.coupler_map.items()
        }
        assert restored.has_independent_couplers

    def test_spec_is_not_written_into_tag_metadata(self, chip):
        chip.generate_uniform_noise(0.01)

        data = to_dict(chip)

        assert "noise_model" not in data["tag"]["metadata"]
        assert data["spec"]["noise_model"] == {"name": "uniform homogeneous", "p": 0.01}

    def test_pre_v4_chip_derives_couplers_with_its_migrated_mode(self, chip):
        chip.generate_uniform_noise(0.01)
        chip.derive_coupler_noise("max")
        data = to_dict(chip)
        # rewrite as a v3 export: record in metadata, no spec, no coupler map
        data["format_version"] = 3
        del data["spec"]
        data["couplers"] = {}
        data["tag"]["metadata"]["noise_model"] = {"name": "uniform homogeneous", "p": 0.01}
        data["tag"]["metadata"]["coupler_model"] = {"name": "derived", "mode": "max"}

        restored = from_dict(data)

        assert restored.spec.coupler_mode == "max"
        assert restored.has_independent_couplers is False


class TestCouplerFormatVersioning:
    FIXTURES = Path(__file__).parent / "fixtures"

    def test_v4_golden_file_imports_with_its_spec(self):
        chip = import_flake(self.FIXTURES / "chip_v4.flake")

        assert chip.spec.noise_model == NoiseModelSpec(
            name="gaussian", seed=7, params={"mean": 0.01, "deviation": 0.002}
        )
        assert chip.spec.coupler_mode == "max"
        assert chip.coupler((0, 0), (1, 1)).noise.p == 0.2
        assert chip.tag.metadata == {"note": "a human-only annotation"}

    def test_v4_chip_has_no_coupler_model(self):
        """v4 predates the coupler generator; its couplers were derived or hand-set."""
        chip = import_flake(self.FIXTURES / "chip_v4.flake")

        assert chip.spec.coupler_model is None

    def test_v5_golden_file_imports_with_its_coupler_model(self):
        chip = import_flake(self.FIXTURES / "chip_v5.flake")

        assert chip.spec.noise_model.name == "skewed contour"
        assert chip.spec.coupler_model == NoiseModelSpec(
            name="correlated contour",
            seed=4,
            params={"location": 0.05, "deviation": 0.01, "skew": 1.0, "correlation": 0.6, "center": "mean", "slope": 5},
        )
        assert chip.has_independent_couplers

    def test_v3_chip_lifts_its_record_out_of_metadata(self):
        """Pre-v4 exports recorded the noise model and coupler mode in `tag.metadata`;
        the migration moves both onto `chip.spec` and leaves metadata free text."""
        chip = import_flake(self.FIXTURES / "chip_v3.flake")

        assert chip.spec.noise_model == NoiseModelSpec(name="uniform homogeneous", params={"p": 0.01})
        assert chip.spec.coupler_mode == "mean"
        assert "noise_model" not in chip.tag.metadata
        assert "coupler_model" not in chip.tag.metadata

    def test_v3_golden_file_imports_with_its_couplers(self):
        chip = import_flake(self.FIXTURES / "chip_v3.flake")

        assert chip.coupler((0, 0), (1, 1)).noise.p == 0.2
        assert chip.coupler((2, 2), (3, 3)).noise.p == 0.01

    def test_v2_chip_predates_couplers_and_derives_them_on_import(self):
        """Pre-v3 exports carry no coupler rates, so they are derived from the restored
        qubit noise - which reproduces the endpoint mean those chips were built under."""
        chip = import_flake(self.FIXTURES / "chip_square_v2.flake")

        # a 6x6 dense lattice: 2 * 6 * 5 cardinal links
        assert len(chip.couplers) == 60
        assert all(
            c.noise.p == pytest.approx(mean([chip.loc(e).noise.p for e in c.ends]))
            for c in chip.couplers
        )

    def test_v1_chip_migrates_through_to_couplers(self):
        chip = import_flake(self.FIXTURES / "chip_v1.flake")

        assert len(chip.couplers) == 81

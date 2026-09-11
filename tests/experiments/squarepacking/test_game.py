# TODO - once code is relatively stable, set this up
# class TestSquarePacking:

import pytest

from qsnow.experiments.experiment import ExperimentResults
from qsnow.experiments.squarepacking.game import SquarePackingExp
from qsnow.interface.chip import Chip
from qsnow.interface.codes.rsc import SCTile


class TestProfileCharacterization:
    """Pins the exact candidate placements produced today.

    Collected LER sweeps are keyed by these origins, so any refactor of the
    footprint/bound arithmetic must leave them untouched.
    """

    def test_small_chip_profile_is_exact(self):
        exp = SquarePackingExp(chip=Chip(5, 5), tile=SCTile(3))

        assert exp.profile == [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [(3, 85), (5, 41)],
    )
    def test_placement_counts_on_larger_chip(self, distance, expected):
        exp = SquarePackingExp(chip=Chip(10, 10), tile=SCTile(distance))

        assert len(exp.profile) == expected

    def test_profile_origins_are_all_valid_placements(self):
        exp = SquarePackingExp(chip=Chip(10, 10), tile=SCTile(3))

        # every origin the sweep offers must survive the chip's own validation
        assert all(exp.chip.is_valid_tile_placement(exp.tile, o) for o in exp.profile)


def test_interactive_styles_bundle(chip, logical_tile):
    exp = SquarePackingExp(chip=chip, tile=logical_tile)
    results = ExperimentResults(
        experiment_ref=None,
        run_config={},
        results={(0, 0): {"ler": 0.01, "shots": 100, "errors": 1}},
    )
    styles = exp._interactive_styles(results)
    assert list(styles) == ["PER", "Candidate Placements", "Avg. PER", "LER"]
    assert styles["LER"].colorbar.label == "LER"
    assert all(style.desc for style in styles.values())


def _strong_id_map(exp, batch_size):
    """Run a tiny experiment and return {loc: strong_id} for every placement."""
    # min_errors=0: the `chip` fixture is noiseless, so every placement would
    # otherwise spend the full top-up budget chasing an error that cannot happen.
    results = exp.run(shots=100, max_errors=100, min_errors=0, batch_size=batch_size)
    return {loc: r["strong_id"] for loc, r in results.items()}


def test_run_chunking_is_result_identical(chip, logical_tile):
    """Sampling placements in chunks must cover exactly the same tasks and map
    each placement to the same task as a single-batch collect. ``strong_id`` is
    a deterministic hash of (circuit, decoder), so it is invariant to how tasks
    are batched even though the sampled error counts are stochastic."""
    profile = SquarePackingExp(chip=chip, tile=logical_tile).profile
    assert len(profile) > 1  # otherwise chunking is not exercised

    single_batch = _strong_id_map(
        SquarePackingExp(chip=chip, tile=logical_tile), batch_size=len(profile) + 5
    )
    chunked = _strong_id_map(
        SquarePackingExp(chip=chip, tile=logical_tile), batch_size=1
    )

    assert set(single_batch) == set(profile)
    assert chunked == single_batch


def test_run_rejects_nonpositive_batch_size(chip, logical_tile):
    import pytest

    exp = SquarePackingExp(chip=chip, tile=logical_tile)
    with pytest.raises(ValueError):
        exp.run(shots=10, max_errors=10, min_errors=0, batch_size=0)


def test_run_records_sampling_settings_in_config(chip, logical_tile):
    """The error-floor settings decide how much sampling a result reflects, so they
    have to ride along in the config that gets serialized with it."""
    exp = SquarePackingExp(chip=chip, tile=logical_tile)
    # tiny ceiling: the `chip` fixture is noiseless, so the top-up pass can never
    # find an error and would otherwise spend the full default budget
    exp.run(shots=10, max_errors=10, min_errors=1, max_topup_shots=10)

    assert exp.config["min_errors"] == 1
    assert exp.config["max_topup_shots"] == 10
    assert exp.config["shots"] == 10
    # the top-up pass is timed separately from the first sampling pass
    assert "topup" in exp.config["stats"]["runtime"]


def test_run_records_no_topup_ceiling_when_floor_is_disabled(chip, logical_tile):
    exp = SquarePackingExp(chip=chip, tile=logical_tile)
    exp.run(shots=10, max_errors=10, min_errors=0)

    assert exp.config["min_errors"] == 0
    assert exp.config["max_topup_shots"] is None
    assert "topup" not in exp.config["stats"]["runtime"]


class TestRunLeavesTheSetupUntouched:
    """`run()` reads as a query on the setup, so it must not leave sweep state behind."""

    def test_run_takes_the_tile_back_off_the_chip(self, chip, logical_tile):
        exp = SquarePackingExp(chip=chip, tile=logical_tile)

        exp.run(shots=10, max_errors=10, min_errors=0)

        assert exp.chip.tiles == []
        assert not exp.tile.initialized()

    def test_run_can_be_repeated(self, chip, logical_tile):
        # regression: the tile stayed at the last placement, so a second run found
        # its first placement occupied by itself
        exp = SquarePackingExp(chip=chip, tile=logical_tile)

        first = dict(exp.run(shots=10, max_errors=10, min_errors=0))
        second = exp.run(shots=10, max_errors=10, min_errors=0)

        assert set(second) == set(first) == set(exp.profile)

    def test_save_after_run_does_not_embed_the_sweep_tile(self, chip, logical_tile):
        from qsnow.helpers.serialize import to_dict

        exp = SquarePackingExp(chip=chip, tile=logical_tile)
        exp.run(shots=10, max_errors=10, min_errors=0)

        data = to_dict(exp)

        assert data["exp"]["chip"]["tiles"] == []

    def test_rejected_profile_placement_raises_instead_of_sampling_a_stale_circuit(
        self, chip, logical_tile
    ):
        # regression: add_tile's False was ignored, so a placement the chip refused
        # was sampled with whatever circuit the tile last held
        exp = SquarePackingExp(chip=chip, tile=logical_tile)
        blocker = logical_tile.copy()
        assert exp.chip.add_tile(blocker, exp.profile[0])

        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="not valid on the chip"):
                exp.run(shots=10, max_errors=10, min_errors=0)

        # the failed run still released the sweep tile; the blocker is untouched
        assert exp.chip.tiles == [blocker]
        assert not exp.tile.initialized()


class TestShowAcceptsEitherResultsShape:
    """`run()` returns a dict and `import_flake` returns an `ExperimentResults`;
    every display surface has to take both, and default to the experiment's own."""

    @pytest.fixture
    def shown(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "qsnow.experiments.squarepacking.game.visualize_interactive",
            lambda chip, styles, **kw: seen.update(chip=chip, styles=styles, **kw),
        )
        return seen

    def test_results_record_keeps_a_record_and_wraps_a_dict_with_the_config(
        self, chip, logical_tile
    ):
        exp = SquarePackingExp(chip=chip, tile=logical_tile)
        exp.config["shots"] = 10
        live = {(0, 0): {"ler": 0.01}}
        record = ExperimentResults(experiment_ref=None, run_config={}, results=live)

        assert exp._results_record(record) is record
        wrapped = exp._results_record(live)
        assert wrapped.results is live
        assert wrapped.run_config["shots"] == 10
        exp.results = live
        assert exp._results_record().results is live

    def test_show_defaults_to_the_experiments_own_results(
        self, chip, logical_tile, shown
    ):
        exp = SquarePackingExp(chip=chip, tile=logical_tile)
        exp.results = {loc: {"ler": 0.01} for loc in exp.profile}

        exp.show()

        assert shown["chip"] is exp.chip
        assert "LER" in shown["styles"]

    def test_show_accepts_the_dict_run_returns(self, chip, logical_tile, shown):
        # regression: show() reached for `.results` on the dict and raised
        exp = SquarePackingExp(chip=chip, tile=logical_tile)
        results = exp.run(shots=10, max_errors=10, min_errors=0)

        exp.show(results)

        assert shown["chip"] is exp.chip

    def test_html_export_accepts_the_dict_run_returns(self, tmp_path, chip, logical_tile):
        from qsnow.visualize.interactive import export_square_packing

        exp = SquarePackingExp(chip=chip, tile=logical_tile)
        exp.results = {loc: {"ler": 0.01} for loc in exp.profile}

        path = export_square_packing(
            exp, exp.results, tmp_path / "sp.html", include_plotlyjs="cdn"
        )

        assert path.exists()

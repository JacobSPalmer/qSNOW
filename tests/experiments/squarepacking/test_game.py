# TODO - once code is relatively stable, set this up
# class TestSquarePacking:

from qsnow.experiments.experiment import ExperimentResults
from qsnow.experiments.squarepacking.game import SquarePackingExp


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
    results = exp.run(shots=100, max_errors=100, batch_size=batch_size)
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
        exp.run(shots=10, max_errors=10, batch_size=0)

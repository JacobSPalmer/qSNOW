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
    assert list(styles) == ["PER", "Valid Placements", "LER"]
    assert styles["LER"].colorbar.label == "LER"

import importlib.util
import sys
from pathlib import Path

import pytest

# `src/cli` is a script directory, not part of the installed package, so load the
# runner from its path rather than relying on `src` being importable.
_RUNNER = Path(__file__).resolve().parents[2] / "src" / "cli" / "run_experiment.py"
_spec = importlib.util.spec_from_file_location("run_experiment", _RUNNER)
run_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiment)

DERIVED = run_experiment.DERIVED
distribution_from_flags = run_experiment.distribution_from_flags
main = run_experiment.main
run_and_serialize_spp_experiment = run_experiment.run_and_serialize_spp_experiment
save_configuration = run_experiment.save_configuration
from qsnow.helpers import serialize
from qsnow.interface.chip import Chip
from qsnow.interface.noise import NormalContour, RandomGaussian, SkewContour, Uniform

SMALL = dict(distances=[3], shots=50, max_errors=None, min_errors=1, shot_ceiling=None, additional_label="t")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path):
    yield
    serialize.set_data_dir()


def _run(chip, tmp_path, **kw):
    params = dict(mean=0.01, deviation=0.003, skew=1.5, center="mean", seed=3, noise_model="skewed-contour")
    params.update(kw)
    return run_and_serialize_spp_experiment(chip, data_directory=tmp_path / "data", **SMALL, **params)


class TestDistributionFromFlags:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("gaussian", RandomGaussian(0.01, 0.003, seed=3)),
            ("derived-contour", NormalContour(0.01, 0.003, seed=3)),
            ("skewed-contour", SkewContour(0.01, 0.003, 1.5, "median", seed=3)),
            ("uniform", Uniform(0.01)),
        ],
    )
    def test_maps_each_model_name_to_its_class(self, model, expected):
        assert distribution_from_flags(model, 0.01, 0.003, 1.5, "median", 3) == expected

    def test_unknown_model_raises_listing_the_choices(self):
        with pytest.raises(ValueError, match="gaussian"):
            distribution_from_flags("lognormal", 0.01)


class TestCouplerFlags:
    def test_default_keeps_couplers_derived(self, tmp_path):
        chip = _run(Chip(5, 5), tmp_path)
        assert chip.spec.coupler_model is None
        assert not chip.has_independent_couplers

    def test_coupler_model_and_correlation_are_applied_and_recorded(self, tmp_path):
        chip = _run(
            Chip(5, 5), tmp_path,
            coupler_model="skewed-contour", coupler_mean=0.05, coupler_deviation=0.01,
            coupler_skew=1.0, coupler_seed=4, correlation=0.6,
        )
        assert chip.spec.coupler_model == SkewContour(0.05, 0.01, 1.0, seed=4)
        assert chip.spec.coupler_correlation == 0.6
        assert chip.has_independent_couplers

    def test_saved_experiment_carries_the_coupler_record(self, tmp_path):
        _run(Chip(5, 5), tmp_path, coupler_model="gaussian", coupler_mean=0.05, coupler_deviation=0.01, coupler_seed=4, correlation=0.5)
        exp = serialize.import_latest("*t_d3*")
        assert exp.chip.spec.coupler_model == RandomGaussian(0.05, 0.01, seed=4)
        assert exp.chip.spec.coupler_correlation == 0.5

    def test_a_chip_with_its_own_coupler_model_is_not_regenerated(self, tmp_path):
        chip = Chip(5, 5)
        chip.generate_noise(SkewContour(0.01, 0.003, 1.5, seed=3))
        chip.generate_coupler_noise(Uniform(0.2))
        before = [c.noise.p for c in chip.couplers]
        _run(chip, tmp_path, coupler_model="gaussian", coupler_mean=0.05, coupler_deviation=0.01)
        assert [c.noise.p for c in chip.couplers] == before
        assert chip.spec.coupler_model == Uniform(0.2)

    def test_missing_coupler_mean_raises(self, tmp_path):
        with pytest.raises(ValueError, match="coupler_mean"):
            _run(Chip(5, 5), tmp_path, coupler_model="gaussian")


class TestSaveConfiguration:
    def test_records_the_resolved_seeds_and_every_coupler_flag(self, tmp_path):
        chip = _run(
            Chip(5, 5), tmp_path, seed=None,
            coupler_model="skewed-contour", coupler_mean=0.05, coupler_deviation=0.01,
            coupler_skew=1.0, coupler_seed=None, correlation=0.6,
        )
        args = {
            "name": "cfg", "model": "skewed-contour", "mean": 0.01, "seed": None,
            "coupler_model": "skewed-contour", "coupler_mean": 0.05, "coupler_seed": None,
            "correlation": 0.6,
        }
        save_configuration(dict(args), chip, "cfg")

        text = (serialize.get_data_dir() / "cfg.txt").read_text().split()
        assert text[text.index("--seed") + 1] == str(chip.spec.noise_model.seed)
        assert text[text.index("--coupler_seed") + 1] == str(chip.spec.coupler_model.seed)
        assert text[text.index("--correlation") + 1] == "0.6"


class TestMainValidation:
    def _main(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", *argv])
        with pytest.raises(SystemExit) as info:
            main()
        return info.value.code

    def test_coupler_model_without_mean_is_a_parser_error(self, monkeypatch):
        code = self._main(monkeypatch, ["--mean", "0.01", "--distances", "3", "--coupler_model", "gaussian"])
        assert code == 2

    def test_correlation_outside_unit_interval_is_a_parser_error(self, monkeypatch):
        code = self._main(monkeypatch, ["--mean", "0.01", "--distances", "3", "--correlation", "1.5"])
        assert code == 2

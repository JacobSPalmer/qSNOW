import pytest

from qsnow.experiments.experiment import Experiment
from qsnow.experiments.progress import PhasedProgress


def make_progress(phases=None, title="Test"):
    # disable=True suppresses the tqdm bar; row bookkeeping still runs.
    return PhasedProgress(title, phases=phases, disable=True)


def test_phase_records_completion():
    with make_progress(phases=2) as prog:
        with prog.phase("first", total=10) as ph:
            ph.advance(10)
        assert prog.tasks[0].completed == 10
        assert prog.tasks[0].finished

        with prog.phase("second", total=5) as ph:
            ph.advance(5)
        assert prog.tasks[1].finished

    assert len(prog.tasks) == 2


def test_phase_ending_early_is_shrunk_to_finished():
    with make_progress(phases=1) as prog:
        with prog.phase("sampling", total=100) as ph:
            ph.advance(30)  # e.g. sinter stopped at max_errors
        task = prog.tasks[0]
        assert task.total == 30
        assert task.finished


def test_phase_exception_leaves_phase_unfinished():
    with pytest.raises(ValueError):
        with make_progress(phases=2) as prog:
            with prog.phase("boom", total=10):
                raise ValueError("boom")
    assert prog.tasks[0].completed == 0
    assert not prog.tasks[0].finished


def test_track_yields_items_and_counts_them():
    with make_progress(phases=1) as prog:
        seen = list(prog.track([1, 2, 3], "items"))
        assert seen == [1, 2, 3]
        task = prog.tasks[0]
        assert task.total == 3
        assert task.completed == 3
        assert task.finished


def test_track_without_len_still_finishes():
    with make_progress(phases=1) as prog:
        seen = list(prog.track((i for i in range(4)), "gen items"))
        assert seen == [0, 1, 2, 3]
        task = prog.tasks[0]
        assert task.total == 4
        assert task.finished


def test_phase_label_includes_index_when_phase_count_known():
    with make_progress(phases=2, title="Run") as prog:
        with prog.phase("Generating", total=1) as ph:
            ph.advance(1)
        assert prog.tasks[0].description == "[1/2] Generating"


def test_phase_label_plain_without_phase_count():
    with make_progress(phases=None) as prog:
        with prog.phase("Generating", total=1) as ph:
            ph.advance(1)
        assert prog.tasks[0].description == "Generating"


def test_experiment_progress_defaults_title_to_class_name():
    class DummyExp(Experiment):
        pass

    prog = DummyExp().progress(phases=2, disable=True)
    with prog:
        with prog.phase("x", total=1) as ph:
            ph.advance(1)
    assert "DummyExp" in prog._title

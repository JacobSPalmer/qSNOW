import pytest

from qsnow.experiments.experiment import Experiment
from qsnow.experiments.progress import PhasedProgress, PhasedTimeColumn


def make_progress(phases=None, title="Test"):
    # disable=True suppresses terminal rendering; task bookkeeping still runs
    return PhasedProgress(title, phases=phases, disable=True)


def test_overall_row_created_with_title_and_phase_total():
    with make_progress(phases=3, title="My run") as prog:
        overall = prog.tasks[0]
        assert "My run" in overall.description
        assert overall.total == 3
        assert overall.completed == 0


def test_phase_advances_and_completes_overall():
    with make_progress(phases=2) as prog:
        with prog.phase("first", total=10) as ph:
            ph.advance(10)
        assert prog.tasks[0].completed == 1

        with prog.phase("second", total=5) as ph:
            ph.advance(5)

        overall, first, second = prog.tasks
        assert overall.finished
        assert first.finished and second.finished


def test_phase_ending_early_is_shrunk_to_finished():
    with make_progress(phases=1) as prog:
        with prog.phase("sampling", total=100) as ph:
            ph.advance(30)  # e.g. sinter stopped at max_errors
        task = prog.tasks[1]
        assert task.total == 30
        assert task.finished


def test_phase_exception_does_not_advance_overall():
    with pytest.raises(ValueError):
        with make_progress(phases=2) as prog:
            with prog.phase("boom", total=10):
                raise ValueError("boom")
    assert prog.tasks[0].completed == 0


def test_track_yields_items_and_counts_them():
    with make_progress(phases=1) as prog:
        seen = list(prog.track([1, 2, 3], "items"))
        assert seen == [1, 2, 3]
        task = prog.tasks[1]
        assert task.total == 3
        assert task.completed == 3
        assert prog.tasks[0].completed == 1


def test_track_without_len_still_finishes():
    with make_progress(phases=1) as prog:
        seen = list(prog.track((i for i in range(4)), "gen items"))
        assert seen == [0, 1, 2, 3]
        task = prog.tasks[1]
        assert task.total == 4
        assert task.finished


def test_overall_finalized_on_clean_exit_even_if_phases_overdeclared():
    with make_progress(phases=3) as prog:
        with prog.phase("only phase", total=1) as ph:
            ph.advance(1)
    overall = prog.tasks[0]
    assert overall.total == 1
    assert overall.finished


def test_overall_left_unfinished_on_exception():
    with pytest.raises(ValueError):
        with make_progress(phases=2) as prog:
            with prog.phase("boom", total=10):
                raise ValueError("boom")
    assert not prog.tasks[0].finished


def test_time_column_shows_elapsed_for_overall_and_remaining_for_phases():
    column = PhasedTimeColumn()
    with make_progress(phases=1) as prog:
        with prog.phase("work", total=10):
            overall, phase_task = prog.tasks
            assert overall.fields.get("overall") is True
            # overall renders elapsed (0:00:00 style), never the remaining
            # placeholder; an un-estimated phase renders the placeholder
            assert str(column.render(overall)) != "-:--:--"
            assert str(column.render(phase_task)) == "-:--:--"


def test_experiment_progress_defaults_title_to_class_name():
    class DummyExp(Experiment):
        pass

    prog = DummyExp().progress(phases=2, disable=True)
    with prog:
        assert "DummyExp" in prog.tasks[0].description

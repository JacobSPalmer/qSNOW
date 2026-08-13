"""Tests for the reusable sinter sampling policy.

The two-pass logic is exercised through an injected ``collect_fn`` so the
behaviour under test (which tasks get resampled, how stats merge) is decided by
the sampler rather than by stochastic sampling, and no worker processes are spawned.
"""
import pytest
import sinter
import stim

from qsnow.experiments.sampling import (
    DEFAULT_TOPUP_SHOT_MULTIPLIER,
    ErrorFloorSampler,
    _metadata_key,
)

TRIVIAL_CIRCUIT = stim.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")


def _task(loc):
    return sinter.Task(circuit=TRIVIAL_CIRCUIT, json_metadata={"loc": loc})


def _loc(task_or_stat):
    return tuple(task_or_stat.json_metadata["loc"])


class FakeCollector:
    """Stands in for ``sinter.collect``, returning scripted per-task results.

    ``pass_results`` maps a placement to the (shots, errors) it records on each
    successive pass it takes part in.
    """

    def __init__(self, pass_results):
        self.pass_results = pass_results
        self.calls = []
        self._passes_taken = {}

    def __call__(self, *, num_workers, tasks, decoders, max_shots, max_errors):
        self.calls.append(
            {
                "locs": [_loc(t) for t in tasks],
                "max_shots": max_shots,
                "max_errors": max_errors,
                "decoders": decoders,
                "num_workers": num_workers,
            }
        )
        out = []
        for task in tasks:
            loc = _loc(task)
            # counted per task, so batching a pass across several calls does not
            # advance a task past its next scripted result
            pass_index = self._passes_taken.get(loc, 0)
            self._passes_taken[loc] = pass_index + 1
            shots, errors = self.pass_results[loc][pass_index]
            out.append(
                sinter.TaskStats(
                    strong_id=f"sid-{loc}",
                    decoder=decoders[0],
                    json_metadata=task.json_metadata,
                    shots=shots,
                    errors=errors,
                )
            )
        return out


def _sampler(collector, **kwargs):
    kwargs.setdefault("shots", 1_000)
    kwargs.setdefault("batch_size", 64)
    return ErrorFloorSampler(collect_fn=collector, **kwargs)


# ----------------------------------------------------------------------
# Top-up targeting
# ----------------------------------------------------------------------


def test_topup_targets_only_tasks_below_the_error_floor():
    tasks = [_task((0, 0)), _task((0, 2)), _task((2, 0))]
    collector = FakeCollector(
        {
            (0, 0): [(1_000, 0), (5_000, 1)],  # quiet: needs a top-up
            (0, 2): [(1_000, 7)],  # already clears the floor
            (2, 0): [(1_000, 0), (2_000, 3)],  # quiet: needs a top-up
        }
    )
    _sampler(collector, min_errors=1).collect(tasks)

    assert len(collector.calls) == 2
    assert collector.calls[1]["locs"] == [(0, 0), (2, 0)]


def test_topup_pass_uses_the_error_floor_and_shot_ceiling():
    tasks = [_task((0, 0))]
    collector = FakeCollector({(0, 0): [(1_000, 0), (9_000, 3)]})
    _sampler(collector, shots=1_000, max_errors=500, min_errors=3).collect(tasks)

    first, second = collector.calls
    assert (first["max_shots"], first["max_errors"]) == (1_000, 500)
    assert (second["max_shots"], second["max_errors"]) == (
        1_000 * DEFAULT_TOPUP_SHOT_MULTIPLIER,
        3,
    )


def test_max_topup_shots_overrides_the_default_ceiling():
    tasks = [_task((0, 0))]
    collector = FakeCollector({(0, 0): [(1_000, 0), (77, 1)]})
    _sampler(collector, max_topup_shots=77).collect(tasks)

    assert collector.calls[1]["max_shots"] == 77


def test_min_errors_zero_skips_the_topup_pass_entirely():
    tasks = [_task((0, 0)), _task((0, 2))]
    collector = FakeCollector({(0, 0): [(1_000, 0)], (0, 2): [(1_000, 4)]})
    run = _sampler(collector, min_errors=0).collect(tasks)

    assert len(collector.calls) == 1
    assert "topup" not in run.timings
    assert {_loc(s): s.errors for s in run.stats} == {(0, 0): 0, (0, 2): 4}


# ----------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------


def test_merged_stats_accumulate_shots_and_errors_across_passes():
    tasks = [_task((0, 0)), _task((0, 2))]
    collector = FakeCollector(
        {
            (0, 0): [(1_000, 0), (40_000, 2)],
            (0, 2): [(1_000, 6)],
        }
    )
    run = _sampler(collector).collect(tasks)

    merged = {_loc(s): s for s in run.stats}
    assert (merged[(0, 0)].shots, merged[(0, 0)].errors) == (41_000, 2)
    # untouched by the top-up pass
    assert (merged[(0, 2)].shots, merged[(0, 2)].errors) == (1_000, 6)


def test_every_task_is_represented_exactly_once_in_the_result():
    tasks = [_task((0, 0)), _task((0, 2)), _task((2, 0))]
    collector = FakeCollector(
        {
            (0, 0): [(1_000, 0), (10, 1)],
            (0, 2): [(1_000, 3)],
            (2, 0): [(1_000, 0), (10, 1)],
        }
    )
    run = _sampler(collector).collect(tasks)

    assert sorted(_loc(s) for s in run.stats) == [(0, 0), (0, 2), (2, 0)]


def test_batching_does_not_change_the_merged_result():
    locs = [(0, 0), (0, 2), (2, 0), (2, 2)]
    plan = {
        (0, 0): [(1_000, 0), (500, 1)],
        (0, 2): [(1_000, 5)],
        (2, 0): [(1_000, 0), (500, 1)],
        (2, 2): [(1_000, 9)],
    }

    def collect_with(batch_size):
        run = _sampler(
            FakeCollector({k: list(v) for k, v in plan.items()}),
            batch_size=batch_size,
        ).collect([_task(loc) for loc in locs])
        return {_loc(s): (s.shots, s.errors) for s in run.stats}

    assert collect_with(1) == collect_with(64)


# ----------------------------------------------------------------------
# Stalled placements
# ----------------------------------------------------------------------


def test_tasks_still_at_zero_errors_after_the_ceiling_are_reported_as_stalled():
    tasks = [_task((0, 0)), _task((0, 2))]
    collector = FakeCollector(
        {
            (0, 0): [(1_000, 0), (20_000, 0)],  # noiseless: never errors
            (0, 2): [(1_000, 0), (300, 1)],
        }
    )
    run = _sampler(collector).collect(tasks)

    assert run.stalled == [{"loc": (0, 0)}]


def test_timings_cover_both_passes():
    tasks = [_task((0, 0))]
    collector = FakeCollector({(0, 0): [(1_000, 0), (10, 1)]})
    run = _sampler(collector).collect(tasks)

    assert set(run.timings) == {"sampling", "topup"}
    assert all(v >= 0 for v in run.timings.values())


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_duplicate_metadata_is_rejected_with_a_clear_error():
    collector = FakeCollector({})
    with pytest.raises(ValueError, match="unique json_metadata"):
        _sampler(collector).collect([_task((0, 0)), _task((0, 0))])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"shots": 0},
        {"min_errors": -1},
        {"max_topup_shots": 0},
    ],
)
def test_invalid_settings_are_rejected_eagerly(kwargs):
    with pytest.raises(ValueError):
        ErrorFloorSampler(**kwargs)


def test_phase_count_reflects_whether_a_topup_runs():
    assert ErrorFloorSampler(min_errors=1).phases == 2
    assert ErrorFloorSampler(min_errors=0).phases == 1


def test_run_config_records_the_resolved_topup_ceiling():
    assert ErrorFloorSampler(shots=100, min_errors=1).run_config()[
        "max_topup_shots"
    ] == 100 * DEFAULT_TOPUP_SHOT_MULTIPLIER
    assert ErrorFloorSampler(shots=100, min_errors=0).run_config()["max_topup_shots"] is None


def test_metadata_key_is_stable_across_the_tuple_list_round_trip():
    """sinter may hand metadata back with tuples turned into lists; the key a stat
    is filed under must not depend on which form it arrives in."""
    assert _metadata_key({"loc": (0, 2)}) == _metadata_key({"loc": [0, 2]})


# ----------------------------------------------------------------------
# Against real sinter
# ----------------------------------------------------------------------


def test_error_floor_is_met_against_real_sinter():
    """The point of the whole exercise: a circuit quiet enough to record zero errors
    in the first pass must not be left at a 0% error rate."""
    quiet = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.002,
    )
    noisy = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.06,
    )
    tasks = [
        sinter.Task(circuit=quiet, json_metadata={"loc": (0, 0)}),
        sinter.Task(circuit=noisy, json_metadata={"loc": (0, 2)}),
    ]

    run = ErrorFloorSampler(
        shots=200, max_errors=None, min_errors=1, max_topup_shots=2_000_000
    ).collect(tasks)

    assert not run.stalled
    assert all(s.errors >= 1 for s in run.stats)
    assert all(s.shots >= 200 for s in run.stats)

import copy
from pathlib import Path

import pytest

from qsnow.helpers import migrations
from qsnow.helpers.migrations import (
    _MIGRATIONS,
    _format_version,
    _migrate,
    _read_version,
    _version_str,
    _version_dict,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _stamped(version, **fields):
    major, minor, patch = version
    return {
        "__qsnow__": "Chip",
        "format_version": {"major": major, "minor": minor, "patch": patch},
        **fields,
    }


@pytest.fixture
def install_at(monkeypatch):
    """Pretend this install writes `version`, with no migrations registered."""

    def set_version(version):
        major, minor, patch = version
        monkeypatch.setattr(migrations, "FORMAT_VERSION_MAJOR", major)
        monkeypatch.setattr(migrations, "FORMAT_VERSION_MINOR", minor)
        monkeypatch.setattr(migrations, "FORMAT_VERSION_PATCH", patch)
        monkeypatch.setattr(migrations, "_MIGRATIONS", {})

    return set_version


class TestReadVersion:
    def test_subdict_stamp(self):
        assert _read_version(_stamped((1, 2, 3))) == (1, 2, 3)

    def test_missing_patch_means_zero(self):
        assert _read_version({"format_version": {"major": 0, "minor": 9}}) == (0, 9, 0)

    def test_legacy_integer_is_a_minor_of_major_zero(self):
        assert _read_version({"format_version": 7}) == (0, 7, 0)

    def test_unstamped_flake_is_v0_1_0(self):
        assert _read_version({}) == (0, 1, 0)

    @pytest.mark.parametrize(
        "stamp",
        [
            {"major": 0},
            {"major": 0, "minor": True},
            {"major": -1, "minor": 0},
            {"major": "0", "minor": 9},
            True,
            "0.9.0",
            0.9,
        ],
    )
    def test_malformed_stamp_raises(self, stamp):
        with pytest.raises(ValueError):
            _read_version({"format_version": stamp})


class TestMigrate:
    @pytest.mark.parametrize("newer", [(0, 9, 1), (0, 10, 0), (1, 0, 0)])
    def test_newer_flake_raises_naming_its_writer(self, install_at, newer):
        install_at((0, 9, 0))

        with pytest.raises(
            ValueError, match=r"written by qsnow 9\.9\.9.*Upgrade qsnow"
        ):
            _migrate(_stamped(newer, written_by="qsnow 9.9.9"))

    def test_pending_steps_apply_in_version_order(self, install_at):
        install_at((1, 0, 0))
        applied = []
        for target in [(0, 10, 0), (0, 8, 0), (1, 0, 0), (0, 9, 0), (1, 0, 1)]:
            migrations._MIGRATIONS[target] = lambda d, t=target: applied.append(t) or d

        _migrate(_stamped((0, 8, 0)))

        # (0, 8, 0) is the flake's own version and (1, 0, 1) is beyond the install
        assert applied == [(0, 9, 0), (0, 10, 0), (1, 0, 0)]

    def test_bump_without_a_step_restamps_untouched(self, install_at):
        """A minor or patch bump leaves old flakes readable, so it registers no step."""
        install_at((0, 9, 1))

        data = _migrate(_stamped((0, 9, 0), length=5))

        assert data["format_version"] == {"major": 0, "minor": 9, "patch": 1}
        assert data["length"] == 5

    def test_legacy_integer_is_restamped_as_a_subdict(self):
        data = _migrate({"__qsnow__": "Chip", "format_version": 7})

        major, minor, patch = (0, 9, 0)
        assert data["format_version"] == _version_dict((major, minor, patch))

    def test_migrations_move_to_current_version(self, monkeypatch):
        """Test that there are not empty versions such that the current version is higher than any migration upgrade
        """

        major, minor, patch = _format_version()
        monkeypatch.setattr(migrations, "FORMAT_VERSION_MAJOR", major)
        monkeypatch.setattr(migrations, "FORMAT_VERSION_MINOR", minor + 2)
        monkeypatch.setattr(migrations, "FORMAT_VERSION_PATCH", patch)

        with pytest.raises(ValueError):
            _migrate({"__qsnow__": "Chip", "format_version": 7})

    def test_written_by_is_left_as_the_original_writer(self):
        data = _migrate(_stamped(_format_version(), written_by="qsnow 0.0.1"))

        assert data["written_by"] == "qsnow 0.0.1"

    def test_input_dict_is_not_mutated(self):
        original = {"__qsnow__": "SquarePackingExp", "format_version": 7}
        untouched = copy.deepcopy(original)

        _migrate(original)

        assert original == untouched


class TestRegistryGuards:
    def test_no_migration_targets_a_version_beyond_this_install(self):
        """A step registered without bumping the version constants would never run."""
        assert max(_MIGRATIONS) <= _format_version()

    def test_current_version_has_a_golden_chip_fixture(self):
        assert (FIXTURES / f"chip_{_version_str(_format_version())}.flake").exists()

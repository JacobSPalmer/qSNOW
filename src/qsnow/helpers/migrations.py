"""
Flake format versioning and the migrations that upgrade old flakes on import.

Every export is stamped with the format version it was written at and the qsnow
release that wrote it:

    "format_version": {"major": 0, "minor": 9, "patch": 0},
    "written_by": "qsnow 0.1.1"

Flakes written before v0.9.0 carry a single integer `format_version: N`, read as
v0.N.0; the earliest carry no version at all and are v0.1.0. Old flakes are upgraded
in memory on import and never rewritten on disk.

Versions follow Serialization Versioning, the Semantic Versioning variant for stored
data (https://github.com/dgraph-io/badger/blob/main/VERSIONING.md). Given a format
version MAJOR.MINOR.PATCH, increment the:
    MAJOR version for changes that require transforming a flake before it can be
          used again,
    MINOR version when old flakes are still readable as written,
    PATCH version for backward-compatible bug fixes.

While MAJOR is 0 the format is in initial development and anything may change
(SemVer item 4), which is why v0.2.0-v0.8.0 are minor bumps that still migrate. From
v1.0.0 on, a change that needs a migration is a major bump.

Any change to an export's structure must:
    1. Bump FORMAT_VERSION_MAJOR, _MINOR or _PATCH per the rules above,
    2. Register a `@_migration(to_version)` step if old flakes need transforming, and
    3. Check in a golden fixture named for the new version
       (e.g. tests/helpers/fixtures/chip_v0.9.0.flake).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from importlib import metadata
from typing import Any, Dict

from qsnow.interface.lattice import CHECKERBOARD

FORMAT_VERSION_MAJOR = 0
FORMAT_VERSION_MINOR = 9
FORMAT_VERSION_PATCH = 0

# NOTE - As a sanity check for future Jacob mostly, I am including patch for future
# purposes but likely won't need it while doing lots of active development so assume that
# if patch number is excluded, it's assumed to be 0. (i.e. if you see v0.x that is equivalent
# to saying v0.x.0)

try:
    _QSNOW_VERSION = metadata.version("qsnow")
except metadata.PackageNotFoundError:
    _QSNOW_VERSION = "unknown"

Version = tuple[int, int, int]
_MIGRATIONS: Dict[Version, Callable[[Dict], Dict]] = {}


# TODO - (code - helper - low) at some point create a version class to keep some sort of changelog and in which all this can be encapsulated in
def _format_version() -> Version:
    """The (major, minor, patch) version this install writes."""
    return (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR, FORMAT_VERSION_PATCH)


def _version_str(v: Version) -> str:
    return "v" + ".".join(str(part) for part in v)


def _version_dict(v: Version) -> Dict[str, int]:
    major, minor, patch = v
    return {"major": major, "minor": minor, "patch": patch}

#TODO (clean - medium) - remove this written_by and replace with just a "qsnow_version". also decide either a dict or char str format for versioning
# and change both the qsnow vers and format versioning
def _export_stamp() -> Dict[str, Any]:
    """The version fields every export dict carries."""
    return {
        "format_version": _version_dict(_format_version()),
        "written_by": f"qsnow {_QSNOW_VERSION}",
    }


def _is_version_part(value: Any) -> bool:
    # bool is an int subclass, so `true` in a flake would otherwise read as 1
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _extract_version(stamp: Dict[str, Any]) -> Version:
    """Read a `{"major", "minor", "patch"}` stamp; a missing patch means 0."""
    parts = (stamp.get("major"), stamp.get("minor"), stamp.get("patch", 0))
    if not all(_is_version_part(p) for p in parts):
        raise ValueError(
            "Format version must be non-negative integers {major, minor, patch}; "
            f"got {stamp!r}."
        )
    return parts


def _read_version(data: Dict) -> Version:
    """
    The (major, minor, patch) version a persisted dict was written at.

    Exports before v0.9.0 carry a single integer `format_version: N`, read as v0.N.0;
    the earliest exports carry no version at all and are v0.1.0.
    """
    if "format_version" not in data:
        return (0, 1, 0)
    stamp = data["format_version"]
    if isinstance(stamp, dict):
        return _extract_version(stamp)
    if _is_version_part(stamp):
        return (0, stamp, 0)
    raise ValueError(
        f"Unrecognized format_version {stamp!r}; expected an integer or a "
        "{major, minor, patch} dict."
    )


def _migration(to_version: Version):
    """Decorator registering the step that upgrades a flake to `to_version`."""

    def register(fn: Callable[[Dict], Dict]) -> Callable[[Dict], Dict]:
        _MIGRATIONS[to_version] = fn
        return fn

    return register


def _migrate(data: Dict) -> Dict:
    """
    A copy of a persisted dict upgraded to the current format version.

    Applies, in order, every registered step newer than the flake and no newer than
    this install. A version bump with no step needs no transform.
    """
    data = copy.deepcopy(data)
    found, current = _read_version(data), _format_version()
    if found > current:
        writer = f" (written by {data['written_by']})" if "written_by" in data else ""
        raise ValueError(
            f"Export uses format {_version_str(found)}{writer}, newer than this "
            f"install's {_version_str(current)} (qsnow {_QSNOW_VERSION}). "
            "Upgrade qsnow to import this file."
        )
    if len(_MIGRATIONS) > 0 and sorted(_MIGRATIONS)[-1] < current:
        raise ValueError(
            f"No migration exists to current version."
            f"Latest migration is to {_version_str(sorted(_MIGRATIONS)[-1])} but current verison is {_version_str(current)}"
        )
    for target in sorted(_MIGRATIONS):
        if found < target <= current:
            data = _MIGRATIONS[target](data)
    data["format_version"] = _version_dict(current)
    return data


# TODO - migrate to add date to each file as a 'created' attr

# NOTE - This migration process is necessary so that when (inevitably) some sort of attribute change takes place the serialize function doesn't shit the bed
#        Below is a demo patch as if
#   @_migration((0, 2, 0))
#   def _demo(data: Dict) -> Dict:
#       """v0.2.0 changed chip 'noise' values from a bare p float to {'p', 'scale'}."""
#       if data.get("__qsnow__") == "Chip":
#           data["noise"] = {k: {"p": p, "scale": 1.0} for k, p in data["noise"].items()}
#       return data


@_migration((0, 2, 0))
def _v0_1_to_v0_2(data: Dict) -> Dict:
    """v2 records the lattice a chip/tile is laid out on; everything before it was
    implicitly a checkerboard.

    Tiles are discriminated by 'base_circuit' rather than '__qsnow__' because tile
    dicts carry arbitrary subclass names there.
    """
    if data.get("__qsnow__") == "Chip" or "base_circuit" in data:
        data.setdefault("lattice", CHECKERBOARD.name)
    return data


@_migration((0, 3, 0))
def _v0_2_to_v0_3(data: Dict) -> Dict:
    """v3 records per-coupler error rates; before it, two-qubit noise was always the mean
    of a gate's two endpoint qubits.

    Left empty rather than derived here: the migration sees only the dict, and the import
    derives couplers from the restored qubit noise when this map is empty - which
    reproduces the pre-v3 rates exactly.
    """
    if data.get("__qsnow__") == "Chip":
        data.setdefault("couplers", {})
    return data


@_migration((0, 4, 0))
def _v0_3_to_v0_4(data: Dict) -> Dict:
    """v4 moves the noise-model record and coupler derivation mode out of
    `tag.metadata` into a typed `spec`, so code no longer branches on free-form
    annotation. Pre-v4 chips wrote `noise_model` and `coupler_model` into metadata;
    both are lifted out here and removed, so a migrated chip carries them once.
    """
    if data.get("__qsnow__") == "Chip":
        metadata = data.get("tag", {}).get("metadata", {})
        noise_model = metadata.pop("noise_model", None)
        coupler_model = metadata.pop("coupler_model", None) or {}
        data.setdefault(
            "spec",
            {
                "noise_model": noise_model,
                "coupler_mode": coupler_model.get("mode", "mean"),
            },
        )
    return data


@_migration((0, 5, 0))
def _v0_4_to_v0_5(data: Dict) -> Dict:
    """v5 adds `spec.coupler_model`, the coupler generator record. A v4 chip never had
    one, so the field is absent, which `ChipSpec` reads as "derived from endpoints"."""
    if data.get("__qsnow__") == "Chip":
        data.setdefault("spec", {}).setdefault("coupler_model", None)
    return data


@_migration((0, 6, 0))
def _v0_5_to_v0_6(data: Dict) -> Dict:
    """v6 records noise as `NoiseDistribution` objects. Two v5 records need renaming
    to match their classes: the coupler record "correlated contour" is `SkewContour`
    applied with a correlation, which now lives in `spec.coupler_correlation`; and
    "uniform random" spelled its bounds `range`, now `bounds`."""
    if data.get("__qsnow__") == "Chip":
        spec = data.setdefault("spec", {})
        spec.setdefault("coupler_correlation", None)
        coupler_model = spec.get("coupler_model")
        if coupler_model and coupler_model.get("name") == "correlated contour":
            coupler_model["name"] = "skewed contour"
            spec["coupler_correlation"] = coupler_model.pop("correlation", None)
        noise_model = spec.get("noise_model")
        if (
            noise_model
            and noise_model.get("name") == "uniform random"
            and "range" in noise_model
        ):
            noise_model["bounds"] = noise_model.pop("range")
    return data


@_migration((0, 7, 0))
def _v0_6_to_v0_7(data: Dict) -> Dict:
    """v7 renames the "derived contour" record to "normal contour", matching the
    `NormalContour` class and its `normal-contour` CLI flag.

    "Derived" already means "follows its endpoints" for couplers, so the old name read
    as a coupler mode rather than as the normal-marginal member of the contour family
    (`NormalContour`/`SkewContour`/`LogSkewContour`). Both the site and the coupler
    record can hold it.
    """
    if data.get("__qsnow__") == "Chip":
        spec = data.get("spec", {})
        for key in ("noise_model", "coupler_model"):
            record = spec.get(key)
            if record and record.get("name") == "derived contour":
                record["name"] = "normal contour"
    return data

@_migration((0, 8, 0))
def _v0_7_to_v0_8(data: Dict) -> Dict:
    """v0.8.0 renames `SquarePackingExp` to `ProfilingExp`."""
    if data.get("__qsnow__") == "SquarePackingExp":
        data["__qsnow__"] = "ProfilingExp"
    return data

@_migration((0, 9, 0))
def _v0_8_to_v0_9(data: Dict) -> Dict:
    """v0.9.0 introduces a new version formatting for flake files. Post v0.9, the format version
     follows the v<MAJOR>.<MINOR>.<PATCH> format.
     """
    version = data.get("format_version")
    if isinstance(version, int):
        data["format_version"] = _version_dict((0, version, 0))
    return data
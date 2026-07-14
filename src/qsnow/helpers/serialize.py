"""
JSON import/export for qSNOW objects.

Supports round-tripping `Chip`, `LogicalTile` (and code subclasses like `SCTile`),
`SquarePackingExp`, and the generic `Experiment` through plain JSON files, capturing
everything needed to rebuild the object with the exact same setup:

    from qsnow.helpers.serialize import export_json, import_json, import_latest

    export_json(chip)                      # -> data/chips/chip_5x5_<timestamp>.json
    export_json(tile)                      # -> data/tiles/tile_rsc_memory_z_d3_<timestamp>.json
    export_json(experiment, label='exp1')  # -> data/experiment/
    chip = import_latest("chip")           # newest matching export
    chip = import_json("data/chips/chip_5x5_2026-07-10_12-00-00.json")
    export_json(chip, "somewhere/else.json")  # explicit path still works

Notes:
  - Chip qubit statuses/types are not stored directly; they are re-derived by
    re-placing each tile via `chip.add_tile()`, which is how they were set
    originally.
  - Tile circuits are stored as Stim program text from `tile.base_circuit`
    (initial shift already baked in), so the spec's callables (`generator`,
    `initial_shift_fn`) are not serialized.
  - Every object's `Tag` (name/desc/metadata) round-trips; tiles additionally
    carry a `TileSpec` with the reconstruction fields code reads.
  - Ruleset injection rules are fully serialized. Custom triggers/filters
    (beyond the built-in defaults) hold arbitrary callables and cannot be
    serialized; a warning is raised if any are present at export.
  - Code subclasses (e.g. `SCTile`) are rebuilt through their own constructor
    using the spec's `generator_args`. Additional subclasses are registered with
    `register_tile_type()`.
  - Exports are stamped with `format_version`; older files are upgraded in
    memory on import. Any change to an export's structure must bump
    FORMAT_VERSION, register a matching `@_migration` step, and check in a new
    golden fixture under tests/helpers/fixtures/ (see the migration section).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from warnings import warn

from stim import Circuit

from qsnow.experiments.experiment import Experiment, ExperimentResults
from qsnow.experiments.squarepacking.game import SquarePackingExp
from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.models import Coord, Tag, TileSpec
from qsnow.interface.rules import (
    _DEFAULT_FILTERS,
    _DEFAULT_TRIGGERS,
    ChannelRule,
    InjectionRule,
    Ruleset,
)

FORMAT_VERSION = 1

# ------------------------------------------------------------------
# Format versioning / migrations
#
# Every exported dict is stamped with 'format_version'. When an export's
# structure changes, bump FORMAT_VERSION by one and register a @_migration(N)
# step that upgrades a dict from version N to N+1 (branching on the dict's
# '__qsnow__' kind if the change only affects one object type), and check in a
# new golden fixture under tests/helpers/fixtures/. Old files are upgraded in
# memory on import, one step at a time, and are never rewritten on disk.
# ------------------------------------------------------------------

_MIGRATIONS: Dict[int, Callable[[Dict], Dict]] = {}


def _migration(from_version: int):
    """Decorator registering an upgrade step from `from_version` to `from_version + 1`."""

    def register(fn: Callable[[Dict], Dict]) -> Callable[[Dict], Dict]:
        _MIGRATIONS[from_version] = fn
        return fn

    return register


# NOTE - This migration process is necessary so that when (inevitably) some sort of attribute change takes place the serialize function doesn't shit the bed
#        Below is a demo patch as if
#   @_migration(1)
#   def _v1_to_v2(data: Dict) -> Dict:
#       """v2 changed chip 'noise' values from a bare p float to {'p', 'scale'}."""
#       if data.get("__qsnow__") == "Chip":
#           data["noise"] = {k: {"p": p, "scale": 1.0} for k, p in data["noise"].items()}
#       return data


def _migrate(data: Dict) -> Dict:
    """Upgrade a persisted dict to the current FORMAT_VERSION before deserialization."""
    version = data.get("format_version", 1)  # earliest exports are v1
    if version > FORMAT_VERSION:
        raise ValueError(
            f"Export uses format v{version}, newer than this qSNOW install (v{FORMAT_VERSION}). "
            f"Upgrade qsnow to import this file."
        )
    while version < FORMAT_VERSION:
        data = _MIGRATIONS[version](data)
        version += 1
        data["format_version"] = version
    return data


def _find_repo_root() -> Path:
    """Locate the repository root so `data/` is stable regardless of the caller's cwd."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    # cwd is outside the repo; fall back to the source layout (src/qsnow/helpers -> root)
    return Path(__file__).resolve().parents[3]


# Default export root: `data/` at the repo root. Override with set_data_dir()
# for tests, notebooks, or an absolute location.
_DEFAULT_DATA_DIR = _find_repo_root() / "data"
_DATA_DIR = _DEFAULT_DATA_DIR

_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def set_data_dir(path: Optional[Union[str, Path]] = None) -> None:
    """
    Change the root folder used for automatic export paths.
    Call with no argument to restore the default (`data/` at the repo root).
    """
    global _DATA_DIR
    _DATA_DIR = Path(path) if path is not None else _DEFAULT_DATA_DIR


def get_data_dir() -> Path:
    """The current root folder used for automatic export paths."""
    return _DATA_DIR


def _subfolder(obj: Any) -> str:
    match obj:
        case Chip():
            return "chips"
        case LogicalTile():
            return "tiles"
        case SquarePackingExp() | Experiment():
            return "experiments"
        case _:
            raise TypeError(f"Cannot serialize object of type {type(obj).__name__}.")


def _prefix(obj: Any) -> str:
    """Object-kind prefix that starts every auto-generated filename."""
    match obj:
        case Chip():
            return "chip"
        case LogicalTile():
            return "tile"
        case SquarePackingExp() | Experiment():
            return "experiment"
        case _:
            raise TypeError(f"Cannot serialize object of type {type(obj).__name__}.")


def _label(obj: Any) -> str:
    """Default descriptor used when no explicit label is given."""
    match obj:
        case Chip():
            return f"{obj.length // 2}x{obj.height // 2}"
        case LogicalTile():
            return obj.tag.name or obj.spec.tile_type or type(obj).__name__.lower()
        case SquarePackingExp():
            return (
                f"squarepacking_{obj.tile.tag.name or type(obj.tile).__name__.lower()}"
            )
        case Experiment():
            return type(obj).__name__.lower()
        case _:
            raise TypeError(f"Cannot serialize object of type {type(obj).__name__}.")


# ------------------------------------------------------------------
# Coordinate helpers (JSON object keys must be strings)
# ------------------------------------------------------------------


def _coord_to_key(coord: Coord) -> str:
    return f"{coord[0]},{coord[1]}"


def _key_to_coord(key: str) -> Coord:
    parts = [float(p) for p in key.split(",")]
    return tuple(int(p) if p.is_integer() else p for p in parts)  # type: ignore


# ------------------------------------------------------------------
# Ruleset
# ------------------------------------------------------------------


def ruleset_to_dict(ruleset: Ruleset) -> Dict:
    default_triggers = {t.name for t in _DEFAULT_TRIGGERS}
    default_filters = {f.name for f in _DEFAULT_FILTERS}
    custom = (set(ruleset._triggers) - default_triggers) | (
        set(ruleset._filters) - default_filters
    )
    if custom:
        warn(
            f"Ruleset contains custom triggers/filters {sorted(custom)} which hold callables "
            f"and cannot be serialized. They must be re-registered manually after import.",
            stacklevel=2,
        )
    return {
        "rules": [
            {
                "operation": r.operation,
                "trigger": r.trigger,
                "before": [_channel_to_dict(c) for c in r.before],
                "after": [_channel_to_dict(c) for c in r.after],
                "exclusive": r.exclusive,
                "name": r.name,
            }
            for r in ruleset.rules
        ]
    }


def _channel_to_dict(channel: ChannelRule) -> Dict:
    return {
        "channel": channel.channel,
        "filter": channel.filter,
        "scalar": channel.scalar,
        "name": channel.name,
    }


def ruleset_from_dict(data: Dict) -> Ruleset:
    return Ruleset(
        [
            InjectionRule(
                operation=r["operation"],
                trigger=r["trigger"],
                before=[ChannelRule(**c) for c in r["before"]],
                after=[ChannelRule(**c) for c in r["after"]],
                exclusive=r["exclusive"],
                name=r["name"],
            )
            for r in data["rules"]
        ]
    )


# ------------------------------------------------------------------
# Tag / TileSpec
# ------------------------------------------------------------------


def tag_to_dict(tag: Tag) -> Dict:
    return {
        "name": tag.name,
        "desc": tag.desc,
        "metadata": tag.metadata,
    }


def tag_from_dict(data: Dict) -> Tag:
    return Tag(
        name=data["name"],
        desc=data["desc"],
        metadata=data["metadata"],
    )


def spec_to_dict(spec: TileSpec) -> Dict:
    # callables (generator, initial_shift_fn) are not serialized: the generator is
    # rebuilt by code-subclass constructors and the shift is baked into base_circuit
    return {
        "tile_type": spec.tile_type,
        "distance": spec.distance,
        "rounds": spec.rounds,
        "generator_args": spec.generator_args,
    }


def spec_from_dict(data: Dict) -> TileSpec:
    return TileSpec(
        tile_type=data["tile_type"],
        distance=data["distance"],
        rounds=data["rounds"],
        generator_args=data["generator_args"],
    )


# ------------------------------------------------------------------
# LogicalTile and code subclasses
# ------------------------------------------------------------------


def tile_to_dict(tile: LogicalTile) -> Dict:
    # A tile's grid dimensions are fixed at construction (circuit extent at the construction origin + buffers), while `add_tile`/`shift_by` move the tile
    # without resizing it. So a placed tile is exported against its base circuit and reconstructed fresh at (0, 0); `chip_from_dict` then
    # re-places it at 'origin' via `add_tile`, mirroring the original workflow. An unplaced tile keeps its constructor origin directly.
    placed = tile.initialized()
    ref_circuit = tile._base_circuit if placed else tile._circuit
    coords = list(ref_circuit.get_final_qubit_coordinates().values())
    dims = tuple(int(max(c) + 1) for c in zip(*coords))[:2]
    return {
        "__qsnow__": type(tile).__name__,
        "format_version": FORMAT_VERSION,
        "circuit": str(tile.base_circuit),
        "origin": list(tile.origin),
        "construct_origin": [0, 0] if placed else list(tile.origin),
        "x_buffer": tile.length - dims[0],
        "y_buffer": tile.height - dims[1],
        "tag": tag_to_dict(tile.tag),
        "spec": spec_to_dict(tile.spec),
        "ruleset": ruleset_to_dict(tile._ruleset),
    }


def _logical_tile_from_dict(data: Dict) -> LogicalTile:
    return LogicalTile(
        circuit=Circuit(data["circuit"]),
        origin=tuple(data["construct_origin"]),
        x_buffer=data["x_buffer"],
        y_buffer=data["y_buffer"],
        ruleset=ruleset_from_dict(data["ruleset"]),
        tag=tag_from_dict(data["tag"]),
        spec=spec_from_dict(data["spec"]),
    )


def _sc_tile_from_dict(data: Dict) -> SCTile:
    args = data["spec"]["generator_args"]
    task = args["task"]
    return SCTile(
        distance=args["distance"],
        rounds=args["rounds"],
        task=task,
        origin=tuple(data["construct_origin"]),
    )


_TILE_IMPORTERS: Dict[str, Callable[[Dict], LogicalTile]] = {
    "LogicalTile": _logical_tile_from_dict,
    "SCTile": _sc_tile_from_dict,
}


def register_tile_type(name: str, importer: Callable[[Dict], LogicalTile]) -> None:
    """Register an importer for a `LogicalTile` subclass so `from_dict` can rebuild it."""
    _TILE_IMPORTERS[name] = importer


def tile_from_dict(data: Dict) -> LogicalTile:
    data = _migrate(data)
    tile_type = data["__qsnow__"]
    importer = _TILE_IMPORTERS.get(tile_type)
    if importer is None:
        warn(
            f"Unknown tile type '{tile_type}'; importing as a generic LogicalTile. "
            f"Use register_tile_type() to preserve the subclass.",
            stacklevel=2,
        )
        importer = _logical_tile_from_dict
    tile = importer(data)
    # subclass importers rebuild through their constructor, which regenerates the
    # tag; restore the stored annotations so they survive the round trip (the spec
    # is owned by the constructor and matches the stored one by construction)
    tile.tag = tag_from_dict(data["tag"])
    return tile


# ------------------------------------------------------------------
# Chip
# ------------------------------------------------------------------


def chip_to_dict(chip: Chip) -> Dict:
    return {
        "__qsnow__": "Chip",
        "format_version": FORMAT_VERSION,
        "tag": tag_to_dict(chip.tag),
        # the original constructor arguments (grid is 2L x 2H internally)
        "length": chip.length // 2,
        "height": chip.height // 2,
        "noise": {_coord_to_key(c): q.noise.p for c, q in chip.grid.items()},
        "tiles": [tile_to_dict(t) for t in chip.tiles],
    }


def chip_from_dict(data: Dict) -> Chip:
    data = _migrate(data)
    chip = Chip(data["length"], data["height"])
    chip.tag = tag_from_dict(data["tag"])
    for key, p in data["noise"].items():
        chip.loc(_key_to_coord(key)).noise.p = p
    # Re-placing each tile rebuilds qubit statuses/types exactly as add_tile did originally.
    for tile_data in data["tiles"]:
        tile = tile_from_dict(tile_data)
        if not chip.add_tile(tile, tuple(tile_data["origin"])):
            raise ValueError(
                f"Failed to re-place tile at {tile_data['origin']} during chip import."
            )
    return chip


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------


def _experiment_headings(exp: Experiment, kind: str) -> Dict:
    """
    The envelope shared by every `Experiment` export: the config object, the
    freeform `desc` text, and an 'exp' subdict holding subclass-specific state.
    """
    return {
        "__qsnow__": kind,
        "format_version": FORMAT_VERSION,
        "tag": tag_to_dict(exp.tag),
        "config": exp.config,
        "results_refs": exp.results_refs,
        "exp": {},
    }


def square_packing_to_dict(exp: SquarePackingExp) -> Dict:
    data = _experiment_headings(exp, "SquarePackingExp")
    data["exp"] = {
        "chip": chip_to_dict(exp.chip),
        "tile": tile_to_dict(exp.tile),
        "bad": exp.bad,
        "profile": {
            _coord_to_key(loc): {
                "origin": list(p["origin"]),
                "bound": list(p["bound"]),
                "circuit": str(p["circuit"]) if p.get("circuit") is not None else None,
            }
            for loc, p in exp.profile.items()
        },
    }
    return data


def square_packing_from_dict(data: Dict) -> SquarePackingExp:
    data = _migrate(data)
    payload = data["exp"]
    exp = SquarePackingExp(
        chip=chip_from_dict(payload["chip"]),
        tile=tile_from_dict(payload["tile"]),
        bad=payload["bad"],
        profile={
            _key_to_coord(key): {
                "origin": tuple(p["origin"]),
                "bound": tuple(p["bound"]),
                "circuit": Circuit(p["circuit"]) if p["circuit"] is not None else None,
            }
            for key, p in payload["profile"].items()
        },
    )
    exp.tag = tag_from_dict(data["tag"])
    exp.config = data["config"]
    exp.results_refs = data["results_refs"]
    return exp


def experiment_to_dict(exp: Experiment) -> Dict:
    return _experiment_headings(exp, "Experiment")


def experiment_from_dict(data: Dict) -> Experiment:
    data = _migrate(data)
    exp = Experiment(**data["config"])
    exp.tag = tag_from_dict(data["tag"])
    exp.results_refs = data["results_refs"]
    return exp


def results_to_dict(exp: Experiment) -> Dict:
    """Serialize an experiment's `results` as a standalone record referencing its setup."""
    if exp.source is None:
        warn(
            "Experiment has no saved setup file, so these results will not reference "
            "a saved experiment; call exp.save() first to link them.",
            stacklevel=2,
        )
    return {
        "__qsnow__": "ExperimentResults",
        "format_version": FORMAT_VERSION,
        "experiment": exp.source.name if exp.source is not None else None,
        "desc": exp.desc,
        "run_config": exp.config,
        "results": {
            _coord_to_key(loc) if isinstance(loc, tuple) else str(loc): r
            for loc, r in exp.results.items()
        },
    }


def results_from_dict(data: Dict) -> ExperimentResults:
    data = _migrate(data)
    return ExperimentResults(
        experiment_ref=data["experiment"],
        run_config=data["run_config"],
        results={_key_to_coord(key): r for key, r in data["results"].items()},
        desc=data["desc"],
    )


def export_results(
    exp: Experiment,
    path: Optional[Union[str, Path]] = None,
    *,
    label: Optional[str] = None,
    indent: Optional[int] = 2,
) -> Path:
    """
    Write an experiment's `results` to their own JSON file, separate from the
    setup export. Auto-organized as `data/experiments/results_<label>_<ts>.json`.
    Prefer `exp.save_results()`, which also records the back-link on the experiment.
    """
    if path is None:
        stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        path = (
            _DATA_DIR / _subfolder(exp) / f"results_{label or _label(exp)}_{stamp}.json"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_to_dict(exp), f, indent=indent)
    return path


# ------------------------------------------------------------------
# Top-level dispatching
# ------------------------------------------------------------------


def to_dict(obj: Any) -> Dict:
    """Serialize a supported qSNOW object into a JSON-safe dict."""
    match obj:
        case Chip():
            return chip_to_dict(obj)
        case LogicalTile():
            return tile_to_dict(obj)
        case SquarePackingExp():
            return square_packing_to_dict(obj)
        case Experiment():
            return experiment_to_dict(obj)
        case _:
            raise TypeError(f"Cannot serialize object of type {type(obj).__name__}.")


def from_dict(data: Dict) -> Any:
    """Rebuild a qSNOW object from a dict produced by `to_dict`."""
    data = _migrate(data)
    kind = data.get("__qsnow__")
    match kind:
        case "Chip":
            return chip_from_dict(data)
        case "SquarePackingExp":
            return square_packing_from_dict(data)
        case "Experiment":
            return experiment_from_dict(data)
        case "ExperimentResults":
            return results_from_dict(data)
        case str() if kind in _TILE_IMPORTERS or kind is not None and "circuit" in data:
            return tile_from_dict(data)
        case _:
            raise ValueError(
                f"Unrecognized or missing '__qsnow__' type marker: {kind!r}."
            )


def export_json(
    obj: Any,
    path: Optional[Union[str, Path]] = None,
    *,
    label: Optional[str] = None,
    desc: Optional[str] = None,
    indent: Optional[int] = 2,
) -> Path:
    """
    Serialize `obj` and write it as JSON. Returns the written path.

    With no `path`, the file is auto-organized under the data root as
    `data/<kind>/<obj name>_<label>_<timestamp>.json`, where <kind> is
    chips/tiles/experiments, <obj name> is always the object kind (chip/tile/
    experiment), and <label> defaults to a descriptor derived from the object
    (e.g. `chip_5x5_...`, `tile_rsc_memory_z_d3_...`). Pass `label` to override
    the descriptor, or `path` for full control.

    `desc` sets a freeform description on the object's tag before writing, so
    it is stored in the export and survives reimport.
    """
    if desc is not None:
        # every serializable kind carries a Tag; desc is object state so it round-trips
        # TODO - should this overwrite or append desc. circle back once solidified v1 exporter and see what works best
        obj.tag.desc = desc
    if path is None:
        stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        path = (
            _DATA_DIR
            / _subfolder(obj)
            / f"{_prefix(obj)}_{label or _label(obj)}_{stamp}.json"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_dict(obj), f, indent=indent)
    return path


def import_json(path: Union[str, Path]) -> Any:
    """Load a JSON file written by `export_json` and rebuild the object."""
    with open(path) as f:
        obj = from_dict(json.load(f))
    if isinstance(obj, Experiment):
        obj.source = Path(path)
    return obj


def list_exports(pattern: str = "*", kind: Optional[str] = None) -> List[Path]:
    """
    List exported JSON files under the data root, newest first.

    `pattern` glob-matches from the start of the filename (e.g. "chip*", "tile_rsc*d3").
    If nothing matches, it is retried as a substring match (`*<pattern>*`) so
    label-only searches like "d3-20x20chip" find `experiment_d3-20x20chip_...`.
    `kind` restricts the search to one subfolder ('chips', 'tiles', 'experiments').
    """
    root = _DATA_DIR / kind if kind else _DATA_DIR

    def _glob(pat: str) -> List[Path]:
        name = f"{pat}.json" if pat.endswith("*") else f"{pat}*.json"
        return list(root.glob(f"**/{name}" if kind is None else name))

    matches = _glob(pattern)
    if not matches and not pattern.startswith("*"):
        matches = _glob(f"*{pattern}")
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


def summarize_exports(pattern: str = "*", kind: Optional[str] = None) -> Dict[Path, Optional[str]]:
    """
    Map each export matching `pattern`/`kind` (newest first) to its tag `desc`,
    making the data folder browsable without opening files.
    """
    summary: Dict[Path, Optional[str]] = {}
    for path in list_exports(pattern, kind):
        with open(path) as f:
            data = json.load(f)
        summary[path] = data.get("tag", {}).get("desc") or data.get("desc")
    return summary


def import_latest(pattern: str = "*", kind: Optional[str] = None) -> Any:
    """
    Import the most recent export whose filename matches `pattern`.

    Examples: `import_latest("chip")`, `import_latest("rsc*d3", kind="tiles")`.
    """
    matches = list_exports(pattern, kind)
    if not matches:
        raise FileNotFoundError(
            f"No snowflakes matching '{pattern}'{f' in {kind}/' if kind else ''} under {_DATA_DIR}/."
        )
    if matches:
        print(
            f"Found {len(matches)} matching snowflakes...\nImporting flake at {matches[0]}"
        )
    return import_json(matches[0])

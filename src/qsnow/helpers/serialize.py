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
    (initial shift already baked in), so `initial_shift_fn` is not needed on
    reimport and is dropped from the tag.
  - Ruleset injection rules are fully serialized. Custom triggers/filters
    (beyond the built-in defaults) hold arbitrary callables and cannot be
    serialized; a warning is raised if any are present at export.
  - Code subclasses (e.g. `SCTile`) are rebuilt through their own constructor
    using the tag's `generator_args`. Additional subclasseses registered with
    `register_tile_type()`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

from stim import Circuit

from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.models import Coord, TileTag
from qsnow.interface.rules import (
    ChannelRule,
    InjectionRule,
    Ruleset,
    _DEFAULT_FILTERS,
    _DEFAULT_TRIGGERS,
)
from qsnow.experiments.experiment import Experiment
from qsnow.experiments.squarepacking.game import SquarePackingExp

FORMAT_VERSION = 1

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

def _subfolder(obj: Any) -> str:
    match obj:
        case Chip():
            return 'chips'
        case LogicalTile():
            return 'tiles'
        case SquarePackingExp() | Experiment():
            return 'experiments'
        case _:
            raise TypeError(f"Cannot serialize object of type {type(obj).__name__}.")

def _prefix(obj: Any) -> str:
    """Object-kind prefix that starts every auto-generated filename."""
    match obj:
        case Chip():
            return 'chip'
        case LogicalTile():
            return 'tile'
        case SquarePackingExp() | Experiment():
            return 'experiment'
        case _:
            raise TypeError(f"Cannot serialize object of type {type(obj).__name__}.")

def _label(obj: Any) -> str:
    """Default descriptor used when no explicit label is given."""
    match obj:
        case Chip():
            return f"{obj.length // 2}x{obj.height // 2}"
        case LogicalTile():
            return obj.tag.name or type(obj).__name__.lower()
        case SquarePackingExp():
            return f"squarepacking_{obj.tile.tag.name or type(obj.tile).__name__.lower()}"
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
    custom = (set(ruleset._triggers) - default_triggers) | (set(ruleset._filters) - default_filters)
    if custom:
        warn(
            f"Ruleset contains custom triggers/filters {sorted(custom)} which hold callables "
            f"and cannot be serialized. They must be re-registered manually after import.",
            stacklevel=2,
        )
    return {
        'rules': [
            {
                'operation': r.operation,
                'trigger': r.trigger,
                'before': [_channel_to_dict(c) for c in r.before],
                'after': [_channel_to_dict(c) for c in r.after],
                'exclusive': r.exclusive,
                'name': r.name,
            }
            for r in ruleset.rules
        ]
    }

def _channel_to_dict(channel: ChannelRule) -> Dict:
    return {
        'channel': channel.channel,
        'filter': channel.filter,
        'scalar': channel.scalar,
        'name': channel.name,
    }

def ruleset_from_dict(data: Dict) -> Ruleset:
    return Ruleset([
        InjectionRule(
            operation=r['operation'],
            trigger=r['trigger'],
            before=[ChannelRule(**c) for c in r['before']],
            after=[ChannelRule(**c) for c in r['after']],
            exclusive=r['exclusive'],
            name=r['name'],
        )
        for r in data['rules']
    ])

# ------------------------------------------------------------------
# TileTag
# ------------------------------------------------------------------

def tag_to_dict(tag: TileTag) -> Dict:
    return {
        'name': tag.name,
        'tile_type': tag.tile_type,
        'distance': tag.distance,
        'rounds': tag.rounds,
        'generator_args': tag.generator_args,
        'metadata': tag.metadata,
    }

def tag_from_dict(data: Dict) -> TileTag:
    return TileTag(
        name=data['name'],
        tile_type=data['tile_type'],
        distance=data['distance'],
        rounds=data['rounds'],
        generator_args=data['generator_args'],
        metadata=data['metadata'],
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
        '__qsnow__': type(tile).__name__,
        'format_version': FORMAT_VERSION,
        'circuit': str(tile.base_circuit),
        'origin': list(tile.origin),
        'construct_origin': [0, 0] if placed else list(tile.origin),
        'x_buffer': tile.length - dims[0],
        'y_buffer': tile.height - dims[1],
        'tag': tag_to_dict(tile.tag),
        'ruleset': ruleset_to_dict(tile._ruleset),
    }

def _logical_tile_from_dict(data: Dict) -> LogicalTile:
    return LogicalTile(
        circuit=Circuit(data['circuit']),
        origin=tuple(data['construct_origin']),
        x_buffer=data['x_buffer'],
        y_buffer=data['y_buffer'],
        ruleset=ruleset_from_dict(data['ruleset']),
        tag=tag_from_dict(data['tag']),
    )

def _sc_tile_from_dict(data: Dict) -> SCTile:
    args = data['tag']['generator_args']
    # code_task is e.g. "surface_code:rotated_memory_z" -> task "memory_z"
    task = args['code_task'].split('rotated_', 1)[1]
    return SCTile(
        distance=args['distance'],
        rounds=args['rounds'],
        task=task,
        origin=tuple(data['construct_origin']),
    )

_TILE_IMPORTERS: Dict[str, Callable[[Dict], LogicalTile]] = {
    'LogicalTile': _logical_tile_from_dict,
    'SCTile': _sc_tile_from_dict,
}

def register_tile_type(name: str, importer: Callable[[Dict], LogicalTile]) -> None:
    """Register an importer for a `LogicalTile` subclass so `from_dict` can rebuild it."""
    _TILE_IMPORTERS[name] = importer

def tile_from_dict(data: Dict) -> LogicalTile:
    tile_type = data['__qsnow__']
    importer = _TILE_IMPORTERS.get(tile_type)
    if importer is None:
        warn(
            f"Unknown tile type '{tile_type}'; importing as a generic LogicalTile. "
            f"Use register_tile_type() to preserve the subclass.",
            stacklevel=2,
        )
        importer = _logical_tile_from_dict
    return importer(data)

# ------------------------------------------------------------------
# Chip
# ------------------------------------------------------------------

def chip_to_dict(chip: Chip) -> Dict:
    return {
        '__qsnow__': 'Chip',
        'format_version': FORMAT_VERSION,
        # the original constructor arguments (grid is 2L x 2H internally)
        'length': chip.length // 2,
        'height': chip.height // 2,
        'noise': {_coord_to_key(c): q.noise.p for c, q in chip.grid.items()},
        'tiles': [tile_to_dict(t) for t in chip.tiles],
    }

def chip_from_dict(data: Dict) -> Chip:
    chip = Chip(data['length'], data['height'])
    for key, p in data['noise'].items():
        chip.loc(_key_to_coord(key)).noise.p = p
    # Re-placing each tile rebuilds qubit statuses/types exactly as add_tile did originally.
    for tile_data in data['tiles']:
        tile = tile_from_dict(tile_data)
        if not chip.add_tile(tile, tuple(tile_data['origin'])):
            raise ValueError(f"Failed to re-place tile at {tile_data['origin']} during chip import.")
    return chip

# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------

def square_packing_to_dict(exp: SquarePackingExp) -> Dict:
    return {
        '__qsnow__': 'SquarePackingExp',
        'format_version': FORMAT_VERSION,
        'chip': chip_to_dict(exp.chip),
        'tile': tile_to_dict(exp.tile),
        'bad': exp.bad,
        'profile': {
            _coord_to_key(loc): {
                'origin': list(p['origin']),
                'bound': list(p['bound']),
                'circuit': str(p['circuit']) if p.get('circuit') is not None else None,
                'ler': p.get('ler', 'N/A'),
                'time': p.get('time'),
            }
            for loc, p in exp.profile.items()
        },
    }

def square_packing_from_dict(data: Dict) -> SquarePackingExp:
    return SquarePackingExp(
        chip=chip_from_dict(data['chip']),
        tile=tile_from_dict(data['tile']),
        bad=data['bad'],
        profile={
            _key_to_coord(key): {
                'origin': tuple(p['origin']),
                'bound': tuple(p['bound']),
                'circuit': Circuit(p['circuit']) if p['circuit'] is not None else None,
                'ler': p['ler'],
                'time': p['time'],
            }
            for key, p in data['profile'].items()
        },
    )

def experiment_to_dict(exp: Experiment) -> Dict:
    return {
        '__qsnow__': 'Experiment',
        'format_version': FORMAT_VERSION,
        'config': exp.config,
    }

def experiment_from_dict(data: Dict) -> Experiment:
    return Experiment(**data['config'])

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
    kind = data.get('__qsnow__')
    match kind:
        case 'Chip':
            return chip_from_dict(data)
        case 'SquarePackingExp':
            return square_packing_from_dict(data)
        case 'Experiment':
            return experiment_from_dict(data)
        case str() if kind in _TILE_IMPORTERS or kind is not None and 'circuit' in data:
            return tile_from_dict(data)
        case _:
            raise ValueError(f"Unrecognized or missing '__qsnow__' type marker: {kind!r}.")

def export_json(obj: Any,
                path: Optional[Union[str, Path]] = None,
                *,
                label: Optional[str] = None,
                indent: Optional[int] = 2) -> Path:
    """
    Serialize `obj` and write it as JSON. Returns the written path.

    With no `path`, the file is auto-organized under the data root as
    `data/<kind>/<obj name>_<label>_<timestamp>.json`, where <kind> is
    chips/tiles/experiments, <obj name> is always the object kind (chip/tile/
    experiment), and <label> defaults to a descriptor derived from the object
    (e.g. `chip_5x5_...`, `tile_rsc_memory_z_d3_...`). Pass `label` to override
    the descriptor, or `path` for full control.
    """
    if path is None:
        stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        path = _DATA_DIR / _subfolder(obj) / f"{_prefix(obj)}_{label or _label(obj)}_{stamp}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(to_dict(obj), f, indent=indent)
    return path

def import_json(path: Union[str, Path]) -> Any:
    """Load a JSON file written by `export_json` and rebuild the object."""
    with open(path) as f:
        return from_dict(json.load(f))

def list_exports(pattern: str = "*", kind: Optional[str] = None) -> List[Path]:
    """
    List exported JSON files under the data root, newest first.

    `pattern` glob-matches the label portion of filenames (e.g. "chip*", "rsc*d3").
    `kind` restricts the search to one subfolder ('chips', 'tiles', 'experiments').
    """
    root = _DATA_DIR / kind if kind else _DATA_DIR
    name = f"{pattern}.json" if pattern.endswith("*") else f"{pattern}*.json"
    matches = root.glob(f"**/{name}" if kind is None else name)
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

def import_latest(pattern: str = "*", kind: Optional[str] = None) -> Any:
    """
    Import the most recent export whose filename matches `pattern`.

    Examples: `import_latest("chip")`, `import_latest("rsc*d3", kind="tiles")`.
    """
    matches = list_exports(pattern, kind)
    if not matches:
        raise FileNotFoundError(
            f"No exports matching '{pattern}'{f' in {kind}/' if kind else ''} under {_DATA_DIR}/."
        )
    return import_json(matches[0])

import argparse
import math
import sys
import textwrap
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

pd.set_option("mode.string_storage", "python")

DEFAULT_CALIBRATION_DIR = Path("./calibration_data")   # --calibration-dir
DEFAULT_PLOT_DIR = Path("./plots")
NCOLS = 3           # panels per row; --cols overrides
PANEL_W, PANEL_H = 4.2, 3.0   # inches per panel when --size is not given
NBINS = 30          # per panel; --bins overrides
DEFAULT_DPI = 170   # matches noisegen/viz/lattice_map.py

# Dead-component thresholds. Same values as noisegen/ingest.py; change both.
T1_MIN_US = 20.0    # below this a qubit is dead, not merely bad
RO_MAX = 0.30       # readout assignment error above this: dead
CZ_MAX = 0.05       # CZ error above this is a failed calibration, dropped per edge

# Qubit-table quantities, in display order. `readout_length` is omitted: it is
# a per-device constant and a histogram of it is a single bar.
QUBIT_PANELS: tuple[tuple[str, str, bool], ...] = (
    # column,            title,               log-binned
    ("T1",               "T1 (us)",           False),
    ("T2",               "T2 (us)",           False),
    ("readout_error",    "readout_error",     True),
    ("init_error",       "init_error",        True),
    ("prob_meas0_prep1", "prob_meas0_prep1",  True),
    ("prob_meas1_prep0", "prob_meas1_prep0",  True),
)


@dataclass(frozen=True)
class DeadCount:
    qubits: int = 0        # dead qubits (their gate rows go too)
    gate_rows: int = 0     # gate rows removed because an endpoint was dead
    cz_failed: int = 0     # CZ edges over CZ_MAX, both endpoints alive

    def __add__(self, other: "DeadCount") -> "DeadCount":
        return DeadCount(self.qubits + other.qubits, self.gate_rows + other.gate_rows,
                         self.cz_failed + other.cz_failed)

    @property
    def any(self) -> bool:
        return bool(self.qubits or self.gate_rows or self.cz_failed)

    def describe(self) -> str:
        return (f"{self.qubits} dead qubit{'s' if self.qubits != 1 else ''} "
                f"(T1 < {T1_MIN_US:g} us, readout > {RO_MAX:g}, or not operational; "
                f"{self.gate_rows} gate rows with them), "
                f"{self.cz_failed} failed CZ edge{'s' if self.cz_failed != 1 else ''} "
                f"(> {CZ_MAX:g})")


def dead_qubits(df: pd.DataFrame) -> set[str]:
    q = df[df["table"] == "qubit"]
    t1 = pd.to_numeric(q["T1"], errors="coerce") if "T1" in q else pd.Series(np.nan, index=q.index)
    ro = pd.to_numeric(q["readout_error"], errors="coerce") if "readout_error" in q else pd.Series(np.nan, index=q.index)
    dead = (t1 < T1_MIN_US) | (ro > RO_MAX)
    if "operational" in q:
        op = q["operational"].astype(str).str.strip().str.lower()
        dead |= op.isin(("no", "false", "0"))
    ids = pd.to_numeric(q.loc[dead, "qubit"], errors="coerce").dropna().astype(int).astype(str)
    return set(ids)


def drop_dead(df: pd.DataFrame) -> tuple[pd.DataFrame, DeadCount]:
    dead = dead_qubits(df)
    is_q = df["table"] == "qubit"
    is_g = df["table"] == "gate"
    qid = pd.to_numeric(df["qubit"], errors="coerce")
    q_dead = is_q & qid.isin({int(d) for d in dead})
    ends = df["qubits"].astype(str).str.split(",")
    g_dead = is_g & ends.apply(lambda e: any(x.strip() in dead for x in e) if isinstance(e, list) else False)
    err = pd.to_numeric(df["gate_error"], errors="coerce")
    cz_failed = is_g & ~g_dead & (df["gate"].astype(str) == "cz") & (err > CZ_MAX)
    kept = df[~(q_dead | g_dead | cz_failed)]
    # Both directions of an edge carry the same error, so they always fail
    # together; drop both but report the edge once.
    n_cz = len(undirected_rows(df.loc[cz_failed])) if cz_failed.any() else 0
    return kept, DeadCount(int(q_dead.sum()), int(g_dead.sum()), n_cz)


@dataclass(frozen=True)
class Series:
    """One legend entry: a single snapshot, or several averaged."""
    label: str
    frames: list[pd.DataFrame]
    dead: DeadCount = DeadCount()
    paths: tuple[Path, ...] = ()          # the analysis files, when it came from disk

    @property
    def weight(self) -> float:
        """Per-value histogram weight, so N pooled snapshots read as one average."""
        return 1.0 / len(self.frames)

    def values(self, panel: "Panel") -> np.ndarray:
        v = np.concatenate([panel.values(f) for f in self.frames]) if self.frames else np.array([])
        return v[np.isfinite(v)]


@dataclass(frozen=True)
class Panel:
    title: str
    log_scale: bool
    unit_label: str                       # y-axis: what is being counted
    values: Callable[[pd.DataFrame], np.ndarray]
    error_rate: bool = True               # only positive values carry information
    names: tuple[str, ...] = ()           # what `--panels` may call it: the column, or every gate in the group


# ---- calibration tree --------------------------------------------------------
#
# Copies of the layout rules in ibmcalib/paths.py, carried here so this file
# runs standalone. tests/test_calib_histograms.py pins each one to the original.

API_SUBDIR = "api"
CONSOLE_SUBDIR = "console"
SOURCES = (API_SUBDIR, CONSOLE_SUBDIR)
LONG_SUFFIX = ".long.csv"


def long_form_path(path: Path, source: str) -> Path:
    path = Path(path)
    if source == CONSOLE_SUBDIR and not path.name.endswith(LONG_SUFFIX):
        return path.with_name(path.name[: -len(".csv")] + LONG_SUFFIX)
    return path


def wide_form_path(path: Path) -> Path | None:
    path = Path(path)
    if not path.name.endswith(LONG_SUFFIX):
        return None
    wide = path.with_name(path.name[: -len(LONG_SUFFIX)] + ".csv")
    return wide if wide.exists() else None


# Per-qubit partner lists in the wide export, `7:84;16:84;...`. The gate-length
# column is listed first because IBM fills it in even for a coupler whose error
# is blank, which is exactly the coupler the long form loses.
PARTNER_COLUMNS = ("Gate length (ns)", "CZ error")


def wide_roster(path: Path) -> tuple[set[str], set[tuple[int, int]]]:
    df = pd.read_csv(path)
    if "Qubit" not in df.columns:
        return set(), set()
    qubits = set(df["Qubit"].map(component_key))
    edges: set[tuple[int, int]] = set()
    for col in PARTNER_COLUMNS:
        if col not in df.columns:
            continue
        for q, cell in zip(df["Qubit"], df[col]):
            if not isinstance(cell, str):
                continue
            for part in cell.split(";"):
                head = part.split(":")[0].strip()
                try:
                    a, b = int(q), int(head)
                except (TypeError, ValueError):
                    continue
                edges.add((a, b) if a <= b else (b, a))
    return qubits, edges


def calibration_stamp(path: Path, device: str) -> str | None:
    """IBM's stamp from a saved filename, either source's format.
    Copy of `paths.calibration_stamp`."""
    name = Path(path).name
    if name.endswith(LONG_SUFFIX):
        name = name[: -len(LONG_SUFFIX)]
    elif name.endswith(".csv"):
        name = name[: -len(".csv")]
    prefix = f"{device}_"
    return name[len(prefix):] if name.startswith(prefix) else None


class Pull(NamedTuple):
    """One saved calibration: where it is and which IBM stamp it carries."""
    device: str
    source: str
    family: str
    path: Path
    calibration: str | None


def recent_pulls(directory: Path, device: str, source: str, family: str,
                 n: int | None = None) -> list[Pull]:
    files = [p for p in Path(directory).glob(f"{device}_*.csv") if not p.name.endswith(LONG_SUFFIX)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [Pull(device, source, family, f, calibration_stamp(f, device)) for f in files[:n]]


def find_device(output_dir: Path, name: str) -> list[Pull]:
    found: list[Pull] = []
    for source in SOURCES:
        source_dir = Path(output_dir) / source
        if not source_dir.is_dir():
            continue
        for family_dir in sorted(source_dir.iterdir()):
            dev_dir = family_dir / name
            if family_dir.name.startswith(".") or not dev_dir.is_dir():
                continue
            found += recent_pulls(dev_dir, name, source, family_dir.name, n=1)
    return sorted(found, key=lambda p: p.path.stat().st_mtime, reverse=True)


# ---- inputs -----------------------------------------------------------------

def series_label(path: Path, device: str | None = None) -> str:
    path = Path(path)
    if device is None:
        device = path.parent.name if path.parent.name else path.stem
    stamp = calibration_stamp(path, device)
    if stamp is None:
        return path.name
    return f"{device} {stamp}"


def _stamp_date(stamp: str | None) -> str:
    return (stamp or "?")[:10]


def range_label(device: str, stamps: list[str | None]) -> str:
    dates = sorted(_stamp_date(st) for st in stamps)
    n = len(stamps)
    span = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"
    return f"{device}: {n} snapshot{'s' if n != 1 else ''}, {span}"


def resolve_devices(names: list[str], output_dir: Path,
                    snapshots: int | None = 1, keep_dead: bool = False) -> list[Series]:
    out: list[Series] = []
    for name in names:
        matches = find_device(output_dir, name)
        if not matches:
            raise FileNotFoundError(f"no calibration data for {name!r} under {output_dir}")
        pick = matches[0]  # find_device() sorts newest pull first
        if len(matches) > 1:
            print(f"{name}: pulled via {', '.join(m.source for m in matches)}; "
                  f"using the newest ({pick.source})", file=sys.stderr)
        recent = recent_pulls(pick.path.parent, pick.device, pick.source, pick.family,
                              n=snapshots)
        if snapshots is not None and len(recent) < snapshots:
            print(f"{name}: {snapshots} snapshots requested, {len(recent)} available; "
                  f"using {len(recent)}", file=sys.stderr)
        files = [long_form_path(r.path, r.source) for r in recent]
        if len(recent) == 1:
            label = series_label(files[0], pick.device)
        else:
            label = range_label(pick.device, [r.calibration for r in recent])
        out.append(load_series(label, files, keep_dead))
    return out


def series_from_frames(label: str, frames: list[pd.DataFrame], keep_dead: bool = False) -> Series:
    kept, dead = [], DeadCount()
    for df in frames:
        if not keep_dead:
            df, d = drop_dead(df)
            dead = dead + d
        kept.append(df)
    return Series(label, kept, dead)


def load_series(label: str, files: list[Path], keep_dead: bool = False) -> Series:
    s = series_from_frames(label, [load_long(f) for f in files], keep_dead)
    return replace(s, paths=tuple(Path(f) for f in files))


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "table" not in df.columns:
        sibling = long_form_path(path, CONSOLE_SUBDIR)
        raise ValueError(
            f"{path} is IBM's wide console export, not the long schema.\n"
            f"Pass its normalized sibling instead: {sibling}"
        )
    return df


# ---- panels -----------------------------------------------------------------

def _qubit_values(col: str) -> Callable[[pd.DataFrame], np.ndarray]:
    def f(df: pd.DataFrame) -> np.ndarray:
        q = df[df["table"] == "qubit"]
        return pd.to_numeric(q[col], errors="coerce").dropna().to_numpy() if col in q else np.array([])
    return f


def undirected_rows(g: pd.DataFrame) -> pd.DataFrame:
    return g.loc[~g["qubits"].map(component_key).duplicated()]


def _end_key(end) -> tuple:
    """Sort qubit ids numerically when they are numeric, else lexically, so
    `10,9` and `9,10` canonicalize to the same key either way."""
    try:
        return (0, int(float(str(end).strip())), "")
    except ValueError:
        return (1, 0, str(end).strip())


def component_key(cell) -> str:
    parts = sorted((x.strip() for x in str(cell).split(",")), key=_end_key)
    out = []
    for x in parts:
        try:
            out.append(str(int(float(x))))
        except ValueError:
            out.append(x)
    return ",".join(out)


def _gate_values(gate: str) -> Callable[[pd.DataFrame], np.ndarray]:
    def f(df: pd.DataFrame) -> np.ndarray:
        g = df[(df["table"] == "gate") & (df["gate"] == gate)]
        g = undirected_rows(g)
        return pd.to_numeric(g["gate_error"], errors="coerce").dropna().to_numpy()
    return f


def gate_order(names) -> list[str]:
    names = set(names)
    rest = sorted(n for n in names if n != "cz")
    return (["cz"] if "cz" in names else []) + rest


def _gate_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    g = df[df["table"] == "gate"].copy()
    g["gate_error"] = pd.to_numeric(g["gate_error"], errors="coerce")
    g = g.dropna(subset=["gate", "gate_error"])
    return {name: sub.set_index(sub["qubits"].astype(str))["gate_error"]
            for name, sub in g.groupby(g["gate"].astype(str))}


def _coincident(a: str, b: str, per_frame: list[dict[str, pd.Series]]) -> bool:
    seen = False
    for series in per_frame:
        sa, sb = series.get(a), series.get(b)
        if sa is None and sb is None:
            continue
        if sa is None or sb is None or len(sa) != len(sb):
            return False
        if not sa.index.equals(sb.index):
            sb = sb.reindex(sa.index)
        if not np.array_equal(sa.to_numpy(), sb.to_numpy()):
            return False
        seen = True
    return seen


def gate_groups(frames: list[pd.DataFrame]) -> list[list[str]]:
    per_frame = [_gate_series(df) for df in frames]
    # Names come from the raw column, not the series: a gate whose errors are
    # all NaN (reset over the API) still gets a panel, so the drop note names it.
    names = gate_order({str(n) for df in frames
                        for n in df[df["table"] == "gate"]["gate"].dropna().unique()})
    groups: list[list[str]] = []
    for name in names:
        for grp in groups:
            if _coincident(grp[0], name, per_frame):
                grp.append(name)
                break
        else:
            groups.append([name])
    # Same rule as gate_order, lifted to groups: the cz group first, then by
    # the alphabetically-first member.
    return sorted((sorted(g) for g in groups), key=lambda g: ("cz" not in g, g[0]))


def gate_panel_title(members: list[str]) -> str:
    if len(members) == 1:
        return f"{members[0]} gate_error"
    return f"({', '.join(members)}) gate_error"


def panel_specs(frames: list[pd.DataFrame], linear: bool = False) -> tuple[list[Panel], list[str]]:
    panels: list[Panel] = []
    for col, title, log_scale in QUBIT_PANELS:
        panels.append(Panel(title, log_scale and not linear, "qubits", _qubit_values(col),
                            error_rate=log_scale, names=(col,)))

    # Each gate is counted once per component. Coincident gates (id/rx/sx/x on
    # IBM devices) share one panel and the extractor reads the group's first
    # member only; a two-qubit gate, listed from both endpoints, is collapsed
    # to one row per undirected pair by `undirected_rows`.
    for members in gate_groups(frames):
        panels.append(Panel(gate_panel_title(members), not linear, "gates",
                            _gate_values(members[0]), names=tuple(members)))

    kept, dropped = [], []
    for p in panels:
        pooled = np.concatenate([p.values(df) for df in frames]) if frames else np.array([])
        pooled = pooled[np.isfinite(pooled)]
        # The drop rule ignores `linear` on purpose: an all-zero column is
        # empty information either way.
        has_data = (pooled > 0).any() if p.error_rate else pooled.size > 0
        (kept if has_data else dropped).append(p if has_data else p.title)
    return kept, dropped


PANEL_KEYWORDS = {"qubits": "qubits", "gates": "gates"}   # --panels word -> Panel.unit_label


def panel_names(text: str) -> list[str]:
    names = [t.strip() for t in text.split(",")]
    if not any(names):
        raise argparse.ArgumentTypeError("expected one or more panel names")
    return [n for n in names if n]


def select_panels(panels: list[Panel], wanted: list[str],
                  dropped: list[str] = ()) -> list[Panel]:
    import re
    chosen: list[Panel] = []
    for name in wanted:
        if name in PANEL_KEYWORDS:
            hits = [p for p in panels if p.unit_label == PANEL_KEYWORDS[name]]
        else:
            hits = [p for p in panels if name in p.names or name == p.title]
        if not hits:
            if any(re.search(rf"\b{re.escape(name)}\b", t) for t in dropped):
                raise ValueError(f"panel {name!r} has no data in these inputs and was dropped")
            available = [n for p in panels for n in p.names] + list(PANEL_KEYWORDS)
            raise ValueError(f"unknown panel {name!r}; choose from: {', '.join(available)}")
        chosen += [p for p in hits if p not in chosen]
    return chosen


def bin_edges(pooled: np.ndarray, log_scale: bool, nbins: int = NBINS) -> np.ndarray:
    if log_scale:
        pooled = pooled[pooled > 0]
        lo, hi = np.log10(pooled.min()), np.log10(pooled.max())
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        return np.logspace(lo, hi, nbins + 1)
    if pooled.max() <= pooled.min():
        return np.linspace(pooled.min() - 0.5, pooled.max() + 0.5, nbins + 1)
    return np.histogram_bin_edges(pooled, bins=nbins)


# ---- summary statistics -------------------------------------------------------

STAT_NAMES = ("mean", "median", "std", "skew")


def pearson3_skew(v: np.ndarray) -> float:
    x = np.asarray(v, dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    s = x.std(ddof=1)
    if s == 0:
        return float("nan")
    return float(n / ((n - 1) * (n - 2)) * np.sum((x - x.mean()) ** 3) / s ** 3)


def summary_stats(v: np.ndarray) -> dict[str, float]:
    x = np.asarray(v, dtype=float)
    return {"mean": float(x.mean()) if x.size else float("nan"),
            "median": float(np.median(x)) if x.size else float("nan"),
            "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
            "skew": pearson3_skew(x)}


def log_summary_stats(v: np.ndarray) -> dict[str, float]:
    x = np.asarray(v, dtype=float)
    x = x[x > 0]
    lx = np.log10(x)
    return {"mean": float(x.mean()) if x.size else float("nan"),
            "median": float(np.median(x)) if x.size else float("nan"),
            "log sd": float(lx.std(ddof=1)) if x.size > 1 else float("nan"),
            "log skew": pearson3_skew(lx)}


def tukey_fence(lx: np.ndarray, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = np.quantile(lx, [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def tukey_keep(lx: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Mask of values inside `tukey_fence`."""
    lo, hi = tukey_fence(lx, k)
    return (lx >= lo) & (lx <= hi)


def bulk_log_stats(v: np.ndarray) -> dict[str, float]:
    x = np.asarray(v, dtype=float)
    x = x[x > 0]
    if x.size < 2:
        return {"mean": float("nan"), "median": float("nan"), "log sd": float("nan"), "log skew": float("nan"),
                "kept": float("nan")}
    keep = tukey_keep(np.log10(x))
    st = log_summary_stats(x[keep])
    return {"mean": st['mean'], "median": st["median"], "log sd": st["log sd"], "log skew": st["log skew"],
            "kept": float(keep.mean())}


def _panel_rows(df: pd.DataFrame, panel: "Panel") -> tuple[pd.Series, pd.Series]:
    if panel.unit_label == "qubits":
        q = df[df["table"] == "qubit"]
        col = panel.names[0]
        vals = pd.to_numeric(q[col], errors="coerce") if col in q else pd.Series(np.nan, index=q.index)
        return q["qubit"].map(component_key), vals
    g = undirected_rows(df[(df["table"] == "gate") & (df["gate"] == panel.names[0])])
    return g["qubits"].map(component_key), pd.to_numeric(g["gate_error"], errors="coerce")


def panel_components(frames: list[pd.DataFrame], panel: "Panel",
                     qubits: set[str] | None = None,
                     couplers: set[tuple[int, int]] | None = None) -> tuple[int, int]:
    roster: set[str] = set()
    live: set[str] = set()
    qubit_ids: set[str] = set(qubits or ())
    for df in frames:
        qubit_ids |= set(df.loc[df["table"] == "qubit", "qubit"].map(component_key))
        ids, vals = _panel_rows(df, panel)
        ok = vals.notna() & (vals > 0) if panel.error_rate else vals.notna()
        roster |= set(ids)
        live |= set(ids[ok])
    two_qubit = any("," in c for c in roster)
    if not two_qubit:
        roster |= qubit_ids          # a value-less one-qubit gate keeps no row
    elif couplers:
        roster |= {f"{a},{b}" for a, b in couplers}
    return len(roster), len(roster - live)


def _panel_fence_note(panels: list["Panel"], k: float) -> str:
    rates = [p.error_rate for p in panels]
    where = ("log10 of the pooled positive values" if all(rates) else
             "the pooled values" if not any(rates) else
             "log10 of the pooled positive values, on the raw values for a non-rate panel")
    return (f"(Tukey k={k:g} fences on {where}; "
            f"dead = roster entries with no usable value in any snapshot)")


def series_roster(series: "Series") -> tuple[set[str], set[tuple[int, int]]]:
    qubits: set[str] = set()
    edges: set[tuple[int, int]] = set()
    for path in series.paths:
        wide = wide_form_path(path)
        if wide is not None:
            q, e = wide_roster(wide)
            qubits |= q
            edges |= e
    return qubits, edges


def panel_summary(series: "Series", panels: list["Panel"], k: float = 1.5) -> pd.DataFrame:
    qubits, couplers = series_roster(series)
    rows = []
    for panel in panels:
        v = series.values(panel)
        if panel.error_rate:
            v = v[v > 0]
        roster, dead = panel_components(series.frames, panel, qubits, couplers)
        if v.size >= 2:
            x = np.log10(v) if panel.error_rate else v
            lo, hi = tukey_fence(x, k)
            keep = (x >= lo) & (x <= hi)
            lo, hi = (10 ** lo, 10 ** hi) if panel.error_rate else (lo, hi)
            n_keep = int(keep.sum())
        else:
            lo = hi = float("nan")
            n_keep = 0
        rows.append({"panel": panel.title, "components": roster, "dead": dead,
                     "pooled": int(v.size), "kept": n_keep, "dropped": int(v.size) - n_keep,
                     "kept_frac": n_keep / v.size if v.size else float("nan"),
                     "lower_fence": lo, "upper_fence": hi,
                     "data_max": float(v.max()) if v.size else float("nan")})
    return pd.DataFrame(rows)


PANEL_SUMMARY_COLUMNS = (("panel", "Panel"), ("components", "Components"), ("dead", "Dead"),
                       ("pooled", "Pooled"), ("kept", "Kept"), ("dropped", "Dropped"),
                       ("upper_fence", "Upper fence"), ("data_max", "Data max"))


def format_panel_summary(stats: pd.DataFrame, title: str | None = None,
                         note: str | None = None) -> str:
    def cell(row, key):
        if key == "panel":
            return str(row[key])
        if key == "kept":
            return f"{row['kept']:,} ({row['kept_frac']:.1%})"
        if key in ("upper_fence", "data_max"):
            return f"{row[key]:.3g}"
        return f"{row[key]:,}"

    header = [label for _, label in PANEL_SUMMARY_COLUMNS]
    body = [[cell(r, key) for key, _ in PANEL_SUMMARY_COLUMNS] for _, r in stats.iterrows()]
    width = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
             for i, h in enumerate(header)]
    just = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                               for i, (c, w) in enumerate(zip(r, width)))
    lines = []
    if title:
        lines.append(title)
    lines += [just(header), "  ".join("-" * w for w in width)]
    lines += [just(r) for r in body]
    if note:
        lines.append(note)
    return "\n".join(lines)


def print_panel_summary(series: "Series", panels: list["Panel"], k: float = 1.5,
                      file=None) -> pd.DataFrame:
    """Print `format_panel_summary` and hand back the frame it was built from."""
    stats = panel_summary(series, panels, k=k)
    print(format_panel_summary(stats, title=series.label,
                               note=_panel_fence_note(panels, k)), file=file or sys.stdout)
    return stats


def pe3_lmom_fit(x: np.ndarray) -> dict[str, float] | None:
    from lmoments3 import distr
    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.ptp(x) == 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = distr.pe3.lmom_fit(x)
    except (ValueError, ZeroDivisionError):
        return None
    return {k: float(fit[k]) for k in ("skew", "loc", "scale")}


def fitted_stats(v: np.ndarray, log_scale: bool) -> dict[str, float]:
    from scipy.stats import pearson3
    x = np.asarray(v, dtype=float)
    nan = float("nan")
    if log_scale:
        x = x[x > 0]
        fit = pe3_lmom_fit(np.log10(x))
        if fit is None:
            return {"med": nan, "log sd": nan, "log skew": nan}
        median = 10 ** pearson3(fit["skew"], loc=fit["loc"], scale=fit["scale"]).median()
        return {"med": float(median), "log sd": fit["scale"], "log skew": fit["skew"]}
    fit = pe3_lmom_fit(x)
    if fit is None:
        return {"mean": nan, "sd": nan, "skew": nan}
    return {"mean": fit["loc"], "sd": fit["scale"], "skew": fit["skew"]}


def format_stats(st: dict[str, float]) -> str:
    def g(k: str, x: float) -> str:
        if not np.isfinite(x):
            return "nan"
        return f"{x:.0%}" if k == "kept" else f"{x:.3g}"
    return "  ".join(f"{'med' if k == 'median' else k} {g(k, x)}" for k, x in st.items())


# ---- drawing ----------------------------------------------------------------

def grid_shape(n_panels: int, ncols: int = NCOLS) -> tuple[int, int]:
    cols = max(1, min(ncols, n_panels))
    return max(1, math.ceil(n_panels / cols)), cols


def figure_size(n_panels: int, size: tuple[float, float] | None = None,
                ncols: int = NCOLS) -> tuple[float, float]:
    """Inches: the caller's `--size`, else PANEL_W x PANEL_H per cell of `grid_shape`."""
    if size is not None:
        return size
    nrows, ncols = grid_shape(n_panels, ncols)
    return (PANEL_W * ncols, PANEL_H * nrows)


STATS_MODES = ("raw", "log", "fit")


def panel_stats(v: np.ndarray, log_scale: bool, mode: str) -> dict[str, float]:
    if mode == "fit":
        return fitted_stats(v, log_scale)
    if mode == "log" and log_scale:
        return log_summary_stats(v)
    return summary_stats(v)


def panel_lines(v: np.ndarray, log_scale: bool, mode: str) -> list[str]:
    if mode == "log" and log_scale:
        return [format_stats(log_summary_stats(v)),
                "bulk: " + format_stats(bulk_log_stats(v))]
    prefix = "PE3 fit: " if mode == "fit" else ""
    return [prefix + format_stats(panel_stats(v, log_scale, mode))]


def draw(series: list[Series], 
         panels: list[Panel], 
         title: str | None = None,
         stats: str | None = None,
         nbins: int = NBINS,
         size: tuple[float, float] | None = None, 
         ncols: int = NCOLS):
    import matplotlib.pyplot as plt

    n = len(panels)
    nrows, ncols = grid_shape(n, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size(n, size, ncols),
                             constrained_layout=True, squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    averaged = any(len(s.frames) > 1 for s in series)

    for ax, panel in zip(axes.flat, panels):
        per_series = [s.values(panel) for s in series]
        if panel.log_scale:
            dropped = sum(int((v <= 0).sum()) for v in per_series)
            per_series = [v[v > 0] for v in per_series]
        else:
            dropped = 0
        pooled = np.concatenate(per_series)
        edges = bin_edges(pooled, panel.log_scale, nbins)

        for k, (s, v) in enumerate(zip(series, per_series)):
            if v.size == 0:
                continue
            c = colors[k % len(colors)]
            w = np.full(v.size, s.weight)
            ax.hist(v, bins=edges, weights=w, histtype="stepfilled", alpha=0.35, color=c,
                    label=s.label if ax is axes.flat[0] else None)
            ax.hist(v, bins=edges, weights=w, histtype="step", color=c, linewidth=1.2)

        if panel.log_scale:
            ax.set_xscale("log")
        if stats:
            lines = [(colors[k % len(colors)], text)
                     for k, v in enumerate(per_series) if v.size
                     for text in panel_lines(v, panel.log_scale, stats)]
            # One text artist per line so each takes its series colour;
            # stacked from the top-right corner, over a translucent backing.
            for j, (c, text) in enumerate(lines):
                ax.text(0.98, 0.97 - 0.09 * j, text, transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, family="monospace", color=c,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
        sub = f"\n({dropped} non-positive value{'s' if dropped != 1 else ''} not shown)" if dropped else ""
        ax.set_title(panel.title + sub, fontsize=10)
        ax.set_ylabel(f"{panel.unit_label} per snapshot" if averaged else panel.unit_label)
        ax.grid(True, alpha=0.3)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), max(2, ncols - 1)),
                   frameon=False, fontsize=9)
    excluded = sum((s.dead for s in series), DeadCount())
    if excluded.any:
        # Wrapped to the figure width: the saved PNG is cropped to its
        # contents, so an over-long title line would widen the image past the
        # panel grid (a two-panel selection would come out three panels wide).
        # ~10 characters per inch at 11 pt is a conservative DejaVu Sans width.
        width = max(40, int(fig.get_size_inches()[0] * 10))
        note = textwrap.fill(f"excluded across all series: {excluded.describe()}", width)
        title = (title or "Calibration histograms") + "\n" + note
    # fig.suptitle(title or "Calibration histograms", fontsize=11)
    return fig


def plot_histograms(paths: "list[Path | str] | tuple" = (), 
                    devices: "list[str] | tuple" = (), *,
                    calibration_dir: Path = DEFAULT_CALIBRATION_DIR, 
                    snapshots: int | None = 1,
                    keep_dead: bool = False,
                    stats: str | None = None, 
                    nbins: int = NBINS,
                    linear: bool = False,
                    ncols: int = NCOLS,
                    size: tuple[float, float] | None = None, 
                    title: str | None = None,
                    include: list[str] | None = None,
                    verbose: bool = True) -> tuple["Figure", list[Series], list[Panel]]:
    if stats is not None and stats not in STATS_MODES:
        raise ValueError(f"stats must be one of {STATS_MODES} or None, got {stats!r}")
    series = [load_series(series_label(Path(p)), [Path(p)], keep_dead) for p in paths]
    series += resolve_devices(list(devices), Path(calibration_dir), snapshots=snapshots,
                              keep_dead=keep_dead)
    if not series:
        raise ValueError("nothing to plot: give at least one CSV path or device name")
    for s in series:
        if s.dead.any and verbose:
            print(f"{s.label}: excluded {s.dead.describe()}", file=sys.stderr)

    panels, dropped = panel_specs([f for s in series for f in s.frames], linear=linear)
    if dropped and verbose:
        print("dropped empty panel(s): " + ", ".join(dropped), file=sys.stderr)
    if not panels:
        raise ValueError("nothing to plot: no panel has data in any input")
    if include:
        panels = select_panels(panels, list(include), dropped)

    # if title is None:
    #     n_snap = sum(len(s.frames) for s in series)
    #     title = (f"Calibration histograms: {len(series)} series, "
    #              f"{n_snap} snapshot{'s' if n_snap != 1 else ''}")
    fig = draw(series, panels, title=title, stats=stats, nbins=nbins, size=size, ncols=ncols)
    return fig, series, panels


def series_device(label: str) -> str:
    """The device a series label names: `ibm_miami 2026-...` and
    `ibm_miami: 8 snapshots, ...` both give `ibm_miami`; a bare filename
    label gives the filename."""
    return label.split(" ")[0].rstrip(":")


def default_out(labels: list[str]) -> Path:
    """`plots/histograms_<device>_<device>.png`, each device once, in order."""
    seen: list[str] = []
    for d in map(series_device, labels):
        if d not in seen:
            seen.append(d)
    return DEFAULT_PLOT_DIR / f"histograms_{'_'.join(seen)}.png"
"""Loader for legacy NBA SportVU optical tracking JSON files (2015-16 archives).

Real archive format (verified against e.g. ``0021500492.json`` from the public
mirrors)::

    {"gameid": "0021500492", "gamedate": "2016-01-01", "events": [...]}

Each event::

    {"eventId": int, "visitor": [...], "home": [...], "moments": [...]}

Each moment is a **list**::

    [quarter, epoch_ms, game_clock_s, shot_clock_s|None, None, positions]

where each position entry is ``[team_id, player_id, x_ft, y_ft, z_ft]`` and the
ball carries ``player_id == -1``. Coordinates are in **feet** on a full-court
frame (origin top-left, x along the length); we convert to meters and keep the
full-court frame for downstream rim-frame normalization. The epoch timestamp
(index 1) drives stoppage detection in ``split_contiguous_runs``.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEET_TO_METERS = 0.3048
BALL_PLAYER_ID = -1
FPS = 25.0
DT = 1.0 / FPS


def _parse_clock(s: str | None) -> float:
    """'PT11M36.00S' -> seconds remaining (nan if unparseable)."""
    if not s:
        return float("nan")
    m = re.match(r"^PT(?:(\d+)M)?([\d.]+)S$", str(s))
    if not m:
        return float("nan")
    return float(m.group(1) or 0) * 60.0 + float(m.group(2))


@dataclass
class GameTracking:
    """A single game's tracking data, keyed by event id."""

    gameid: str
    frames: dict[int, dict] = field(default_factory=dict)

    @property
    def event_ids(self) -> list[int]:
        return sorted(self.frames.keys())


def load_game(path: str | Path) -> GameTracking:
    """Load one SportVU game JSON (plain or gzipped)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        raw = json.load(f)
    gt = GameTracking(gameid=str(raw.get("gameid", path.stem)))
    for ev in raw.get("events", []):
        eid = int(ev.get("eventId", len(gt.frames)))
        gt.frames[eid] = ev
    return gt


def ensure_extracted(raw_dir: str | Path) -> list[Path]:
    """Extract any *.7z archives found under raw_dir (requires py7zr).

    Returns the list of JSON paths available after extraction.

    Each archive is extracted to a per-archive subdirectory named after the
    archive stem, so multi-game batches never collide or shadow each other:
    ``data/raw/01.01.2016.CHA.at.TOR.7z`` extracts to
    ``data/raw/01.01.2016.CHA.at.TOR/0021500492.json``. Archives whose
    subdirectory already contains a JSON are skipped (idempotent re-runs).
    """
    raw_dir = Path(raw_dir)
    archives = sorted(raw_dir.rglob("*.7z"))
    if archives:
        try:
            import py7zr
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                f"Found {len(archives)} .7z archive(s) but py7zr is not installed. "
                "Run: pip install py7zr"
            ) from e
        for arc in archives:
            out_dir = arc.parent / arc.stem
            if any(out_dir.glob("*.json")):
                continue  # already extracted
            out_dir.mkdir(parents=True, exist_ok=True)
            with py7zr.SevenZipFile(arc) as z:
                z.extractall(out_dir)
    return find_sportvu_files(raw_dir)


def find_sportvu_files(root: str | Path) -> list[Path]:
    """Glob a data directory for raw game files (*.json or *.json.gz).

    Duplicate copies of the same game (e.g. a loose JSON next to a freshly
    extracted archive copy) are deduplicated by (filename, size), keeping the
    shallowest path.
    """
    root = Path(root)
    candidates = [p for pat in ("*.json", "*.json.gz") for p in root.rglob(pat)]
    best: dict[tuple[str, int], Path] = {}
    for p in candidates:
        key = (p.name, p.stat().st_size)
        if key not in best or len(p.parts) < len(best[key].parts):
            best[key] = p
    return sorted(best.values())


def event_to_arrays(event: dict) -> dict:
    """Convert one raw event into aligned per-frame arrays (meters).

    The player ordering is whatever the file uses (typically 5 home + 5 visitor
    + ball), verified per-frame via an entity signature; the run terminates at
    a lineup change or a tracking gap. Returns dict with:

        xy          : (T, 11, 2) float32, full-court meters
        quarter     : (T,) int16
        game_clock  : (T,) float32 seconds remaining
        shot_clock  : (T,) float32 seconds (nan where absent)
        entity_meta : (11, 2) int — [team_id, player_id] per slot (ball last)
    """
    moments = event.get("moments", [])
    rows: list[list[float]] = []
    quarters: list[int] = []
    epochs: list[float] = []
    gclocks: list[float] = []
    sclocks: list[float] = []
    entity_meta: np.ndarray | None = None

    for mom in moments:
        quarter = int(mom[0])
        epoch_ms = float(mom[1])
        gclock = float(mom[2])  # seconds remaining
        sclock = float(mom[3]) if mom[3] is not None else float("nan")
        positions = mom[5] if len(mom) > 5 and mom[5] is not None else []
        players = [p for p in positions if int(p[1]) != BALL_PLAYER_ID]
        balls = [p for p in positions if int(p[1]) == BALL_PLAYER_ID]
        if len(players) != 10 or len(balls) != 1:
            continue
        sig = tuple((int(p[0]), int(p[1])) for p in players)
        if entity_meta is None:
            entity_meta = np.array([list(s) for s in sig] + [[-1, -1]], dtype=np.int64)
        elif tuple(map(tuple, entity_meta[:-1].tolist())) != sig:
            break  # lineup change terminates the contiguous run

        row: list[float] = []
        for p in players:
            row.extend([p[2], p[3]])
        row.extend([balls[0][2], balls[0][3]])
        rows.append(row)
        quarters.append(quarter)
        epochs.append(epoch_ms)
        gclocks.append(gclock)
        sclocks.append(sclock)

    if not rows:
        return {}
    xy = np.asarray(rows, dtype=np.float32).reshape(-1, 11, 2) * FEET_TO_METERS
    return {
        "xy": xy,
        "quarter": np.asarray(quarters, dtype=np.int16),
        "epoch_s": np.asarray(epochs, dtype=np.float64) / 1000.0,
        "game_clock": np.asarray(gclocks, dtype=np.float32),
        "shot_clock": np.asarray(sclocks, dtype=np.float32),
        "entity_meta": entity_meta,
    }


def split_contiguous_runs(arrs: dict, max_gap: float = 0.12) -> list[dict]:
    """Split an event's arrays into contiguous 25 fps runs.

    Breaks where the epoch-timestamp gap deviates from 0.04 s by more than
    `max_gap` (whistles, stoppages) or where the quarter changes.
    """
    if not arrs:
        return []
    xy, q, ep, gc, sc = (
        arrs["xy"],
        arrs["quarter"],
        arrs["epoch_s"],
        arrs["game_clock"],
        arrs["shot_clock"],
    )
    meta = arrs["entity_meta"]
    run_bounds = [0]
    for i in range(1, len(xy)):
        if q[i] != q[i - 1] or abs((ep[i] - ep[i - 1]) - DT) > max_gap:
            run_bounds.append(i)
    run_bounds.append(len(xy))
    runs = []
    for a, b in zip(run_bounds[:-1], run_bounds[1:]):
        if b - a >= 2:
            runs.append(
                {
                    "xy": xy[a:b],
                    "quarter": q[a:b],
                    "game_clock": gc[a:b],
                    "shot_clock": sc[a:b],
                    "entity_meta": meta,
                }
            )
    return runs


def sliding_windows(xy: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    """Slice a continuous run into (window, 11, 2) windows fully in one half.

    Half-court membership is determined from the ball's x coordinate relative
    to mid-court (94 ft court, meters).
    """
    n = xy.shape[0]
    if n < window:
        return []
    mid = 0.5 * 94.0 * FEET_TO_METERS
    out = []
    for start in range(0, n - window + 1, stride):
        w = xy[start : start + window]
        ball_x = w[:, 10, 0]
        if ball_x.min() > mid or ball_x.max() < mid:
            out.append(w.astype(np.float32))
    return out

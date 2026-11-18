"""Estructura matricial de una jugada: entidades x frames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from .geometry import FPS
from .loader import SIDE_BALL, SIDE_DEF, SIDE_OFF

SNAP_EVENTS = ("ball_snap", "autoevent_ballsnap")
THROW_EVENTS = ("pass_forward", "autoevent_passforward")
ARRIVE_EVENTS = ("pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete",
                 "pass_outcome_interception", "pass_outcome_touchdown", "autoevent_passinterrupted")


@dataclass
class PlayArrays:
    game_id: int
    play_id: int
    frames: np.ndarray            # (F,) frameId
    ids: np.ndarray               # (E,) nflId, -1 para el balón
    side: np.ndarray              # (E,) 0 ataque, 1 defensa, 2 balón
    team: list[str]
    jersey: np.ndarray            # (E,) float, NaN si no hay
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    a: np.ndarray
    dis: np.ndarray
    o: np.ndarray
    dir: np.ndarray
    events: dict[str, int] = field(default_factory=dict)  # evento -> primer frameId
    flipped: bool = False         # True si la jugada original iba hacia la izquierda

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    def idx(self, frame_id: int | None) -> int | None:
        """Índice de columna para un frameId (None si no existe)."""
        if frame_id is None:
            return None
        pos = int(np.searchsorted(self.frames, frame_id))
        if pos < self.n_frames and self.frames[pos] == frame_id:
            return pos
        return None

    def first_event(self, names: tuple[str, ...]) -> int | None:
        for n in names:
            if n in self.events:
                return self.events[n]
        return None

    @property
    def snap_frame(self) -> int | None:
        return self.first_event(SNAP_EVENTS)

    @property
    def throw_frame(self) -> int | None:
        return self.first_event(THROW_EVENTS)

    @property
    def arrive_frame(self) -> int | None:
        found = [self.events[n] for n in ARRIVE_EVENTS if n in self.events]
        return min(found) if found else None

    def entity(self, player_id: int) -> int | None:
        hit = np.flatnonzero(self.ids == player_id)
        return int(hit[0]) if hit.size else None

    def player_mask(self) -> np.ndarray:
        return self.side != SIDE_BALL

    def ball_index(self) -> int | None:
        hit = np.flatnonzero(self.side == SIDE_BALL)
        return int(hit[0]) if hit.size else None


def split_plays(df: pd.DataFrame, game_id: int) -> Iterator[PlayArrays | tuple[int, str]]:
    """Genera PlayArrays por jugada. Si una jugada no es rectangular produce (play_id, motivo)."""
    play_col = df["play_id"].to_numpy()
    if play_col.size == 0:
        return
    bounds = np.flatnonzero(np.diff(play_col)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [play_col.size]))
    pid = df["player_id"].to_numpy()
    fr = df["frame"].to_numpy()
    cols = {c: df[c].to_numpy(np.float32) for c in ("x", "y", "s", "a", "dis", "o", "dir")}
    side = df["side"].to_numpy()
    team = df["team"].to_numpy()
    jersey = df["jersey"].to_numpy(np.float32) if "jersey" in df else np.full(len(df), np.nan, np.float32)
    event = df["event"].to_numpy()
    flipped = df["flipped"].to_numpy()
    for s0, e0 in zip(starts, ends):
        p = pid[s0:e0]
        ent_starts = np.concatenate(([0], np.flatnonzero(np.diff(p)) + 1))
        n_ent = ent_starts.size
        n_rows = e0 - s0
        if n_rows % n_ent != 0:
            yield int(play_col[s0]), "entidades con distinto número de frames"
            continue
        n_fr = n_rows // n_ent
        f2 = fr[s0:e0].reshape(n_ent, n_fr)
        if not (f2 == f2[0]).all():
            yield int(play_col[s0]), "frames no alineados entre entidades"
            continue
        ev: dict[str, int] = {}
        evs = event[s0:e0]
        has = pd.notna(evs)
        if has.any():
            for name, frame in zip(evs[has], fr[s0:e0][has]):
                name = str(name)
                if name == "None":
                    continue
                if name not in ev or frame < ev[name]:
                    ev[name] = int(frame)
        shape = (n_ent, n_fr)
        first = ent_starts + s0
        yield PlayArrays(
            game_id=game_id, play_id=int(play_col[s0]), frames=f2[0].astype(np.int32),
            ids=p[ent_starts].astype(np.int64), side=side[first], team=[str(t) for t in team[first]],
            jersey=jersey[first],
            x=cols["x"][s0:e0].reshape(shape), y=cols["y"][s0:e0].reshape(shape),
            s=cols["s"][s0:e0].reshape(shape), a=cols["a"][s0:e0].reshape(shape),
            dis=cols["dis"][s0:e0].reshape(shape), o=cols["o"][s0:e0].reshape(shape),
            dir=cols["dir"][s0:e0].reshape(shape), events=ev, flipped=bool(flipped[s0]),
        )


def nearest_opposing(pa: PlayArrays) -> tuple[np.ndarray, np.ndarray]:
    """Distancia (yardas) y entidad del rival más cercano para cada jugador y frame. Balón: NaN."""
    E, F = pa.x.shape
    dist = np.full((E, F), np.nan, dtype=np.float32)
    who = np.full((E, F), -1, dtype=np.int64)
    off = np.flatnonzero(pa.side == SIDE_OFF)
    dfn = np.flatnonzero(pa.side == SIDE_DEF)
    if off.size == 0 or dfn.size == 0:
        return dist, who
    dx = pa.x[off][:, None, :] - pa.x[dfn][None, :, :]
    dy = pa.y[off][:, None, :] - pa.y[dfn][None, :, :]
    d = np.hypot(dx, dy)  # (off, def, F)
    dist[off] = d.min(axis=1)
    who[off] = pa.ids[dfn][d.argmin(axis=1)]
    dist[dfn] = d.min(axis=0)
    who[dfn] = pa.ids[off][d.argmin(axis=0)]
    return dist, who


SECONDS_PER_FRAME = 1.0 / FPS

"""Métricas por jugada: movimiento por jugador, separación en el lanzamiento y objetivo del pase."""
from __future__ import annotations

import re

import numpy as np

from .geometry import FPS
from .loader import SIDE_BALL, SIDE_DEF, SIDE_OFF
from .play import PlayArrays

ROLE_ROUTE, ROLE_QB, ROLE_RUSH, ROLE_COVER = "Pass Route", "Pass", "Pass Rush", "Coverage"
SPEED_BINS = 13  # [0,1), ..., [11,12), >=12 yardas/s
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_TARGET_RE = re.compile(r"\b(?:to|for) ([A-Z])\.\s?([A-Za-z'\-]+)")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def surname_of(full_name: str) -> str:
    parts = [p for p in re.split(r"\s+", full_name.strip()) if p]
    parts = [p for p in parts[1:] if _norm(p) not in SUFFIXES]
    return _norm("".join(parts))


def resolve_target(description: str | None, pass_result: str | None, candidates: list[tuple[int, str]]) -> int | None:
    """Receptor objetivo inferido del texto de la jugada. Solo se acepta si la coincidencia es única."""
    if not description or pass_result not in ("C", "I", "IN"):
        return None
    m = _TARGET_RE.search(description)
    if not m:
        return None
    initial, surname = m.group(1).lower(), _norm(m.group(2))
    if len(surname) < 2:
        return None
    hits = [pid for pid, name in candidates
            if name and _norm(name.split()[0])[:1] == initial and surname_of(name).startswith(surname)]
    return hits[0] if len(hits) == 1 else None


def analyze_play(pa: PlayArrays, meta: dict, roles: dict[int, tuple[str, str]], names: dict[int, str]) -> dict:
    """`meta`: fila de plays (absolute_yardline, pass_result, description). `roles`: id -> (rol, alineación)."""
    F = pa.n_frames
    si = pa.idx(pa.snap_frame) if pa.snap_frame is not None else 0
    si = 0 if si is None else si
    ti = pa.idx(pa.throw_frame)
    ai = pa.idx(pa.arrive_frame)
    ball = pa.ball_index()
    abs_line = meta.get("absolute_yardline")
    if abs_line is not None and np.isfinite(abs_line):
        los_x = float(120.0 - abs_line if pa.flipped else abs_line)
    elif ball is not None:
        los_x = float(pa.x[ball, si])
    else:
        los_x = float("nan")

    ent = np.flatnonzero(pa.side != SIDE_BALL)
    Sw = pa.s[ent, si:]
    bins = np.minimum(np.nan_to_num(Sw).astype(np.int64), SPEED_BINS - 1)
    hist = np.stack([(bins == k).sum(axis=1) for k in range(SPEED_BINS)], axis=1).astype(np.int32)
    role = [roles.get(int(i), (None, None))[0] for i in pa.ids[ent]]
    lined = [roles.get(int(i), (None, None))[1] for i in pa.ids[ent]]
    rows = {
        "player_id": pa.ids[ent], "side": pa.side[ent], "team": [pa.team[i] for i in ent],
        "role": role, "lined_up": lined,
        "max_speed": np.nanmax(Sw, axis=1), "mean_speed": np.nanmean(Sw, axis=1),
        "max_acc": np.nanmax(pa.a[ent, si:], axis=1),
        "dist": np.nansum(pa.dis[ent, si + 1:], axis=1) if F > si + 1 else np.zeros(ent.size),
        "duration": np.full(ent.size, (F - 1 - si) / FPS), "sep_throw": np.full(ent.size, np.nan),
        "hist": hist,
    }

    qb = next((i for i in range(pa.ids.size) if pa.side[i] == SIDE_OFF and roles.get(int(pa.ids[i]), ("",))[0] == ROLE_QB), None)
    routes = [i for i in range(pa.ids.size) if pa.side[i] == SIDE_OFF and roles.get(int(pa.ids[i]), ("",))[0] == ROLE_ROUTE]
    defs = np.flatnonzero(pa.side == SIDE_DEF)
    covers = [i for i in defs if roles.get(int(pa.ids[i]), ("",))[0] == ROLE_COVER]
    rushers = [i for i in defs if roles.get(int(pa.ids[i]), ("",))[0] == ROLE_RUSH]

    play = {"game_id": pa.game_id, "play_id": pa.play_id, "flipped": pa.flipped, "los_x": los_x, "n_frames": F,
            "snap_frame": pa.snap_frame, "throw_frame": pa.throw_frame, "arrive_frame": pa.arrive_frame,
            "qb_id": int(pa.ids[qb]) if qb is not None else None, "time_to_throw": np.nan,
            "rush_min": np.nan, "rush_within5": np.nan, "target_id": None, "n_routes": len(routes)}
    receivers: list[dict] = []
    if ti is not None and qb is not None and ti >= si:
        play["time_to_throw"] = (ti - si) / FPS
        qx, qy = float(pa.x[qb, ti]), float(pa.y[qb, ti])
        if rushers:
            dq = np.hypot(pa.x[rushers, ti] - qx, pa.y[rushers, ti] - qy)
            play["rush_min"] = float(dq.min())
            play["rush_within5"] = int((dq < 5.0).sum())
        if routes and defs.size:
            dmat = np.hypot(pa.x[routes, ti][:, None] - pa.x[defs, ti][None, :],
                            pa.y[routes, ti][:, None] - pa.y[defs, ti][None, :])
            sep = dmat.min(axis=1)
            near = pa.ids[defs][dmat.argmin(axis=1)]
            lookup = {int(i): k for k, i in enumerate(pa.ids[ent])}
            for j, r in enumerate(routes):
                rows["sep_throw"][lookup[int(pa.ids[r])]] = sep[j]
            for c in covers:
                rows["sep_throw"][lookup[int(pa.ids[c])]] = float(
                    np.hypot(pa.x[routes, ti] - pa.x[c, ti], pa.y[routes, ti] - pa.y[c, ti]).min())
            cand = [(int(pa.ids[r]), names.get(int(pa.ids[r]), "")) for r in routes]
            target = resolve_target(meta.get("description"), meta.get("pass_result"), cand)
            play["target_id"] = target
            for j, r in enumerate(routes):
                pre = pa.s[r, si:ti + 1]
                peak = int(np.nanargmax(pre)) + si if pre.size else si
                receivers.append({
                    "game_id": pa.game_id, "play_id": pa.play_id, "player_id": int(pa.ids[r]),
                    "x": float(pa.x[r, ti]), "y": float(pa.y[r, ti]), "sep": float(sep[j]),
                    "nearest_def_id": int(near[j]), "dist_qb": float(np.hypot(pa.x[r, ti] - qx, pa.y[r, ti] - qy)),
                    "depth": float(pa.x[r, ti] - los_x), "speed": float(pa.s[r, ti]),
                    "max_speed_pre": float(np.nanmax(pre)) if pre.size else float("nan"),
                    "peak_frame": int(pa.frames[peak]), "lined_up": roles[int(pa.ids[r])][1],
                    "is_target": int(pa.ids[r]) == target,
                })
    if ai is not None:
        play["arrive_idx"] = ai
    return {"play": play, "rows": rows, "receivers": receivers}

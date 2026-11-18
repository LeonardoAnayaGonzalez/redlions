"""Lógica de negocio de los endpoints: consultas, predicción, simulación y explicaciones."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .analysis import ROLE_QB, analyze_play
from .geometry import FIELD_LENGTH, FIELD_WIDTH, FPS, velocity_to_heading

from .loader import SIDE_BALL, SIDE_DEF, SIDE_OFF
from .motion import GROUPS, HORIZONS, LAG, kinematic_features, project_path, steady_features
from .play import PlayArrays, nearest_opposing
from .store import MIN_PLAYS, Store

MPH = 3600 * 3 / 5280  # yardas/s -> millas/hora
RESULT_LABEL = {"C": "pase completo", "I": "pase incompleto", "S": "sack", "IN": "intercepción", "R": "acarreo del QB (scramble)"}
ROLE_LABEL = {"Pass": "pasador (QB)", "Pass Route": "corre ruta", "Pass Block": "bloquea el pase",
              "Pass Rush": "presiona al QB", "Coverage": "cubre"}
SIDE_NAME = {SIDE_OFF: "off", SIDE_DEF: "def", SIDE_BALL: "ball"}
SPEED_EDGES = [f"{i}-{i + 1}" for i in range(12)] + ["12+"]
SEP_EDGES = list(range(0, 16))


def clean(o):
    """Convierte a tipos JSON-seguros (NaN/inf -> None, NumPy -> Python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.astype(object).where(df.notna(), None).to_dict("records"))


def need_ready(store: Store) -> None:
    st = store.status["state"]
    if st == "error":
        raise HTTPException(500, f"El preprocesamiento falló: {store.status['message']}")
    if st != "ready":
        raise HTTPException(503, f"El análisis global aún se está calculando ({store.status['done']}/{store.status['total']}). Reintenta en unos segundos.")


# ---------------------------------------------------------------------------- catálogo
def games_list(store: Store) -> list[dict]:
    g = store.games[store.games["game_id"].isin(store.files)].copy()
    counts = store.plays.groupby("game_id").size()
    g["plays"] = g["game_id"].map(counts).fillna(0).astype(int)
    g["label"] = g.apply(lambda r: f"Sem {r['week']} · {r['away']} @ {r['home']}", axis=1)
    test = set(store.test_games)
    g["split"] = g["game_id"].map(lambda i: ("validacion" if i in test else "calibracion") if store.status["state"] == "ready" else None)
    return records(g.sort_values("game_id")[["game_id", "week", "date", "home", "away", "plays", "label", "split"]])


def plays_list(store: Store, game_id: int | None, team: str | None, pass_result: str | None,
               play_type: str | None, limit: int, offset: int) -> dict:
    p = store.plays
    if game_id is not None:
        p = p[p["game_id"] == game_id]
    if team:
        p = p[p["possession"] == team]
    if pass_result:
        p = p[p["pass_result"] == pass_result]
    if play_type:
        p = p[p["play_type"] == play_type]
    total = int(len(p))
    p = p.sort_values(["game_id", "play_id"]).iloc[offset: offset + limit]
    cols = ["game_id", "play_id", "quarter", "down", "yards_to_go", "possession", "defense", "game_clock", "description",
            "pass_result", "play_result", "coverage", "coverage_type", "formation", "play_type", "defenders_box"]
    out = p[cols].copy()
    out["result_label"] = out["pass_result"].map(RESULT_LABEL)
    return {"total": total, "items": records(out)}


def play_meta(store: Store, game_id: int, play_id: int) -> dict:
    row = store.plays[(store.plays["game_id"] == game_id) & (store.plays["play_id"] == play_id)]
    if row.empty:
        raise HTTPException(404, f"La jugada {game_id}/{play_id} no existe en plays.csv.")
    return clean(row.iloc[0].to_dict())


def get_play(store: Store, game_id: int, play_id: int) -> PlayArrays:
    if game_id not in store.files:
        raise HTTPException(404, f"No hay tracking para el partido {game_id}.")
    plays = store.get_game(game_id)
    if play_id not in plays:
        raise HTTPException(404, f"La jugada {play_id} del partido {game_id} no tiene tracking utilizable.")
    return plays[play_id]


def roles_of(store: Store, game_id: int, play_id: int) -> dict:
    cache = store.__dict__.setdefault("_roles_cache", {})
    if game_id not in cache:
        cache[game_id] = store.roles_for_game(game_id)
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    return cache[game_id].get(play_id, {})


# ---------------------------------------------------------------------------- jugada
def play_detail(store: Store, game_id: int, play_id: int) -> dict:
    pa = get_play(store, game_id, play_id)
    meta = play_meta(store, game_id, play_id)
    roles = roles_of(store, game_id, play_id)
    dist, who = nearest_opposing(pa)
    cum = np.cumsum(np.nan_to_num(pa.dis), axis=1)
    snap = pa.snap_frame
    los = analyze_play(pa, meta, roles, store.names)["play"]["los_x"]
    r2 = lambda a: np.round(a.astype(float), 2)  # noqa: E731
    ents = []
    for i in range(pa.ids.size):
        pid = int(pa.ids[i])
        side = SIDE_NAME[int(pa.side[i])]
        role, lined = roles.get(pid, (None, None))
        ents.append({
            "id": pid, "name": "Balón" if side == "ball" else store.names.get(pid, f"Jugador {pid}"),
            "jersey": None if np.isnan(pa.jersey[i]) else int(pa.jersey[i]), "team": pa.team[i], "side": side,
            "position": store.positions.get(pid), "role": role, "lined_up": lined,
            "x": r2(pa.x[i]), "y": r2(pa.y[i]), "s": r2(pa.s[i]), "a": r2(pa.a[i]),
            "dir": r2(pa.dir[i]), "o": r2(pa.o[i]), "dist": r2(cum[i]),
            "nearest": r2(dist[i]), "nearest_id": who[i],
        })
    mask_off, mask_def = pa.side == SIDE_OFF, pa.side == SIDE_DEF
    events = sorted(({"frame": f, "event": e} for e, f in pa.events.items()), key=lambda d: d["frame"])
    ytg = meta.get("yards_to_go")
    return clean({
        "game_id": game_id, "play_id": play_id, "n_frames": pa.n_frames, "frames": pa.frames,
        "meta": {**meta, "result_label": RESULT_LABEL.get(meta.get("pass_result")),
                 "los_x": los, "first_down_x": los + ytg if ytg is not None else None,
                 "snap_frame": snap, "throw_frame": pa.throw_frame, "arrive_frame": pa.arrive_frame,
                 "originally_left": pa.flipped, "events": events,
                 "split": ("validacion" if game_id in set(store.test_games) else "calibracion") if store.status["state"] == "ready" else None},
        "entities": ents,
        "speed_by_side": {"off": r2(np.nanmean(pa.s[mask_off], axis=0)), "def": r2(np.nanmean(pa.s[mask_def], axis=0))},
        "units": {"position": "yardas", "speed": "yardas/s", "acceleration": "yardas/s²", "time": "frame = 0.1 s",
                  "orientation": "grados; 0 = hacia la banda superior, sentido horario"},
    })


def narrative(store: Store, game_id: int, play_id: int, player_id: int | None) -> dict:
    pa = get_play(store, game_id, play_id)
    meta = play_meta(store, game_id, play_id)
    roles = roles_of(store, game_id, play_id)
    res = analyze_play(pa, meta, roles, store.names)
    play = res["play"]
    play_txt = []
    down = meta.get("down")
    sit = f"{int(down)}ª y {int(meta['yards_to_go'])}" if down is not None and meta.get("yards_to_go") is not None else "situación sin dato"
    play_txt.append(f"{meta['possession']} ataca a {meta['defense']} ({sit}). Resultado registrado: "
                    f"{RESULT_LABEL.get(meta.get('pass_result'), 'sin dato')}.")
    if not np.isnan(play["time_to_throw"]):
        play_txt.append(f"El lanzamiento ocurre {play['time_to_throw']:.1f} s después del snap.")
    else:
        play_txt.append("No hay evento de lanzamiento en el tracking (por ejemplo sack o scramble).")
    if play["target_id"] is not None:
        rec = next((r for r in res["receivers"] if r["player_id"] == play["target_id"]), None)
        if rec:
            play_txt.append(f"El receptor objetivo (inferido de la descripción) es {store.names.get(play['target_id'])}, "
                            f"con {rec['sep']:.1f} yd de separación del defensor más cercano al lanzar.")
    if not np.isnan(play["rush_min"]):
        play_txt.append(f"Al lanzar, el rival que presiona más cercano está a {play['rush_min']:.1f} yd del QB.")
    if meta.get("coverage"):
        play_txt.append(f"Cobertura registrada por PFF: {meta['coverage']} ({meta.get('coverage_type') or 'sin tipo'}).")

    player_txt: list[str] = []
    if player_id is not None:
        ei = pa.entity(player_id)
        if ei is None or pa.side[ei] == SIDE_BALL:
            raise HTTPException(404, f"El jugador {player_id} no participa en esta jugada.")
        name = store.names.get(player_id, f"Jugador {player_id}")
        role, lined = roles.get(player_id, (None, None))
        si = pa.idx(pa.snap_frame) or 0
        speeds = pa.s[ei, si:]
        pk = int(np.argmax(speeds))
        player_txt.append(f"{name} ({store.positions.get(player_id, 'posición sin dato')}, alineado como {lined or 'sin dato'}) "
                          f"{ROLE_LABEL.get(role, 'participa')} en esta jugada.")
        player_txt.append(f"Su velocidad máxima fue {speeds[pk]:.1f} yd/s (≈ {speeds[pk] * MPH:.0f} mph), "
                          f"{pk / FPS:.1f} s después del snap.")
        player_txt.append(f"Recorrió {float(np.nansum(pa.dis[ei, si + 1:])):.1f} yd desde el snap hasta el final del registro.")
        dist, _ = nearest_opposing(pa)
        d = dist[ei]
        ti = pa.idx(pa.throw_frame)
        end_i = ti if ti is not None else pa.n_frames - 1
        label = "el lanzamiento" if ti is not None else "el final de la jugada"
        delta = float(d[end_i] - d[si])
        trend = "aumenta" if delta > 1.0 else "disminuye" if delta < -1.0 else "cambia poco"
        player_txt.append(f"Su distancia al rival más cercano pasa de {d[si]:.1f} yd en el snap a {d[end_i]:.1f} yd en {label} "
                          f"({trend}: {delta:+.1f} yd).")
        if player_id == play["target_id"]:
            player_txt.append("Fue el objetivo del pase según la descripción de la jugada.")
    return {"play": play_txt, "player": player_txt}


# ---------------------------------------------------------------------------- análisis de pases
def pass_analysis(store: Store, game_id: int, play_id: int) -> dict:
    pa = get_play(store, game_id, play_id)
    meta = play_meta(store, game_id, play_id)
    roles = roles_of(store, game_id, play_id)
    res = analyze_play(pa, meta, roles, store.names)
    play, recs = res["play"], res["receivers"]
    if play["throw_frame"] is None:
        return {"available": False, "reason": "Esta jugada no tiene evento de lanzamiento (sack, scramble o falta de dato)."}
    if not recs:
        return {"available": False, "reason": "No hay receptores elegibles identificados (pff_role = 'Pass Route') en esta jugada."}
    df = pd.DataFrame(recs)
    df["name"] = df["player_id"].map(store.names)
    df["nearest_def_name"] = df["nearest_def_id"].map(store.names)
    df["time_to_throw"], df["rush_min"] = play["time_to_throw"], play["rush_min"]
    probs = store.completion_probability(df) if store.status["state"] == "ready" else None
    df["p_complete"] = probs if probs is not None else np.nan
    df["rank_sep"] = df["sep"].rank(ascending=False, method="min").astype(int)
    if store.status["state"] == "ready" and store.recv is not None and len(store.recv):
        pool = store.recv["sep"].to_numpy()
        df["sep_percentile"] = [100.0 * (pool < v).mean() for v in df["sep"]]
    else:
        df["sep_percentile"] = np.nan
    most_open = df.sort_values("sep", ascending=False).iloc[0]
    tgt = df[df["is_target"]]
    lines = [f"Al lanzar, {most_open['name']} era la opción con mayor separación ({most_open['sep']:.1f} yd)."]
    if not tgt.empty:
        t = tgt.iloc[0]
        lines.append(f"El pase fue dirigido a {t['name']}, que ocupaba el puesto {int(t['rank_sep'])} de {len(df)} en separación "
                     f"({t['sep']:.1f} yd).")
        if int(t["rank_sep"]) > 1:
            lines.append("El receptor más abierto no fue el objetivo. La elección también depende de profundidad, "
                         "presión, trayectoria y lectura de coberturas, que esta medida no captura.")
    else:
        lines.append("No se pudo identificar al receptor objetivo desde la descripción de la jugada.")
    return clean({
        "available": True, "throw_frame": play["throw_frame"], "time_to_throw": play["time_to_throw"],
        "qb_id": play["qb_id"], "result": meta.get("pass_result"), "result_label": RESULT_LABEL.get(meta.get("pass_result")),
        "rush_min": play["rush_min"], "los_x": play["los_x"], "target_id": play["target_id"],
        "completion_model": {"enabled": bool(store.completion and store.completion.get("enabled")),
                             "reason": (store.completion or {}).get("reason") if store.status["state"] == "ready" else "Calculando modelos."},
        "receivers": records(df.sort_values("sep", ascending=False)[[
            "player_id", "name", "lined_up", "x", "y", "sep", "nearest_def_id", "nearest_def_name", "dist_qb", "depth",
            "speed", "is_target", "rank_sep", "sep_percentile", "p_complete"]]),
        "summary": lines,
        "caveat": "La separación describe oportunidad, no la mejor decisión. P(completar) es una probabilidad modelada "
                  "con datos de pases realmente lanzados.",
    })


# ---------------------------------------------------------------------------- resumen global
def _filter_pp(store: Store, game_id, team, player_id, play_type, pass_result):
    pp = store.pp
    mask = np.ones(len(pp), dtype=bool)
    if game_id is not None:
        mask &= (pp["game_id"] == game_id).to_numpy()
    if team:
        mask &= (pp["team"] == team).to_numpy()
    if player_id is not None:
        mask &= (pp["player_id"] == player_id).to_numpy()
    if play_type:
        mask &= (pp["play_type"] == play_type).to_numpy()
    if pass_result:
        mask &= (pp["pass_result"] == pass_result).to_numpy()
    return mask


def summary(store: Store, game_id, team, player_id, play_type, pass_result) -> dict:
    need_ready(store)
    pp = store.pp
    mask = _filter_pp(store, game_id, team, player_id, play_type, pass_result)
    sub = pp[mask]
    filters = {"game_id": game_id, "team": team, "player_id": player_id, "play_type": play_type, "pass_result": pass_result}
    if sub.empty:
        return clean({"filters": filters, "empty": True, "message": "Ningún registro coincide con los filtros seleccionados.",
                      "play_types": sorted(store.plays["play_type"].dropna().unique().tolist()), "capabilities": store.capabilities})
    keys = sub["play_key"].unique()
    plays = store.plays[store.plays["play_key"].isin(keys)]
    ps = store.play_stats[store.play_stats["play_key"].isin(keys)]
    results = plays["pass_result"].value_counts()
    attempts = int(results.reindex(["C", "I", "IN"]).fillna(0).sum())
    hist = store.pp_hist[mask].sum(axis=0)
    total = max(int(hist.sum()), 1)
    speed_hist = [{"bin": b, "count": int(c), "pct": round(100 * c / total, 2)} for b, c in zip(SPEED_EDGES, hist)]
    open_sep = sub[(sub["role"] == "Pass Route") & sub["sep_throw"].notna()]["sep_throw"].to_numpy()
    sep_hist = []
    for i, lo in enumerate(SEP_EDGES):
        hi = SEP_EDGES[i + 1] if i + 1 < len(SEP_EDGES) else np.inf
        sep_hist.append({"bin": f"{lo}-{hi}" if np.isfinite(hi) else f"{lo}+", "count": int(((open_sep >= lo) & (open_sep < hi)).sum())})
    by_pos = (sub.groupby("position").agg(n=("max_speed", "size"), mean_max_speed=("max_speed", "mean"),
                                          mean_dist=("dist", "mean"), mean_acc=("max_acc", "mean"))
              .reset_index().sort_values("mean_max_speed", ascending=False))
    by_pos = by_pos[by_pos["n"] >= 10]
    by_team = (sub.groupby("team").agg(total_dist=("dist", "sum"), n=("dist", "size")).reset_index())
    by_team["avg_dist"] = by_team["total_dist"] / by_team["n"]
    by_player = (sub.groupby("player_id").agg(total_dist=("dist", "sum"), plays=("dist", "size"), team=("team", "first"),
                                              position=("position", "first")).reset_index()
                 .sort_values("total_dist", ascending=False).head(10))
    by_player["name"] = by_player["player_id"].map(store.names)
    kpis = {
        "games": int(sub["game_id"].nunique()), "plays": int(len(keys)), "players": int(sub["player_id"].nunique()),
        "teams": int(sub["team"].nunique()), "max_speed": float(sub["max_speed"].max()),
        "max_speed_mph": float(sub["max_speed"].max() * MPH), "total_distance": float(sub["dist"].sum()),
        "pass_attempts": attempts, "completion_rate": float(results.get("C", 0) / attempts) if attempts else None,
        "sacks": int(results.get("S", 0)),
        "mean_time_to_throw": float(ps["time_to_throw"].mean()) if ps["time_to_throw"].notna().any() else None,
        "mean_open_separation": float(open_sep.mean()) if open_sep.size else None,
        "mean_player_distance": float(sub["dist"].mean()),
    }
    top_speed = sub.loc[sub["max_speed"].idxmax()]
    kpis["max_speed_holder"] = {"name": store.names.get(int(top_speed["player_id"])), "game_id": int(top_speed["game_id"]),
                                "play_id": int(top_speed["play_id"]), "player_id": int(top_speed["player_id"])}
    return clean({
        "filters": filters, "empty": False, "kpis": kpis, "speed_hist": speed_hist, "sep_hist": sep_hist,
        "by_position": records(by_pos), "distance_by_team": records(by_team.sort_values("total_dist", ascending=False)),
        "distance_by_player": records(by_player),
        "pass_results": [{"result": k, "label": RESULT_LABEL.get(k, k), "count": int(v)} for k, v in results.items()],
        "play_types": sorted(store.plays["play_type"].dropna().unique().tolist()),
        "capabilities": store.capabilities, "quality": store.quality,
        "source": {"files": ["games.csv", "plays.csv", "players.csv", "pffScoutingData.csv", f"tracking/ ({len(store.files)} archivos)"],
                   "origin": "NFL Big Data Bowl 2022, temporada 2021, semanas 1 a 8 (jugadas de pase)"},
    })


def teams(store: Store) -> list[str]:
    return sorted(set(store.games["home"]) | set(store.games["away"]))


# ---------------------------------------------------------------------------- jugadores
DIMS = [("speed", "Velocidad", "yd/s", "Percentil 90 de la velocidad máxima por jugada"),
        ("acceleration", "Aceleración", "yd/s²", "Percentil 90 de la aceleración máxima por jugada"),
        ("separation", "Separación", "yd", "Receptores: distancia media al defensor más cercano al lanzar. "
                                           "Defensores de cobertura: distancia media al receptor más cercano (menor = cobertura más pegada)"),
        ("consistency", "Consistencia", "", "Menor variación (coef. de variación) de la velocidad máxima entre jugadas"),
        ("mobility", "Movilidad", "yd/s", "Distancia total recorrida dividida entre el tiempo total en jugada")]


def _player_row(store: Store, player_id: int) -> pd.Series:
    t = store.player_table
    if player_id not in t.index:
        raise HTTPException(404, f"El jugador {player_id} no aparece en el tracking analizado.")
    return t.loc[player_id]


def players_list(store: Store, q, position, team, min_plays, sort, limit) -> dict:
    need_ready(store)
    t = store.player_table.reset_index()
    t = t[t["plays"] >= min_plays]
    if q:
        t = t[t["name"].str.contains(q, case=False, na=False, regex=False)]
    if position:
        t = t[(t["pos_group"] == position) | (t["position"] == position)]
    if team:
        t = t[t["team"] == team]
    key = {"composite": "composite", "speed": "p90_speed", "max_speed": "max_speed", "acceleration": "p90_acc",
           "distance": "dist_total", "plays": "plays", "mobility": "mobility"}[sort]
    total = int(len(t))
    t = t.sort_values(key, ascending=False, na_position="last").head(limit)
    cols = ["player_id", "name", "team", "position", "pos_group", "plays", "max_speed", "p90_speed", "mean_speed", "max_acc",
            "p90_acc", "dist_total", "mobility", "composite", "n_dims"]
    out = t[cols].copy()
    out["low_sample"] = out["plays"] < MIN_PLAYS
    return {"total": total, "min_plays": min_plays, "sort": sort, "items": records(out)}


def player_detail(store: Store, player_id: int) -> dict:
    need_ready(store)
    r = _player_row(store, player_id)
    t = store.player_table
    peers = t[(t["pos_group"] == r["pos_group"]) & (t["plays"] >= MIN_PLAYS)]
    radar, missing = [], []
    for key, label, unit, desc in DIMS:
        pct, raw, pool = r[f"pct_{key}"], r[f"raw_{key}"], int(r[f"pool_{key}"])
        if key == "separation" and pd.notna(raw):
            raw = raw if r["sep_kind"] == "open" else -raw
        if pd.isna(pct):
            if key == "acceleration" and not store.capabilities["acceleration"]["available"]:
                reason = "El tracking no incluye aceleración."
            elif key == "separation":
                reason = (f"Se calcula con al menos 15 pases observados como receptor o defensor de cobertura "
                          f"(tiene {int(r['sep_n'])}), y con 10 pares del mismo grupo.")
            else:
                reason = f"Hay menos de 10 jugadores con {MIN_PLAYS}+ jugadas en el grupo {r['pos_group']} para normalizar."
            missing.append({"key": key, "label": label, "reason": reason})
        radar.append({"key": key, "label": label, "unit": unit, "description": desc, "pct": pct,
                      "value": None if pd.isna(raw) else raw, "pool": pool})
    med = {"p90_speed": float(peers["p90_speed"].median()) if len(peers) else None,
           "dist_per_play": float((peers["dist_total"] / peers["plays"]).median()) if len(peers) else None}
    return clean({
        "player_id": player_id, "name": r["name"], "team": r["team"], "position": r["position"], "pos_group": r["pos_group"],
        "plays": int(r["plays"]), "max_speed": r["max_speed"], "mean_speed": r["mean_speed"], "p90_speed": r["p90_speed"],
        "max_acc": r["max_acc"], "p90_acc": r["p90_acc"], "dist_total": r["dist_total"], "dist_per_play": r["dist_total"] / r["plays"],
        "mobility": r["mobility"], "cv_speed": r["cv_speed"],
        "separation": {"kind": r["sep_kind"], "n": int(r["sep_n"]), "mean": None if pd.isna(r["sep_value"]) else r["sep_value"]},
        "composite": r["composite"], "n_dims": int(r["n_dims"]), "low_sample": bool(r["plays"] < MIN_PLAYS),
        "radar": radar, "missing_dimensions": missing,
        "peer_group": {"group": r["pos_group"], "n": int(len(peers)), "median": med, "min_plays": MIN_PLAYS},
        "note": "Percentiles descriptivos dentro del mismo grupo posicional en este dataset (semanas 1-8 de 2021). "
                "No son una evaluación definitiva del talento.",
    })


def players_compare(store: Store, a: int, b: int) -> dict:
    da, db = player_detail(store, a), player_detail(store, b)
    warnings = []
    if da["pos_group"] != db["pos_group"]:
        warnings.append(f"Los jugadores pertenecen a grupos posicionales distintos ({da['pos_group']} y {db['pos_group']}). "
                        "Cada percentil se calcula contra su propio grupo, por lo que no son directamente comparables en valor absoluto.")
    for d in (da, db):
        if d["low_sample"]:
            warnings.append(f"{d['name']} tiene solo {d['plays']} jugadas observadas (< {MIN_PLAYS}); sus métricas son poco estables.")
    ratio = max(da["plays"], db["plays"]) / max(min(da["plays"], db["plays"]), 1)
    if ratio >= 2:
        warnings.append(f"Los tamaños de muestra difieren mucho ({da['plays']} vs {db['plays']} jugadas).")
    rows = []
    for ra, rb in zip(da["radar"], db["radar"]):
        rows.append({"key": ra["key"], "label": ra["label"], "a": ra["pct"], "b": rb["pct"],
                     "diff": None if ra["pct"] is None or rb["pct"] is None else ra["pct"] - rb["pct"]})
    return clean({"a": da, "b": db, "dimensions": rows, "warnings": warnings})


# ---------------------------------------------------------------------------- predicción
def _entity_state(store: Store, pa: PlayArrays, player_id: int, frame_id: int):
    ei = pa.entity(player_id)
    if ei is None or pa.side[ei] == SIDE_BALL:
        raise HTTPException(404, f"El jugador {player_id} no participa en esta jugada.")
    fi = pa.idx(frame_id)
    if fi is None:
        raise HTTPException(404, f"El frame {frame_id} no existe en esta jugada (1 a {int(pa.frames[-1])}).")
    if fi < LAG:
        raise HTTPException(422, f"Historial insuficiente: se necesitan al menos {LAG} frames previos (0.5 s). Elige un frame ≥ {int(pa.frames[LAG])}.")
    return ei, fi


def _model(store: Store):
    need_ready(store)
    if store.motion_model is None:
        raise HTTPException(503, "El modelo de movimiento aún no está calibrado.")
    return store.motion_model


def _group_of(store: Store, pid: int) -> int:
    from .motion import pos_group
    return GROUPS.index(pos_group(store.positions.get(pid)))


def predict(store: Store, game_id: int, play_id: int, player_id: int, frame_id: int, horizon_s: float, reveal: bool) -> dict:
    model = _model(store)
    pa = get_play(store, game_id, play_id)
    ei, fi = _entity_state(store, pa, player_id, frame_id)
    k = int(round(horizon_s * FPS))
    feats = kinematic_features(pa)[ei, fi]
    if not np.isfinite(feats).all():
        raise HTTPException(422, "No hay datos de movimiento válidos para este jugador en ese frame.")
    x0, y0 = float(pa.x[ei, fi]), float(pa.y[ei, fi])
    group = _group_of(store, player_id)
    paths = project_path(model, group, feats, x0, y0, k)
    has_future = fi + k < pa.n_frames
    out = {
        "player_id": player_id, "name": store.names.get(player_id), "group": GROUPS[group], "frame_id": frame_id,
        "horizon_s": horizon_s, "current": {"x": x0, "y": y0, "speed": float(pa.s[ei, fi]), "dir": float(pa.dir[ei, fi]),
                                            "acc": float(pa.a[ei, fi])},
        "history": [{"frame": int(pa.frames[j]), "x": float(pa.x[ei, j]), "y": float(pa.y[ei, j])} for j in range(fi + 1)],
        "features": dict(zip(["vx", "vy", "dvx", "dvy", "pvx", "pvy", "ax", "ay"], feats.tolist())),
        "baseline": paths["base"], "model": paths["model"], "has_future": has_future,
        "split": "validacion" if game_id in set(store.test_games) else "calibracion",
        "validation": next((h for h in (store.motion_report or {}).get("horizons", []) if abs(h["horizon_s"] - horizon_s) < 1e-6), None),
        "revealed": False,
    }
    if reveal:
        if not has_future:
            raise HTTPException(422, "No existe posición real posterior: la jugada termina antes del horizonte pedido.")
        j = fi + k
        out["revealed"] = True
        out["actual_path"] = [{"frame": int(pa.frames[m]), "x": float(pa.x[ei, m]), "y": float(pa.y[ei, m])} for m in range(fi, j + 1)]
        ax_, ay_ = float(pa.x[ei, j]), float(pa.y[ei, j])
        out["actual"] = {"x": ax_, "y": ay_}
        out["errors"] = {"baseline": float(np.hypot(paths["base"][-1]["x"] - ax_, paths["base"][-1]["y"] - ay_)),
                         "model": float(np.hypot(paths["model"][-1]["x"] - ax_, paths["model"][-1]["y"] - ay_))}
        out["errors_by_horizon"] = [
            {"t": b["t"], "baseline": float(np.hypot(b["x"] - pa.x[ei, fi + int(round(b["t"] * FPS))], b["y"] - pa.y[ei, fi + int(round(b["t"] * FPS))])),
             "model": float(np.hypot(m["x"] - pa.x[ei, fi + int(round(m["t"] * FPS))], m["y"] - pa.y[ei, fi + int(round(m["t"] * FPS))]))}
            for b, m in zip(paths["base"], paths["model"]) if fi + int(round(b["t"] * FPS)) < pa.n_frames]
    return clean(out)


def simulate(store: Store, game_id: int, play_id: int, player_id: int, frame_id: int, dx: float, dy: float,
             speed: float | None, direction_deg: float | None, horizon_s: float) -> dict:
    model = _model(store)
    pa = get_play(store, game_id, play_id)
    ei, fi = _entity_state(store, pa, player_id, frame_id)
    k = int(round(horizon_s * FPS))
    feats = kinematic_features(pa)[ei, fi]
    x0, y0 = float(pa.x[ei, fi]), float(pa.y[ei, fi])
    group = _group_of(store, player_id)
    orig = project_path(model, group, feats, x0, y0, k)["model"]
    cur_speed, cur_dir = float(pa.s[ei, fi]), float(pa.dir[ei, fi])
    sx = float(np.clip(x0 + dx, 0, FIELD_LENGTH))
    sy = float(np.clip(y0 + dy, 0, FIELD_WIDTH))
    modified = speed is not None or direction_deg is not None
    sp = cur_speed if speed is None else speed
    dr = cur_dir if direction_deg is None else direction_deg
    sfeat = steady_features(sp, dr) if modified else feats
    scen = project_path(model, group, sfeat, sx, sy, k)["model"]
    t0 = {"t": 0.0}
    orig_path = [{**t0, "x": x0, "y": y0}] + orig
    scen_path = [{**t0, "x": sx, "y": sy}] + scen
    real_future = [{"t": (m - fi) / FPS, "x": float(pa.x[ei, m]), "y": float(pa.y[ei, m])}
                   for m in range(fi, min(fi + k, pa.n_frames - 1) + 1)]

    opp_side = SIDE_DEF if pa.side[ei] == SIDE_OFF else SIDE_OFF
    feats_all = kinematic_features(pa)
    opponents, opp_paths = [], []
    for j in np.flatnonzero(pa.side == opp_side):
        f = feats_all[j, fi]
        if not np.isfinite(f).all():
            continue
        path = [{"t": 0.0, "x": float(pa.x[j, fi]), "y": float(pa.y[j, fi])}] + project_path(
            model, _group_of(store, int(pa.ids[j])), f, float(pa.x[j, fi]), float(pa.y[j, fi]), k)["model"]
        opp_paths.append((int(pa.ids[j]), path))
        opponents.append({"id": int(pa.ids[j]), "name": store.names.get(int(pa.ids[j])), "jersey": None if np.isnan(pa.jersey[j]) else int(pa.jersey[j]),
                          "position": store.positions.get(int(pa.ids[j])), "path": path})

    def separation(path):
        rows = []
        for n, p in enumerate(path):
            best = min(((float(np.hypot(p["x"] - op[n]["x"], p["y"] - op[n]["y"])), oid) for oid, op in opp_paths), default=(None, None))
            rows.append({"t": p["t"], "dist": best[0], "opp_id": best[1], "opp_name": store.names.get(best[1]) if best[1] else None})
        return rows
    sep_o, sep_s = separation(orig_path), separation(scen_path)
    min_o = min((r["dist"] for r in sep_o if r["dist"] is not None), default=None)
    min_s = min((r["dist"] for r in sep_s if r["dist"] is not None), default=None)
    return clean({
        "player_id": player_id, "name": store.names.get(player_id), "frame_id": frame_id, "horizon_s": horizon_s,
        "original_state": {"x": x0, "y": y0, "speed": cur_speed, "dir": cur_dir},
        "scenario_state": {"x": sx, "y": sy, "speed": sp, "dir": dr, "modified_motion": modified},
        "original_prediction": orig_path, "scenario_prediction": scen_path, "real_future": real_future,
        "opponents": opponents, "separation_original": sep_o, "separation_scenario": sep_s,
        "min_separation": {"original": min_o, "scenario": min_s,
                           "delta": None if min_o is None or min_s is None else min_s - min_o},
        "assumptions": [
            "Los rivales mantienen su estado de movimiento observado en el frame elegido (modelo calibrado, sin reaccionar al jugador hipotético).",
            "El jugador hipotético conserva velocidad y dirección constantes si se modifican, sin aceleración ni giros." if modified
            else "Con solo cambiar la posición, el jugador conserva su estado observado (velocidad, giro y aceleración).",
            "La trayectoria se estima en pasos de 0.5 s; la separación es la distancia al rival más cercano en cada paso.",
            "No se modelan colisiones, bloqueos, límites de la banda ni decisiones tácticas.",
        ],
        "disclaimer": "Simulación exploratoria basada en un modelo simplificado. No representa necesariamente lo que habría ocurrido durante el partido.",
    })


# ---------------------------------------------------------------------------- modelo y demo
def model_metrics(store: Store) -> dict:
    need_ready(store)
    mr, comp = store.motion_report, store.completion or {}
    hz = mr["horizons"]
    insights = []
    maes = [h["model"]["mae"] for h in hz]
    if len(maes) > 1 and all(b > a for a, b in zip(maes, maes[1:])):
        insights.append(f"El error del modelo crece con el horizonte: de {maes[0]:.2f} yd a {hz[0]['horizon_s']:g} s hasta {maes[-1]:.2f} yd a {hz[-1]['horizon_s']:g} s.")
    wins = [h for h in hz if h["improvement"]["ci95"][0] > 0]
    if len(wins) == len(hz):
        insights.append(f"El modelo calibrado reduce el error medio frente a velocidad constante en todos los horizontes "
                        f"({min(h['improvement']['mae_pct'] for h in hz):.0f}% a {max(h['improvement']['mae_pct'] for h in hz):.0f}%), con intervalos de confianza al 95% por encima de cero.")
    elif wins:
        insights.append(f"El modelo calibrado supera al base solo en {len(wins)} de {len(hz)} horizontes; en el resto no hay mejora demostrada.")
    else:
        insights.append("El modelo calibrado no mejora al modelo de velocidad constante en los partidos de validación.")
    h1 = [g for g in mr["by_group"] if abs(g["horizon_s"] - 1.0) < 1e-6]
    if h1:
        worst = max(h1, key=lambda g: g["mae_model"])
        best = min(h1, key=lambda g: g["mae_model"])
        insights.append(f"A 1 s, el error del modelo es mayor en {worst['group']} ({worst['mae_model']:.2f} yd) y menor en {best['group']} ({best['mae_model']:.2f} yd).")
    if comp.get("metrics"):
        m = comp["metrics"]
        if comp.get("enabled"):
            insights.append(f"El modelo de completitud logra AUC {m['auc']:.2f} y log-loss {m['log_loss']:.3f} (base: {m['log_loss_base']:.3f}) en partidos no usados para entrenar.")
        else:
            insights.append("El modelo de completitud no supera al modelo base en validación; sus probabilidades están desactivadas.")
        sep = next((c for c in comp["coefficients"] if c["feature"] == "sep"), None)
        if sep and comp.get("enabled"):
            word = "mayor" if sep["beta_std"] > 0 else "menor"
            insights.append(f"La separación se asocia con {word} probabilidad de completar (asociación estadística, no efecto causal).")
    comp_pub = {k: v for k, v in comp.items() if k not in ("mu", "sd", "beta", "medians")}
    return clean({
        "motion": {"description": "Predicción de la posición de un jugador a 0.5-3.0 s a partir de su estado en el frame t.",
                   "variables": ["velocidad (s, dir) -> vx, vy", "cambio de velocidad en 0.5 s", "velocidad derivada de las posiciones de los últimos 0.5 s",
                                 "aceleración reportada (signo inferido de la tendencia de rapidez)"],
                   "baseline": "Velocidad constante: posición(t+h) = posición(t) + (vx, vy)·h",
                   "method": f"Regresión ridge (λ=10) por grupo posicional y horizonte, implementada con NumPy, sobre el residuo respecto al modelo base. Grupos: {', '.join(GROUPS)}.",
                   "error_metric": "Error de posición = distancia euclidiana (yardas) entre posición predicha y real.",
                   "report": mr, "coefficients": store.motion_model.coef_table(), "features": ["vx", "vy", "dvx", "dvy", "pvx", "pvy", "ax", "ay"]},
        "completion": comp_pub,
        "split": store.quality["split"] | {"test_game_ids": store.test_games},
        "insights": insights,
        "limitations": [
            "Los modelos de movimiento no conocen el diseño de la jugada, al balón ni a los demás jugadores: capturan inercia, no intención.",
            "La validación usa partidos completos reservados, pero todos son de las semanas 1 a 8 de 2021 y de jugadas de pase.",
            "El error se mide en yardas en el campo normalizado (ataque hacia +x). Con horizontes largos, cambios bruscos de ruta dominan el error.",
            "El receptor objetivo se infiere del texto de la jugada; no es una etiqueta oficial del dataset.",
            "El modelo de completitud se entrena solo con pases realmente lanzados: hay sesgo de selección y no estima qué habría pasado con otro receptor.",
            "Las asociaciones estadísticas no implican causalidad.",
        ],
        "quality": store.quality,
    })


def demo(store: Store) -> dict:
    if store.status["state"] != "ready":
        return {"available": False, "reason": "Los datos aún se están procesando." if store.status["state"] != "error" else store.status["message"]}
    recv = store.recv
    if recv is None or not len(recv):
        return {"available": False, "reason": "No hay receptores identificados en el tracking."}
    d = recv[recv["is_target"] & recv["game_id"].isin(store.test_games)]
    d = d.merge(store.plays[["play_key", "pass_result", "description", "possession", "defense"]], on="play_key")
    d = d.merge(store.play_stats[["play_key", "time_to_throw", "snap_frame", "throw_frame"]], on="play_key")
    d = d[(d["pass_result"] == "C") & d["time_to_throw"].between(1.5, 5.0)]
    if d.empty:
        return {"available": False, "reason": "No hay pases completos con receptor identificado en los partidos de validación."}
    for _, r in d.sort_values("max_speed_pre", ascending=False).head(40).iterrows():
        pa = store.get_game(int(r["game_id"])).get(int(r["play_id"]))
        if pa is None:
            continue
        ei = pa.entity(int(r["player_id"]))
        si = pa.idx(int(r["snap_frame"]))
        ti = pa.idx(int(r["throw_frame"]))
        if ei is None or si is None or ti is None:
            continue
        # instante de predicción: la mayor velocidad del receptor antes del lanzamiento con >= 2 s de futuro real
        lo, hi = max(LAG, si + 3), min(ti, pa.n_frames - 1 - 20)
        if hi <= lo:
            continue
        fi = lo + int(np.argmax(pa.s[ei, lo:hi + 1]))
        horizon = 2.0
        r = r.copy()
        r["peak_frame"] = int(pa.frames[fi])
        r["max_speed_pre"] = float(pa.s[ei, fi])
        probs = pass_analysis(store, int(r["game_id"]), int(r["play_id"]))
        return clean({
            "available": True, "game_id": int(r["game_id"]), "play_id": int(r["play_id"]), "player_id": int(r["player_id"]),
            "name": r["name"], "frame_id": int(r["peak_frame"]), "horizon_s": horizon, "description": r["description"],
            "max_speed": r["max_speed_pre"], "separation": r["sep"], "time_to_throw": r["time_to_throw"],
            "throw_frame": int(r["throw_frame"]), "snap_frame": int(r["snap_frame"]),
            "criteria": [
                "Partido reservado para validación (no usado para calibrar el modelo).",
                "Pase completo con receptor objetivo identificado.",
                f"Tiempo hasta el lanzamiento de {r['time_to_throw']:.1f} s (entre 1.5 y 5 s).",
                "Entre las candidatas, es el receptor que alcanza la mayor velocidad antes del lanzamiento "
                "(medida en el instante elegido para predecir).",
                f"En ese instante corre a {r['max_speed_pre']:.1f} yd/s (≈ {r['max_speed_pre'] * MPH:.0f} mph).",
                f"Hay {horizon:g} s de movimiento real posterior para comparar con la predicción.",
            ],
            "pass": probs,
        })
    return {"available": False, "reason": "Ninguna candidata tiene historial y futuro suficientes para validar la predicción."}


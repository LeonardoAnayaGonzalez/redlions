"""Servicios del Playbook Creator: escenario defensivo real, evaluación y generación de jugadas hipotéticas."""
from __future__ import annotations

import itertools
from collections import OrderedDict

import numpy as np
from fastapi import HTTPException

from . import playbook as PB
from .analysis import analyze_play
from .geometry import FIELD_WIDTH, FPS
from .loader import SIDE_DEF, SIDE_OFF
from .motion import GROUPS
from .services import clean, get_play, need_ready, play_meta, roles_of
from .store import Store

KIND = "hipotetica"
DISCLAIMER = ("Jugada hipotética generada por el modelo. No se jugó en un partido real: la defensa parte de las posiciones "
              "reales en el snap y su reacción es simulada.")
OFFICIAL_TO_POS = {"T": "OL", "G": "OL", "C": "OL", "QB": "QB", "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "HB": "RB"}
_scen_cache: OrderedDict = OrderedDict()


def _models(store: Store) -> PB.Models:
    need_ready(store)
    comp = store.completion or {}
    if not comp.get("enabled") or store.yards_model is None or store.rush_curve is None:
        raise HTTPException(503, store.capabilities.get("playbook", {}).get("reason") or "Modelos de evaluación no disponibles.")
    return PB.Models(completion=comp, yards=store.yards_model, rush_curve=np.asarray(store.rush_curve, dtype=np.float64),
                     window=store.throw_window)


def meta(store: Store) -> dict:
    need_ready(store)
    comp = store.completion or {}
    return clean({
        "formations": PB.FORMATIONS, "routes": {k: {"label": v["label"], "points": v["points"], "settle": v["settle"]}
                                                 for k, v in PB.ROUTE_TEMPLATES.items()},
        "weights": PB.DEFAULT_WEIGHTS, "window": store.throw_window, "steps": _steps(store), "dt": PB.DT,
        "rush_curve": [round(float(x), 3) for x in store.rush_curve] if store.rush_curve is not None else None,
        "limits": {g: store.limits[g] for g in ("WR", "TE", "RB", "QB")} if store.limits else None,
        "yards_model": store.yards_model, "coverage_validation": store.coverage_validation,
        "completion": {"enabled": comp.get("enabled"), "auc": (comp.get("metrics") or {}).get("auc"),
                       "features": comp.get("features")},
        "rules": {"on_line_depth": PB.ON_LINE_DEPTH, "min_spacing": PB.MIN_SPACING, "receiver_spacing": PB.RECEIVER_SPACING,
                  "edge": PB.EDGE},
        "defense_model": {"man_delay_s": PB.MAN_DELAY_S, "man_lead_s": PB.MAN_LEAD_S, "zone_radius": PB.ZONE_RADIUS,
                          "zone_deep_depth": PB.ZONE_DEEP_DEPTH, "zone_under_depth": PB.ZONE_UNDER_DEPTH,
                          "qb_drop_depth": PB.QB_DROP_DEPTH},
        "completion_ranges": comp.get("ranges"),
        "formula": {"ev": "EY_i(t) = P_i(t) · (Y_i(t) + B · 1[Y_i(t) ≥ yardas para 1er down])",
                    "best": "EY_i = max_{t ∈ W} EY_i(t)", "score": "S = w1·EY₍₁₎ + w2·EY₍₂₎ + w3·EY₍₃₎",
                    "yards": "Y = a + b·profundidad + c·separación"},
        "kind": KIND, "disclaimer": DISCLAIMER,
    })


def _steps(store: Store) -> int:
    hi = store.throw_window[1] if store.throw_window else 4.0
    return int(max(40, round(hi * FPS)))


# ---------------------------------------------------------------------------- escenario
def scenario_data(store: Store, game_id: int, play_id: int) -> dict:
    key = (game_id, play_id)
    if key in _scen_cache:
        _scen_cache.move_to_end(key)
        return _scen_cache[key]
    pa = get_play(store, game_id, play_id)
    m = play_meta(store, game_id, play_id)
    roles = roles_of(store, game_id, play_id)
    si = pa.idx(pa.snap_frame)
    if si is None:
        raise HTTPException(422, "La jugada no tiene evento de snap: no se pueden tomar las posiciones iniciales de la defensa.")
    b = pa.ball_index()
    if b is not None and np.isfinite(pa.x[b, si]):
        los, ball_y = float(pa.x[b, si]), float(pa.y[b, si])     # posición del balón en el snap
    else:
        los, ball_y = float(analyze_play(pa, m, roles, store.names)["play"]["los_x"]), FIELD_WIDTH / 2
    ent = np.flatnonzero(pa.side == SIDE_DEF)
    codes = store.group_codes(pa)[ent]
    gl = [GROUPS[c] for c in codes]
    d = PB.Defense(pos0=np.stack([pa.x[ent, si], pa.y[ent, si]], axis=1).astype(np.float64),
                   rush=np.array([roles.get(int(pa.ids[e]), ("",))[0] == "Pass Rush" for e in ent]),
                   v_max=np.array([store.limits[g]["v_max"] for g in gl]), a_max=np.array([store.limits[g]["a_max"] for g in gl]),
                   man=m.get("coverage_type") == "Man", los_x=los, ids=[int(pa.ids[e]) for e in ent])
    out = {"pa": pa, "meta": m, "roles": roles, "si": si, "los": los, "ball_y": ball_y, "defense": d, "ent": ent,
           "ytg": float(m.get("yards_to_go") or 10)}
    _scen_cache[key] = out
    while len(_scen_cache) > 16:
        _scen_cache.popitem(last=False)
    return out


def _real_formation(store: Store, sc: dict) -> list[dict]:
    pa, si, roles = sc["pa"], sc["si"], sc["roles"]
    ti = pa.idx(pa.throw_frame)
    end = min(pa.n_frames - 1, (ti if ti is not None else si + 30) + 5, si + 40)
    used: dict[str, int] = {}
    players = []
    for i in np.flatnonzero(pa.side == SIDE_OFF):
        pid = int(pa.ids[i])
        role, lined = roles.get(pid, (None, None))
        pos = OFFICIAL_TO_POS.get(str(store.positions.get(pid)), "WR")
        slot = str(lined or store.positions.get(pid) or pid)
        used[slot] = used.get(slot, 0) + 1
        if used[slot] > 1:
            slot = f"{slot}{used[slot]}"
        route = []
        if role == "Pass Route":
            route = [[round(float(pa.x[i, k]), 2), round(float(pa.y[i, k]), 2)] for k in range(si + 5, end + 1, 5)]
        players.append({"slot": slot, "pos": pos, "x": round(float(pa.x[i, si]), 2), "y": round(float(pa.y[i, si]), 2),
                        "route": route, "route_name": "real" if route else None, "settle": False,
                        "player_id": pid, "name": store.names.get(pid)})
    # El tracking mide el centro del cuerpo: se normaliza la profundidad a la convención del editor.
    # Los 7 jugadores (sin el QB) más cercanos a la línea quedan en la línea y el resto en el backfield.
    los = sc["los"]
    others = sorted((p for p in players if p["pos"] != "QB"), key=lambda p: -p["x"])
    for k, p in enumerate(others):
        p["x"] = round(los - 0.8, 2) if k < 7 else round(min(p["x"], los - 2.0), 2)
    for p in players:
        if p["pos"] == "QB":
            p["x"] = round(min(p["x"], los - 1.2), 2)
    return players


def scenario(store: Store, game_id: int, play_id: int) -> dict:
    need_ready(store)
    sc = scenario_data(store, game_id, play_id)
    pa, si, roles, m, d = sc["pa"], sc["si"], sc["roles"], sc["meta"], sc["defense"]
    assign_mode = "hombre" if d.man else "zona"
    defenders = [{"id": int(pa.ids[e]), "name": store.names.get(int(pa.ids[e])),
                  "jersey": None if np.isnan(pa.jersey[e]) else int(pa.jersey[e]), "position": store.positions.get(int(pa.ids[e])),
                  "role": roles.get(int(pa.ids[e]), (None,))[0], "x": float(pa.x[e, si]), "y": float(pa.y[e, si]),
                  "mode": "presión" if d.rush[j] else assign_mode} for j, e in enumerate(sc["ent"])]
    return clean({
        "game_id": game_id, "play_id": play_id, "los_x": sc["los"], "ball_y": sc["ball_y"], "yards_to_go": sc["ytg"],
        "down": m.get("down"), "quarter": m.get("quarter"), "possession": m.get("possession"), "defense_team": m.get("defense"),
        "coverage": m.get("coverage"), "coverage_type": m.get("coverage_type"), "coverage_model": assign_mode,
        "description": m.get("description"), "defenders": defenders, "real_formation": _real_formation(store, sc),
        "source": "real", "note": "Posiciones defensivas reales en el snap de esta jugada. Lo que ocurre después en el editor es simulado.",
        "real_formation_note": ("Formación ofensiva real en el snap, con la profundidad normalizada (línea a 0.8 yd, backfield a 2 yd "
                                "o más) porque el tracking mide el centro del cuerpo. Sus rutas son la trayectoria real muestreada "
                                "cada 0.5 s y se recorren con el modelo de movimiento."),
    })


# ---------------------------------------------------------------------------- simulación de una propuesta
def _limits(store: Store, pos: str) -> tuple[float, float]:
    lim = store.limits[PB.POS_GROUP.get(pos, "WR")]
    return float(lim["v_max"]), float(lim["a_max"])


def _runner_path(store: Store, p: dict, steps: int):
    v, a = _limits(store, p["pos"])
    wps = [PB.clip_point(float(x), float(y)) for x, y in p["route"]]
    return PB.follow_route((float(p["x"]), float(p["y"])), wps, v, a, steps, bool(p.get("settle")))


def _check_players(players: list[dict]) -> None:
    slots = [p["slot"] for p in players]
    if len(set(slots)) != len(slots):
        raise HTTPException(422, "Cada jugador debe tener un identificador (slot) único.")


def _run(store: Store, sc: dict, players: list[dict], weights: dict) -> dict:
    """Simula y evalúa una propuesta. Devuelve todo lo necesario para el detalle."""
    m = _models(store)
    steps = _steps(store)
    leg = PB.legality(players, sc["los"], sc["ball_y"])
    runners = [p for p in players if p.get("route") and p["pos"] in PB.RECEIVER_POS]
    qbs = [p for p in players if p["pos"] == "QB"]
    qb = PB.qb_path((float(qbs[0]["x"]), float(qbs[0]["y"])) if qbs else (sc["los"] - PB.QB_DROP_DEPTH, sc["ball_y"]), sc["los"], steps)
    info = [_runner_path(store, p, steps) for p in runners]
    if not runners:
        return {"leg": leg, "runners": [], "info": [], "qb": qb, "res": None, "steps": steps, "players": players, "assign": None}
    rec_pos = np.stack([i[0] for i in info])[None]
    rec_vel = np.stack([i[1] for i in info])[None]
    assign = PB.man_assignment(sc["defense"], rec_pos[0, :, 0])
    res = PB.evaluate_batch(rec_pos, rec_vel, qb, sc["defense"], assign, m, weights, sc["los"], sc["ytg"])
    return {"leg": leg, "runners": runners, "info": info, "qb": qb, "res": res, "steps": steps, "players": players,
            "assign": assign, "models": m}


def _detail(store: Store, sc: dict, run: dict, weights: dict) -> dict:
    r2 = lambda a: np.round(np.asarray(a, dtype=float), 2)  # noqa: E731
    pa, d, steps = sc["pa"], sc["defense"], run["steps"]
    runners, info, res = run["runners"], run["info"], run["res"]
    rslots = [p["slot"] for p in runners]
    offense = []
    for p in run["players"]:
        if p in runners:
            k = runners.index(p)
            path = info[k][0]
        elif p["pos"] == "QB":
            path = run["qb"]
        else:
            path = np.tile([float(p["x"]), float(p["y"])], (steps + 1, 1))
        offense.append({"slot": p["slot"], "pos": p["pos"], "path": r2(path), "has_route": p in runners,
                        "eligible": p["slot"] in run["leg"]["eligible"], "name": p.get("name")})
    out = {"kind": KIND, "disclaimer": DISCLAIMER, "legality": run["leg"], "steps": steps, "dt": PB.DT, "offense": offense,
           "los_x": sc["los"], "ball_y": sc["ball_y"], "yards_to_go": sc["ytg"], "weights": weights}
    if res is None:
        out.update({"valid": False, "score": None, "options": [], "defense": [], "explanation":
                    ["Asigna al menos una ruta a un receptor elegible para evaluar la jugada."], "warnings": _warnings(store)})
        return clean(out)
    defs = res["defs"][0]
    assign = run["assign"]
    out["defense"] = [{"id": d.ids[j], "name": store.names.get(d.ids[j]), "jersey": None if np.isnan(pa.jersey[e]) else int(pa.jersey[e]),
                       "position": store.positions.get(d.ids[j]), "mode": "presión" if d.rush[j] else ("hombre" if assign[j] >= 0 else "zona"),
                       "assigned_to": rslots[assign[j]] if assign[j] >= 0 else None, "path": r2(defs[j])}
                      for j, e in enumerate(sc["ent"])]
    W = res["W"]
    in_field = [bool(i[3]) for i in info]
    options = []
    for k, p in enumerate(runners):
        options.append({"slot": p["slot"], "pos": p["pos"], "route_name": p.get("route_name"), "route_time": info[k][2],
                        "in_field": in_field[k], "ev": float(res["best"][0, k]), "t_star": float(res["t_star"][0, k]),
                        "p": float(res["p"][0, k]), "yhat": float(res["yhat"][0, k]), "sep": float(res["sep_star"][0, k]),
                        "depth": float(res["depth_star"][0, k]), "ev_curve": r2(res["ev"][0, k]), "p_curve": r2(res["p_all"][0, k]),
                        "sep_curve": r2(res["sep"][0, k])})
    order = sorted(range(len(options)), key=lambda i: -options[i]["ev"])
    wts = [weights["w1"], weights["w2"], weights["w3"]]
    comps = [{"rank": n + 1, "slot": options[i]["slot"], "ev": options[i]["ev"], "weight": wts[n], "contribution": wts[n] * options[i]["ev"]}
             for n, i in enumerate(order[:3])]
    collision = bool(res["collision"][0])
    valid = run["leg"]["legal"] and all(in_field) and not collision
    out.update({"valid": valid, "collision": collision, "options": options, "read_order": [options[i]["slot"] for i in order],
                "window": [float(W[0] * PB.DT), float(W[-1] * PB.DT)], "window_t": r2(W * PB.DT),
                "score": {"S": float(res["S"][0]), "components": comps},
                "coverage": {"name": sc["meta"].get("coverage"), "type": sc["meta"].get("coverage_type"), "model": "hombre" if d.man else "zona"}})
    out["explanation"] = _explain(store, sc, out)
    out["warnings"] = _warnings(store)
    return clean(out)


def _explain(store: Store, sc: dict, o: dict) -> list[str]:
    opts = {x["slot"]: x for x in o["options"]}
    comps = o["score"]["components"]
    best = opts[comps[0]["slot"]]
    lines = [f"Puntuación S = {o['score']['S']:.2f} yardas esperadas ponderadas: "
             + " + ".join(f"{c['weight']:g}·{c['ev']:.2f} ({c['slot']})" for c in comps) + "."]
    lines.append(f"Mejor opción: {best['slot']} ({best['route_name'] or 'ruta dibujada'}), lanzando a los {best['t_star']:.1f} s: "
                 f"separación prevista {best['sep']:.1f} yd, profundidad {best['depth']:.1f} yd, P(completar) {best['p'] * 100:.0f}% "
                 f"y {best['yhat']:.1f} yd si se completa (EY = {best['ev']:.2f}).")
    if len(comps) > 1:
        lines.append("Lecturas siguientes: " + "; ".join(f"{c['slot']} con EY = {c['ev']:.2f}" for c in comps[1:])
                     + ". Pesan menos porque solo importan si la primera opción está cubierta.")
    deep = max(o["options"], key=lambda x: x["depth"])
    if deep["slot"] != best["slot"] and deep["depth"] > best["depth"] + 3:
        lines.append(f"La opción más profunda ({deep['slot']}, {deep['depth']:.1f} yd) aporta menos: su P(completar) baja a "
                     f"{deep['p'] * 100:.0f}% y su EY queda en {deep['ev']:.2f}.")
    wide = max(o["options"], key=lambda x: x["sep"])
    if wide["slot"] != best["slot"] and wide["sep"] > best["sep"] + 1:
        why = "está detrás de la línea de golpeo" if wide["depth"] < 0 else f"solo gana {wide['yhat']:.1f} yd si se completa"
        lines.append(f"El receptor más separado ({wide['slot']}, {wide['sep']:.1f} yd) no es la mejor opción porque {why}.")
    lines.append(f"Situación del escenario real: {sc['ytg']:.0f} yd para primer down. Los lanzamientos se evalúan entre "
                 f"{o['window'][0]:.1f} y {o['window'][1]:.1f} s (percentiles 10-90 del tiempo real hasta el lanzamiento).")
    cov = o["coverage"]
    lines.append(f"La defensa real jugó {cov['name'] or 'cobertura sin dato'} ({cov['type'] or 'sin tipo'}); se simula como {cov['model']}.")
    if not o["legality"]["legal"]:
        lines.append("La formación no cumple todas las reglas básicas: revisa la lista de legalidad. No se recomendará.")
    if o.get("collision"):
        lines.append("Dos receptores pasan a menos de 1 yd entre sí: la jugada no se recomienda hasta separar sus rutas.")
    if not all(x["in_field"] for x in o["options"]):
        lines.append("Alguna ruta sale del campo: ajústala antes de evaluar o guardar.")
    return lines


def _warnings(store: Store) -> list[str]:
    w = [DISCLAIMER,
         "La reacción defensiva usa reglas simples (persecución en hombre, zonas con radio fijo y presión según la curva mediana). "
         "Una defensa real puede rotar, cambiar asignaciones o leer la jugada.",
         "P(completar) y las yardas esperadas vienen de modelos ajustados con pases reales lanzados; aplicarlos a jugadas nuevas es una extrapolación.",
         "No se modelan bloqueos individuales, la lectura del QB, el vuelo del balón ni el contacto."]
    cv = next((r for r in (store.coverage_validation or []) if r["coverage"] == "Todas"), None)
    if cv:
        w.append(f"Validación del modelo defensivo en {cv['n']} receptores de partidos reservados: error medio de la separación al lanzar "
                 f"{cv['mae']:.2f} yd (defensores quietos: {cv['mae_static']:.2f} yd), sesgo {cv['bias']:+.2f} yd.")
    return w


def evaluate(store: Store, game_id: int, play_id: int, players: list[dict], weights: dict) -> dict:
    need_ready(store)
    _check_players(players)
    sc = scenario_data(store, game_id, play_id)
    return _detail(store, sc, _run(store, sc, players, weights), weights)


# ---------------------------------------------------------------------------- generador
def generate(store: Store, game_id: int, play_id: int, players: list[dict], weights: dict, n_samples: int,
             locked: list[str], seed: int) -> dict:
    need_ready(store)
    _check_players(players)
    m = _models(store)
    sc = scenario_data(store, game_id, play_id)
    steps = _steps(store)
    base_leg = PB.legality([{**p, "route": []} for p in players], sc["los"], sc["ball_y"])
    formation_issues = [c for c in base_leg["checks"] if not c["ok"] and c["rule"] not in ("Al menos una ruta",)]
    if formation_issues:
        raise HTTPException(422, "La formación no es legal: " + " ".join(c["detail"] for c in formation_issues))
    eligible = [p for p in players if p["slot"] in base_leg["eligible"]]
    if not eligible:
        raise HTTPException(422, "No hay receptores elegibles en la formación.")
    locked_set = {s for s in locked if any(p["slot"] == s and p.get("route") for p in eligible)}
    slots, options = [], []
    for p in eligible:
        opts = []
        if p["slot"] in locked_set or p.get("route"):
            path, vel, done, inside = _runner_path(store, p, steps)
            opts.append({"name": p.get("route_name") or "dibujada", "route": p["route"], "settle": bool(p.get("settle")),
                         "path": path, "vel": vel, "done": done, "inside": inside, "custom": True})
        if p["slot"] not in locked_set:
            v, a = _limits(store, p["pos"])
            for name in PB.ROUTE_TEMPLATES:
                wps, settle = PB.template_waypoints(name, (float(p["x"]), float(p["y"])), sc["ball_y"])
                path, vel, done, inside = PB.follow_route((float(p["x"]), float(p["y"])), wps, v, a, steps, settle)
                if inside:
                    opts.append({"name": name, "route": [list(w) for w in wps], "settle": settle, "path": path, "vel": vel,
                                 "done": done, "inside": True, "custom": False})
        if not opts:
            continue
        slots.append(p)
        options.append(opts)
    sizes = [len(o) for o in options]
    total = int(np.prod(sizes))
    rng = np.random.default_rng(seed)
    if total <= n_samples:
        combos = list(itertools.product(*[range(s) for s in sizes]))
    else:
        combos = {tuple(int(rng.integers(s)) for s in sizes) for _ in range(n_samples)}
        combos = list(combos)
    qbs = [p for p in players if p["pos"] == "QB"]
    qb = PB.qb_path((float(qbs[0]["x"]), float(qbs[0]["y"])), sc["los"], steps)
    rec0 = np.array([[float(p["x"]), float(p["y"])] for p in slots])
    assign = PB.man_assignment(sc["defense"], rec0)
    cache: dict[tuple, tuple[float, bool]] = {}

    def run_batch(cs: list[tuple]) -> None:
        cs = [c for c in cs if c not in cache]
        for i in range(0, len(cs), 150):
            chunk = cs[i:i + 150]
            rp = np.stack([np.stack([options[k][c[k]]["path"] for k in range(len(slots))]) for c in chunk])
            rv = np.stack([np.stack([options[k][c[k]]["vel"] for k in range(len(slots))]) for c in chunk])
            r = PB.evaluate_batch(rp, rv, qb, sc["defense"], assign, m, weights, sc["los"], sc["ytg"])
            for c, s, col in zip(chunk, r["S"], r["collision"]):
                cache[c] = (float(s), bool(col))

    run_batch(combos)
    n_random = len(cache)

    def ranked() -> list[tuple]:
        return sorted(cache, key=lambda c: (cache[c][1], -cache[c][0]))

    for seed_combo in ranked()[:5]:            # ascenso por coordenadas desde las mejores semillas
        cur = list(seed_combo)
        for _ in range(2):
            improved = False
            for k in range(len(slots)):
                trial = [tuple(cur[:k] + [o] + cur[k + 1:]) for o in range(sizes[k])]
                run_batch(trial)
                best = min(trial, key=lambda c: (cache[c][1], -cache[c][0]))
                if best != tuple(cur):
                    cur, improved = list(best), True
            if not improved:
                break
    order = ranked()
    min_diff = 2 if len(slots) >= 3 else 1
    top: list[tuple] = []
    for c in order:
        if all(sum(a != b for a, b in zip(c, t)) >= min_diff for t in top):
            top.append(c)
        if len(top) == 3:
            break
    results = []
    for rank, c in enumerate(top, 1):
        routes = {slots[k]["slot"]: options[k][c[k]] for k in range(len(slots))}
        new_players = []
        for p in players:
            if p["slot"] in routes:
                o = routes[p["slot"]]
                new_players.append({**p, "route": [[round(x, 2), round(y, 2)] for x, y in o["route"]],
                                    "route_name": o["name"], "settle": o["settle"]})
            else:
                new_players.append({**p, "route": [], "route_name": None} if p["pos"] in PB.RECEIVER_POS else p)
        det = _detail(store, sc, _run(store, sc, new_players, weights), weights)
        results.append({"rank": rank, "players": new_players, "routes": {s: o["name"] for s, o in routes.items()}, "detail": det})
    scores = np.array([v[0] for v in cache.values()])
    hist_edges = np.linspace(scores.min(), scores.max() + 1e-9, 16) if scores.size else np.array([0, 1])
    counts, _ = np.histogram(scores, bins=hist_edges)
    current = None
    if any(p.get("route") for p in players):
        cd = evaluate(store, game_id, play_id, players, weights)
        current = {"S": (cd.get("score") or {}).get("S"), "valid": cd.get("valid")}
    expl = [f"Se evaluaron {len(cache)} combinaciones de rutas ({n_random} por muestreo y el resto por ascenso por coordenadas "
            f"desde las 5 mejores), sobre {total:,} posibles con {len(slots)} receptores elegibles.".replace(",", " ")]
    if results:
        b = results[0]["detail"]["score"]["S"]
        expl.append(f"La mejor obtiene S = {b:.2f}. Las tres propuestas difieren al menos en {min_diff} ruta(s) para ofrecer alternativas distintas.")
        if len(results) > 1:
            gap = b - results[1]["detail"]["score"]["S"]
            expl.append(f"La diferencia con la segunda es {gap:.2f}; con el error del modelo defensivo, diferencias pequeñas no son concluyentes.")
        if current and current["S"] is not None:
            d = b - current["S"]
            expl.append(f"Frente al diseño actual (S = {current['S']:.2f}), la mejor propuesta {'mejora' if d > 0 else 'no mejora'} en {abs(d):.2f}.")
    return clean({"kind": KIND, "disclaimer": DISCLAIMER, "evaluated": len(cache), "possible": total, "n_random": n_random,
                  "slots": [p["slot"] for p in slots], "locked": sorted(locked_set), "top": results, "current": current,
                  "distribution": {"edges": np.round(hist_edges, 3), "counts": counts}, "explanation": expl})


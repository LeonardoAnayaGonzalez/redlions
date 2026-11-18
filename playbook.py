"""Playbook Creator: formaciones, legalidad, rutas, modelo defensivo y evaluación de jugadas hipotéticas.

Campo normalizado (el ataque avanza hacia +x), pasos de 0.1 s, unidades en yardas.

Ataque. Los receptores siguen sus puntos de ruta con velocidad y aceleración limitadas por los
percentiles altos de su grupo posicional. Antes de cada quiebre reducen la velocidad según el ángulo
del giro, de modo que puedan frenar con la aceleración disponible. El QB retrocede hasta 7 yd detrás de
la línea de golpeo; la línea ofensiva permanece en su sitio.

Defensa. Las posiciones iniciales son las reales en el snap de la jugada elegida. Después:
  - Presión (pff_role = Pass Rush): la distancia de cada rusher al QB se escala con la curva mediana
    de presión observada en el dataset (no se modelan bloqueos individuales).
  - Cobertura hombre (pff_passCoverageType = Man): cada defensor de cobertura se asigna al receptor
    más cercano en el snap y lo persigue tras 0.3 s, apuntando 0.3 s por delante de él.
  - Cobertura zona (Zone, Other o sin dato, y defensores sin asignar): cada defensor retrocede a un
    punto de zona y persigue al receptor más cercano que entre en un radio de 6 yd de ese punto. Los
    defensores profundos (alineados a 8 yd o más) vigilan un radio de 12 yd y se mantienen 2 yd por
    detrás del receptor que amenaza su zona.
Los movimientos futuros reales de los defensores no se usan.

Evaluación. Para cada receptor i y cada instante de lanzamiento t en la ventana W (percentiles 10-90
del tiempo hasta el lanzamiento observado):
    p_i(t)  = P(completar | separación, distancia al QB, profundidad, t, presión, velocidad)  (logística validada;
              cada variable se acota al rango P1-P99 del entrenamiento)
    Y_i(t)  = yardas esperadas si se completa = a + b·profundidad + c·separación         (OLS validada)
    EY_i(t) = p_i(t) · (Y_i(t) + B · 1[Y_i(t) ≥ yardas para primer down])
    EY_i    = max_{t ∈ W} EY_i(t)
    S       = w1·EY_(1) + w2·EY_(2) + w3·EY_(3)      (EY ordenados de mayor a menor)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import FIELD_LENGTH, FIELD_WIDTH, FPS, heading_to_velocity
from .loader import SIDE_DEF
from .play import PlayArrays
from .stats import sigmoid

DT = 1.0 / FPS
EDGE = 0.5
ON_LINE_DEPTH = 1.5          # a menos de 1.5 yd de la línea de golpeo = jugador en la línea
MIN_SPACING = 0.8            # yd mínimas entre dos jugadores en el snap
RECEIVER_SPACING = 1.0       # yd mínimas entre receptores durante las rutas
QB_DROP_DEPTH = 7.0
QB_DROP_SPEED = 4.5
MAN_DELAY_S = 0.3
MAN_LEAD_S = 0.3
ZONE_RADIUS = 6.0
DEEP_ZONE_RADIUS = 12.0      # radio de responsabilidad de un defensor profundo
DEEP_CUSHION = 2.0           # el defensor profundo se mantiene esta distancia por detrás del receptor
ZONE_DEEP_DEPTH = 12.0
ZONE_UNDER_DEPTH = 6.0
DEEP_ALIGN = 8.0
ARRIVE_S = 0.5
WAYPOINT_TOL = 1.0
MIN_CORNER = 0.2
EXTEND_YD = 12.0
MAX_YARDS = 60.0
DEFAULT_WEIGHTS = {"w1": 1.0, "w2": 0.5, "w3": 0.25, "fd_bonus": 2.0}
RECEIVER_POS = ("WR", "TE", "RB")
POS_GROUP = {"QB": "QB", "OL": "OL", "WR": "WR", "TE": "TE", "RB": "RB"}

_OL = [("LT", "OL", -0.8, 2.6), ("LG", "OL", -0.8, 1.3), ("C", "OL", -0.6, 0.0), ("RG", "OL", -0.8, -1.3), ("RT", "OL", -0.8, -2.6)]


def _f(label: str, desc: str, rows: list[tuple]) -> dict:
    return {"label": label, "description": desc,
            "players": [{"slot": s, "pos": p, "dx": dx, "dy": dy} for s, p, dx, dy in _OL + rows]}


# dx: yardas respecto a la línea de golpeo (negativo = detrás). dy: respecto al balón (+ = izquierda del ataque).
FORMATIONS = {
    "shotgun_2x2": _f("Shotgun 2x2 (11 personal)", "Dos receptores por lado; QB a 5 yd y corredor a su lado.",
                      [("QB", "QB", -5.0, 0.0), ("RB", "RB", -5.0, -1.6), ("X", "WR", -0.8, 20.0), ("H", "WR", -2.0, 11.0),
                       ("Y", "TE", -0.8, -4.2), ("Z", "WR", -2.0, -17.0)]),
    "shotgun_trips": _f("Shotgun Trips derecha (11 personal)", "Tres receptores a la derecha y uno aislado a la izquierda.",
                        [("QB", "QB", -5.0, 0.0), ("RB", "RB", -5.0, 1.6), ("X", "WR", -0.8, 19.0), ("H", "WR", -2.0, -8.0),
                         ("Y", "TE", -2.0, -12.0), ("Z", "WR", -0.8, -20.0)]),
    "empty_3x2": _f("Empty 3x2 (10 personal)", "Cinco receptores abiertos y el QB solo en el backfield.",
                    [("QB", "QB", -5.0, 0.0), ("X", "WR", -0.8, 20.0), ("H", "WR", -2.0, 11.0), ("Z", "WR", -0.8, -21.0),
                     ("Y", "TE", -2.0, -7.0), ("F", "WR", -2.0, -14.0)]),
    "ace_12": _f("Singleback Ace (12 personal)", "Dos alas cerradas en la línea, dos receptores abiertos y un corredor.",
                 [("QB", "QB", -1.2, 0.0), ("RB", "RB", -6.5, 0.0), ("Y", "TE", -0.8, 4.2), ("U", "TE", -0.8, -4.2),
                  ("X", "WR", -2.0, 17.0), ("Z", "WR", -2.0, -17.0)]),
    "i_form_21": _f("I-Form (21 personal)", "Fullback y corredor en fila detrás del QB bajo el centro.",
                    [("QB", "QB", -1.2, 0.0), ("FB", "RB", -4.0, 0.0), ("RB", "RB", -7.0, 0.0), ("X", "WR", -0.8, 18.0),
                     ("Y", "TE", -0.8, -4.2), ("Z", "WR", -2.0, -17.0)]),
    "pistol_2x2": _f("Pistol 2x2 (11 personal)", "QB a 4 yd y corredor detrás de él.",
                     [("QB", "QB", -4.0, 0.0), ("RB", "RB", -7.0, 0.0), ("X", "WR", -0.8, 19.0), ("H", "WR", -2.0, 10.0),
                      ("Y", "TE", -0.8, -4.2), ("Z", "WR", -2.0, -17.0)]),
}

# Puntos (profundidad, hacia afuera) relativos a la posición inicial. "settle" = se detiene al final.
ROUTE_TEMPLATES = {
    "go": {"label": "Go (vertical)", "points": [(18.0, 0.0)], "settle": False},
    "seam": {"label": "Seam", "points": [(8.0, 0.0), (22.0, -1.0)], "settle": False},
    "slant": {"label": "Slant", "points": [(2.0, 0.0), (8.0, -6.0)], "settle": False},
    "out": {"label": "Out", "points": [(10.0, 0.0), (10.0, 8.0)], "settle": True},
    "in": {"label": "In / Dig", "points": [(12.0, 0.0), (12.0, -10.0)], "settle": False},
    "post": {"label": "Post", "points": [(12.0, 0.0), (26.0, -8.0)], "settle": False},
    "corner": {"label": "Corner", "points": [(12.0, 0.0), (24.0, 8.0)], "settle": False},
    "hitch": {"label": "Hitch", "points": [(6.0, 0.0), (5.0, -0.5)], "settle": True},
    "curl": {"label": "Curl", "points": [(12.0, 0.0), (10.0, -1.5)], "settle": True},
    "comeback": {"label": "Comeback", "points": [(15.0, 0.0), (12.0, 3.0)], "settle": True},
    "flat": {"label": "Flat", "points": [(1.0, 4.0), (2.0, 10.0)], "settle": False},
    "drag": {"label": "Drag (cruce corto)", "points": [(2.0, 0.0), (4.0, -14.0)], "settle": False},
    "wheel": {"label": "Wheel", "points": [(1.0, 5.0), (5.0, 9.0), (22.0, 9.0)], "settle": False},
}


# ---------------------------------------------------------------------------- legalidad
def legality(players: list[dict], los_x: float, ball_y: float) -> dict:
    """Reglas básicas de una formación ofensiva legal en el snap."""
    checks: list[dict] = []

    def add(rule: str, ok: bool, detail: str) -> None:
        checks.append({"rule": rule, "ok": bool(ok), "detail": detail})

    n = len(players)
    pos = [p["pos"] for p in players]
    add("11 jugadores", n == 11, f"Hay {n} jugadores en el campo.")
    add("Un quarterback", pos.count("QB") == 1, f"Hay {pos.count('QB')} QB.")
    add("Cinco linieros ofensivos", pos.count("OL") == 5, f"Hay {pos.count('OL')} linieros.")
    out = [p["slot"] for p in players if not (EDGE <= p["x"] <= FIELD_LENGTH - EDGE and EDGE <= p["y"] <= FIELD_WIDTH - EDGE)]
    add("Dentro del campo", not out, "Todos dentro del campo." if not out else f"Fuera del campo: {', '.join(out)}.")
    beyond = [p["slot"] for p in players if p["x"] >= los_x]
    add("Nadie más allá de la línea de golpeo", not beyond,
        "Nadie cruza la línea." if not beyond else f"Adelantados (fuera de lugar): {', '.join(beyond)}.")
    line = [p for p in players if p["pos"] != "QB" and p["x"] >= los_x - ON_LINE_DEPTH and p["x"] < los_x]
    back = [p for p in players if p not in line]
    add("Al menos 7 jugadores en la línea", len(line) >= 7, f"{len(line)} en la línea (a menos de {ON_LINE_DEPTH:g} yd).")
    add("Máximo 4 en el backfield", len(back) <= 4, f"{len(back)} en el backfield.")
    center = [p for p in players if p["pos"] == "OL" and abs(p["y"] - ball_y) <= 1.0 and p in line]
    add("Centro sobre el balón", bool(center), "Hay un liniero sobre el balón." if center else "Ningún liniero en la línea sobre el balón.")
    qb = [p for p in players if p["pos"] == "QB"]
    add("QB detrás del centro", bool(qb) and qb[0]["x"] <= los_x - 1.0,
        "QB al menos 1 yd detrás de la línea." if qb and qb[0]["x"] <= los_x - 1.0 else "El QB debe estar al menos 1 yd detrás de la línea.")
    close = []
    snapper = {p["slot"] for p in center}
    for i in range(n):
        for j in range(i + 1, n):
            pair = {players[i]["pos"], players[j]["pos"]}
            if pair == {"QB", "OL"} and (players[i]["slot"] in snapper or players[j]["slot"] in snapper):
                continue      # el QB bajo el centro recibe el balón en contacto con él
            if np.hypot(players[i]["x"] - players[j]["x"], players[i]["y"] - players[j]["y"]) < MIN_SPACING:
                close.append(f"{players[i]['slot']}-{players[j]['slot']}")
    add("Separación entre jugadores", not close, "Ningún par a menos de 0.8 yd." if not close else f"Demasiado juntos: {', '.join(close)}.")
    ends = set()
    if line:
        ends = {min(line, key=lambda p: p["y"])["slot"], max(line, key=lambda p: p["y"])["slot"]}
    eligible = [p["slot"] for p in players if p["pos"] in RECEIVER_POS and (p["slot"] in ends or p in back)]
    runners = [p for p in players if p.get("route")]
    bad = [p["slot"] for p in runners if p["slot"] not in eligible]
    add("Solo receptores elegibles corren rutas", not bad,
        "Todas las rutas son de jugadores elegibles (extremos de la línea o backfield)." if not bad
        else f"No elegibles con ruta: {', '.join(bad)} (linieros, QB o jugadores tapados en la línea).")
    add("Al menos una ruta", bool(runners), f"{len(runners)} jugadores con ruta.")
    return {"legal": all(c["ok"] for c in checks), "checks": checks, "eligible": eligible,
            "on_line": [p["slot"] for p in line], "backfield": [p["slot"] for p in back]}


# ---------------------------------------------------------------------------- rutas
def clip_point(x: float, y: float) -> tuple[float, float]:
    m = EDGE + 0.5
    return float(np.clip(x, m, FIELD_LENGTH - m)), float(np.clip(y, m, FIELD_WIDTH - m))


def template_waypoints(name: str, start: tuple[float, float], ball_y: float) -> tuple[list[tuple[float, float]], bool]:
    tpl = ROUTE_TEMPLATES[name]
    side = 1.0 if start[1] >= ball_y else -1.0      # +1: el exterior está hacia +y
    return [clip_point(start[0] + d, start[1] + side * o) for d, o in tpl["points"]], tpl["settle"]


def follow_route(p0: tuple[float, float], waypoints: list, v_max: float, a_max: float, steps: int,
                 settle: bool) -> tuple[np.ndarray, np.ndarray, float | None, bool]:
    """Trayectoria (steps+1, 2), velocidades, tiempo para completar la ruta y si permanece en el campo."""
    p = np.array(p0, dtype=np.float64)
    v = np.zeros(2)
    pos = np.empty((steps + 1, 2))
    vel = np.empty((steps + 1, 2))
    pos[0], vel[0] = p, v
    wps = [np.array(w, dtype=np.float64) for w in waypoints]
    n_real = len(wps)
    if wps and not settle:          # tras el último punto sigue en la misma dirección y luego se detiene
        prev = wps[-2] if len(wps) > 1 else p
        d = wps[-1] - prev
        if np.hypot(*d) > 1e-6:
            wps.append(np.array(clip_point(*(wps[-1] + d / np.hypot(*d) * EXTEND_YD))))
    k, done = 0, None
    dv_max = a_max * DT
    for t in range(steps):
        while k < len(wps) and np.hypot(*(wps[k] - p)) < WAYPOINT_TOL:
            k += 1
            if k == n_real and done is None:
                done = t * DT
        if k < len(wps):
            to = wps[k] - p
            dist = float(np.hypot(*to))
            if k + 1 < len(wps):
                a = wps[k] - (wps[k - 1] if k > 0 else pos[0])
                b = wps[k + 1] - wps[k]
                na, nb = np.hypot(*a), np.hypot(*b)
                cos = float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else 1.0
                v_corner = v_max * max(MIN_CORNER, (1.0 + cos) / 2.0)
            else:
                v_corner = 0.0
            v_des = min(v_max, float(np.sqrt(v_corner ** 2 + 2.0 * a_max * max(dist - WAYPOINT_TOL, 0.0))))
            desired = to / max(dist, 1e-9) * v_des
        else:
            desired = np.zeros(2)
        dv = desired - v
        n = float(np.hypot(*dv))
        if n > dv_max:
            dv *= dv_max / n
        v = v + dv
        sp = float(np.hypot(*v))
        if sp > v_max:
            v *= v_max / sp
        p = p + v * DT
        pos[t + 1], vel[t + 1] = p, v
    if done is None and n_real and k >= n_real:
        done = steps * DT
    inside = bool(((pos[:, 0] >= EDGE) & (pos[:, 0] <= FIELD_LENGTH - EDGE) & (pos[:, 1] >= EDGE) &
                   (pos[:, 1] <= FIELD_WIDTH - EDGE)).all())
    return pos, vel, done, inside


def qb_path(p0: tuple[float, float], los_x: float, steps: int) -> np.ndarray:
    target = min(p0[0], los_x - QB_DROP_DEPTH)
    t = np.arange(steps + 1) * DT
    return np.stack([np.maximum(target, p0[0] - QB_DROP_SPEED * t), np.full(steps + 1, p0[1])], axis=1)


# ---------------------------------------------------------------------------- defensa
@dataclass
class Defense:
    pos0: np.ndarray            # (J, 2) posiciones reales en el snap
    rush: np.ndarray            # (J,) bool
    v_max: np.ndarray
    a_max: np.ndarray
    man: bool                   # cobertura hombre según PFF
    los_x: float
    ids: list[int] = field(default_factory=list)


def man_assignment(d: Defense, rec0: np.ndarray) -> np.ndarray:
    """Receptor asignado (índice) a cada defensor de cobertura, o -1 (zona). Emparejamiento voraz por distancia."""
    assign = np.full(d.pos0.shape[0], -1)
    if not d.man or rec0.size == 0:
        return assign
    cover = np.flatnonzero(~d.rush)
    dist = np.hypot(d.pos0[cover, None, 0] - rec0[None, :, 0], d.pos0[cover, None, 1] - rec0[None, :, 1])
    used_d, used_r = set(), set()
    for flat in np.argsort(dist, axis=None):
        i, r = np.unravel_index(flat, dist.shape)
        if i in used_d or r in used_r:
            continue
        assign[cover[i]] = r
        used_d.add(i)
        used_r.add(r)
    return assign


def zone_anchors(d: Defense) -> np.ndarray:
    depth0 = d.pos0[:, 0] - d.los_x
    target = np.where(depth0 >= DEEP_ALIGN, ZONE_DEEP_DEPTH, ZONE_UNDER_DEPTH)
    return np.stack([d.los_x + np.maximum(depth0, target), d.pos0[:, 1]], axis=1)


def simulate_defense(d: Defense, rec_pos: np.ndarray, rec_vel: np.ndarray, qb: np.ndarray, rush_curve: np.ndarray,
                     assign: np.ndarray) -> np.ndarray:
    """Defensores (N, J, T+1, 2) frente a N variantes del ataque. rec_pos/rec_vel: (N, R, T+1, 2)."""
    N, R, T1, _ = rec_pos.shape
    J = d.pos0.shape[0]
    out = np.empty((N, J, T1, 2))
    out[:, :, 0] = d.pos0
    curve = rush_curve[:T1] if rush_curve.size >= T1 else np.pad(rush_curve, (0, T1 - rush_curve.size), mode="edge")
    for j in np.flatnonzero(d.rush):
        u = d.pos0[j] - qb[0]
        D0 = float(np.hypot(*u))
        u = u / max(D0, 1e-9)
        D = np.maximum(0.5, D0 * curve / max(curve[0], 1e-9))
        out[:, j] = qb + u[None, :] * D[:, None]
    cover = np.flatnonzero(~d.rush)
    if cover.size == 0:
        return out
    pos = np.broadcast_to(d.pos0[cover], (N, cover.size, 2)).copy()
    vel = np.zeros_like(pos)
    anchors = zone_anchors(d)[cover]
    deep = (d.pos0[cover, 0] - d.los_x) >= DEEP_ALIGN
    radius = np.where(deep, DEEP_ZONE_RADIUS, ZONE_RADIUS)
    tgt_r = assign[cover]
    is_man = tgt_r >= 0
    zm = ~is_man
    vm, am = d.v_max[cover][None, :], d.a_max[cover][None, :]
    delay = int(round(MAN_DELAY_S * FPS))
    for t in range(T1 - 1):
        tgt = np.broadcast_to(anchors, (N, cover.size, 2)).copy()
        if is_man.any():
            ri = tgt_r[is_man]
            lead = rec_pos[:, ri, t] + rec_vel[:, ri, t] * MAN_LEAD_S
            tgt[:, is_man] = lead if t >= delay else pos[:, is_man]
        if zm.any() and R:
            rp = rec_pos[:, :, t]                                                 # (N, R, 2)
            az = anchors[zm]                                                     # (Kz, 2)
            dz = np.hypot(rp[:, None, :, 0] - az[None, :, None, 0], rp[:, None, :, 1] - az[None, :, None, 1])
            near = dz.argmin(axis=2)                                             # (N, Kz)
            within = dz.min(axis=2) <= radius[zm][None, :]
            chosen = np.take_along_axis(rp, near[..., None].repeat(2, axis=-1), axis=1)
            dz_deep = deep[zm][None, :]
            cushion_x = np.maximum(chosen[..., 0] + DEEP_CUSHION, az[None, :, 0])
            chosen = np.where(dz_deep[..., None], np.stack([cushion_x, chosen[..., 1]], axis=-1), chosen)
            tgt[:, zm] = np.where(within[..., None], chosen, tgt[:, zm])
        to = tgt - pos
        dist = np.hypot(to[..., 0], to[..., 1])
        v_des = np.minimum(vm, dist / ARRIVE_S)
        desired = to / np.maximum(dist, 1e-9)[..., None] * v_des[..., None]
        dv = desired - vel
        n = np.hypot(dv[..., 0], dv[..., 1])
        vel = vel + dv * np.minimum(1.0, am * DT / np.maximum(n, 1e-12))[..., None]
        sp = np.hypot(vel[..., 0], vel[..., 1])
        vel = vel * np.minimum(1.0, vm / np.maximum(sp, 1e-12))[..., None]
        pos = pos + vel * DT
        out[:, cover, t + 1] = pos
    return out


def separation(rec_pos: np.ndarray, defs: np.ndarray) -> np.ndarray:
    """(N, R, T+1): distancia de cada receptor al defensor más cercano."""
    d = np.hypot(rec_pos[:, :, None, :, 0] - defs[:, None, :, :, 0], rec_pos[:, :, None, :, 1] - defs[:, None, :, :, 1])
    return d.min(axis=2)


# ---------------------------------------------------------------------------- evaluación
@dataclass
class Models:
    completion: dict              # parámetros de la logística (store.completion)
    yards: dict                   # {"coef": [a, b_depth, c_sep]}
    rush_curve: np.ndarray        # distancia mediana del rusher más cercano al QB por frame desde el snap
    window: tuple[float, float]   # ventana de lanzamiento (s)


def clip_to_training(comp: dict, name: str, v: np.ndarray) -> np.ndarray:
    """Acota una variable al rango P1-P99 visto al entrenar, para no extrapolar el modelo."""
    lo, hi = (comp.get("ranges") or {}).get(name, (-np.inf, np.inf))
    return np.clip(v, lo, hi)


def completion_prob(comp: dict, feats: dict[str, np.ndarray]) -> np.ndarray:
    cols = []
    for name in comp["features"]:
        v = np.asarray(feats[name], dtype=np.float64)
        v = clip_to_training(comp, name, np.where(np.isfinite(v), v, comp["medians"][name]))
        cols.append(np.clip(v, -5, 15) if name == "rush_min" else v)
    X = np.stack(cols, axis=-1)
    Z = (X - np.array(comp["mu"])) / np.array(comp["sd"])
    beta = np.array(comp["beta"])
    return sigmoid(Z @ beta[:-1] + beta[-1])


def window_idx(window: tuple[float, float], T1: int) -> np.ndarray:
    return np.arange(int(round(window[0] * FPS)), min(int(round(window[1] * FPS)), T1 - 1) + 1)


def evaluate_batch(rec_pos: np.ndarray, rec_vel: np.ndarray, qb: np.ndarray, d: Defense, assign: np.ndarray,
                   m: Models, weights: dict, los_x: float, ytg: float) -> dict:
    defs = simulate_defense(d, rec_pos, rec_vel, qb, m.rush_curve, assign)
    sep = separation(rec_pos, defs)
    T1 = rec_pos.shape[2]
    W = window_idx(m.window, T1)
    rp, rv = rec_pos[:, :, W], rec_vel[:, :, W]
    depth = rp[..., 0] - los_x
    feats = {"sep": sep[:, :, W], "dist_qb": np.hypot(rp[..., 0] - qb[W, 0], rp[..., 1] - qb[W, 1]), "depth": depth,
             "time_to_throw": np.broadcast_to(W * DT, depth.shape), "rush_min": np.broadcast_to(m.rush_curve[W], depth.shape),
             "speed": np.hypot(rv[..., 0], rv[..., 1])}
    p = completion_prob(m.completion, feats)
    a, b, c = m.yards["coef"]
    yhat = np.clip(a + b * clip_to_training(m.completion, "depth", depth)
                   + c * clip_to_training(m.completion, "sep", feats["sep"]), 0.0, MAX_YARDS)
    ev = p * (yhat + weights["fd_bonus"] * (yhat >= ytg))
    best = ev.max(axis=2)
    arg = ev.argmax(axis=2)
    srt = -np.sort(-best, axis=1)
    w = [weights["w1"], weights["w2"], weights["w3"]]
    S = sum(w[k] * srt[:, k] for k in range(min(3, srt.shape[1])))
    R = rec_pos.shape[1]
    if R > 1:
        dd = np.hypot(rec_pos[:, :, None, 1:, 0] - rec_pos[:, None, :, 1:, 0], rec_pos[:, :, None, 1:, 1] - rec_pos[:, None, :, 1:, 1])
        iu = np.triu_indices(R, 1)
        collision = dd[:, iu[0], iu[1]].min(axis=(1, 2)) < RECEIVER_SPACING
    else:
        collision = np.zeros(rec_pos.shape[0], dtype=bool)
    take = lambda arr: np.take_along_axis(arr, arg[..., None], axis=2)[..., 0]  # noqa: E731
    return {"S": S, "best": best, "t_star": W[arg] * DT, "p": take(p), "yhat": take(yhat), "sep_star": take(feats["sep"]),
            "depth_star": take(depth), "ev": ev, "p_all": p, "sep": sep, "defs": defs, "collision": collision, "W": W}


# ---------------------------------------------------------------------------- datos de calibración
CURVE_FRAMES = 61


def _roles_idx(pa: PlayArrays, roles: dict, role: str) -> list[int]:
    return [i for i in range(pa.ids.size) if roles.get(int(pa.ids[i]), ("",))[0] == role]


class RushCurve:
    """Distancia del rusher más cercano al QB desde el snap hasta el lanzamiento, frame a frame."""

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []

    def add(self, pa: PlayArrays, roles: dict) -> None:
        si = pa.idx(pa.snap_frame)
        qb, rush = _roles_idx(pa, roles, "Pass"), _roles_idx(pa, roles, "Pass Rush")
        if si is None or not qb or not rush:
            return
        ti = pa.idx(pa.throw_frame)
        end = min(pa.n_frames - 1, ti if ti is not None else pa.n_frames - 1, si + CURVE_FRAMES - 1)
        q = qb[0]
        d = np.hypot(pa.x[rush, si:end + 1] - pa.x[q, si:end + 1], pa.y[rush, si:end + 1] - pa.y[q, si:end + 1]).min(axis=0)
        row = np.full(CURVE_FRAMES, np.nan)
        row[:d.size] = d
        self.rows.append(row)

    def curve(self, min_n: int = 100) -> tuple[np.ndarray, np.ndarray]:
        a = np.array(self.rows)
        n = np.isfinite(a).sum(axis=0)
        med = np.where(n >= min_n, np.nanmedian(np.where(n >= 1, a, 0.0), axis=0), np.nan)
        for k in range(1, med.size):          # tramo final con poca muestra: se mantiene el último valor fiable
            if not np.isfinite(med[k]):
                med[k] = med[k - 1]
        return med, n


class CoverageValidation:
    """Con receptores y QB siguiendo sus trayectorias reales, compara la separación prevista al lanzar
    contra la real, para el modelo de cobertura y para una referencia de defensores quietos."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float]] = []

    def add(self, pa: PlayArrays, roles: dict, coverage_type: str | None, groups_lim: tuple[np.ndarray, np.ndarray],
            rush_curve: np.ndarray, los_x: float) -> None:
        si, ti = pa.idx(pa.snap_frame), pa.idx(pa.throw_frame)
        if si is None or ti is None or ti - si < 5:
            return
        routes, qb = _roles_idx(pa, roles, "Pass Route"), _roles_idx(pa, roles, "Pass")
        ent = np.flatnonzero(pa.side == SIDE_DEF)
        if not routes or not qb or ent.size == 0:
            return
        T = min(ti - si, CURVE_FRAMES - 1)
        sl = slice(si, si + T + 1)
        d = Defense(pos0=np.stack([pa.x[ent, si], pa.y[ent, si]], axis=1).astype(np.float64),
                    rush=np.array([roles.get(int(pa.ids[e]), ("",))[0] == "Pass Rush" for e in ent]),
                    v_max=groups_lim[0], a_max=groups_lim[1], man=coverage_type == "Man", los_x=los_x)
        rec = np.stack([pa.x[routes, sl], pa.y[routes, sl]], axis=-1).astype(np.float64)[None]
        vx, vy = heading_to_velocity(pa.s[routes, sl].astype(np.float64), pa.dir[routes, sl].astype(np.float64))
        rvel = np.stack([vx, vy], axis=-1)[None]
        qbp = np.stack([pa.x[qb[0], sl], pa.y[qb[0], sl]], axis=-1).astype(np.float64)
        defs = simulate_defense(d, rec, rvel, qbp, rush_curve, man_assignment(d, rec[0, :, 0]))
        real = np.stack([pa.x[ent, sl], pa.y[ent, sl]], axis=-1).astype(np.float64)[None]
        static = np.broadcast_to(d.pos0[None, :, None, :], real.shape)
        pred, true, stat = (separation(rec, x)[0, :, -1] for x in (defs, real, static))
        cov = coverage_type if coverage_type in ("Man", "Zone") else "Otra"
        for k in range(len(routes)):
            self.rows.append((cov, float(pred[k] - true[k]), float(stat[k] - true[k])))

    def report(self) -> list[dict]:
        if not self.rows:
            return []
        out = []
        groups = {"Todas": self.rows}
        for c in ("Man", "Zone", "Otra"):
            r = [x for x in self.rows if x[0] == c]
            if r:
                groups[c] = r
        for name, r in groups.items():
            a = np.array([(x[1], x[2]) for x in r])
            out.append({"coverage": name, "n": int(len(a)), "mae": round(float(np.abs(a[:, 0]).mean()), 3),
                        "bias": round(float(a[:, 0].mean()), 3), "mae_static": round(float(np.abs(a[:, 1]).mean()), 3),
                        "bias_static": round(float(a[:, 1].mean()), 3)})
        return out


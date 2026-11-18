"""Route Optimizer: rutas candidatas, modelo defensivo explícito y función objetivo.

Todo ocurre en el campo normalizado (el ataque avanza hacia +x) y en pasos de 0.1 s.

Receptor. Cada ruta candidata es un plan de dirección deseada por paso y una velocidad
objetivo. El movimiento se integra con un límite vectorial de aceleración:
    v(t+dt) = proyección_{|v| <= v_max}( v(t) + recorte_{|Δv| <= a_max·dt}( v_deseada - v(t) ) )
    p(t+dt) = p(t) + v(t+dt)·dt
Así los cambios de dirección son suaves y la velocidad y la aceleración nunca superan los
límites observados en el dataset. Las rutas que salen del campo se descartan; no se recortan.

Defensores. Dos modelos explícitos, ambos a partir solo del estado observado en el instante t0:
  - Inercia: el modelo de movimiento calibrado (ridge) proyecta a cada defensor.
  - Reactivo: además, los N_REACTIVE defensores de cobertura más cercanos persiguen al
    receptor tras un retraso de reacción, con sus propios límites de velocidad y aceleración.
Las posiciones futuras reales de los defensores nunca intervienen en la selección.

Función objetivo (separación útil):
    sep(t) = min_j || r(t) - d_j(t) ||
    J = media_{t ∈ W} min(sep(t), cap),   W = [max(0.5, H - 1), H]
`cap` es el percentil 95 de la separación observada al lanzar (más separación no cambia
la situación de forma práctica y evitaría premiar alejarse a zonas vacías). Desempates:
mayor separación mínima durante el recorrido y menor tiempo para abrirse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import FIELD_LENGTH, FIELD_WIDTH, FPS, heading_to_velocity
from .loader import SIDE_BALL, SIDE_DEF
from .motion import GROUPS, HORIZONS, LAG, MotionModel, kinematic_features
from .play import PlayArrays

DT = 1.0 / FPS
EDGE_MARGIN = 0.5                 # yardas mínimas a las líneas laterales y de fondo
REACTION_DELAY_S = 0.4            # supuesto explícito del modelo reactivo
N_REACTIVE = 2
HEADING_STEP = 15                 # grados entre direcciones candidatas
BREAK_TIMES = (0.0, 0.5, 1.0)     # instante del cambio de dirección (s)
CURVE_RATES = (-90.0, -45.0, 45.0, 90.0)   # giro constante (grados/s); + = hacia la derecha
ROUTE_HORIZONS = (1.0, 1.5, 2.0, 2.5, 3.0)
WINDOW_S = 1.0
MIN_PACE = 2.0                    # yd/s mínimas para la variante "ritmo actual"
CONTACT_YD = 1.0                  # separación mínima exigida durante el recorrido (sin contacto)
DIVERSITY_YD = 2.0                # dos rutas con finales más cercanos se consideran la misma alternativa
SPEED_Q, ACC_Q = 0.999, 0.99
SPEED_BIN, SPEED_TOP = 0.05, 15.0
ACC_BIN, ACC_TOP = 0.1, 40.0
DEFENSE_MODELS = ("reactive", "inertial")


# ---------------------------------------------------------------------------- límites físicos
class LimitAccumulator:
    """Histogramas por grupo posicional de la velocidad y de la aceleración vectorial |Δv|/Δt."""

    def __init__(self) -> None:
        G = len(GROUPS)
        self.speed = np.zeros((G, int(SPEED_TOP / SPEED_BIN) + 1))
        self.acc = np.zeros((G, int(ACC_TOP / ACC_BIN) + 1))

    def add(self, pa: PlayArrays, groups: np.ndarray, start: int) -> None:
        ent = np.flatnonzero((pa.side != SIDE_BALL) & (groups >= 0))
        if ent.size == 0 or pa.n_frames - start < 2:
            return
        s = pa.s[ent, start:].astype(np.float64)
        vx, vy = heading_to_velocity(s, pa.dir[ent, start:].astype(np.float64))
        acc = np.hypot(np.diff(vx, axis=1), np.diff(vy, axis=1)) * FPS
        moving = (s[:, 1:] >= 1.0) & (s[:, :-1] >= 1.0)   # la dirección es poco fiable casi detenido
        g = groups[ent]
        for gi in np.unique(g):
            m = g == gi
            sv = s[m].ravel()
            sv = sv[np.isfinite(sv)]
            self.speed[gi] += np.bincount(np.minimum((sv / SPEED_BIN).astype(int), self.speed.shape[1] - 1),
                                          minlength=self.speed.shape[1])
            av = acc[m][moving[m]]
            av = av[np.isfinite(av)]
            self.acc[gi] += np.bincount(np.minimum((av / ACC_BIN).astype(int), self.acc.shape[1] - 1),
                                        minlength=self.acc.shape[1])

    @staticmethod
    def _q(hist: np.ndarray, q: float, width: float) -> float:
        total = hist.sum()
        if total == 0:
            return float("nan")
        i = int(np.searchsorted(np.cumsum(hist), q * total))
        return float((i + 1) * width)   # borde superior: el límite nunca queda por debajo del dato

    def limits(self) -> dict[str, dict]:
        out = {}
        glob_s, glob_a = self.speed.sum(axis=0), self.acc.sum(axis=0)
        for gi, g in enumerate(GROUPS):
            hs, ha = self.speed[gi], self.acc[gi]
            if hs.sum() < 1000 or ha.sum() < 1000:
                hs, ha = glob_s, glob_a
            out[g] = {"v_max": round(self._q(hs, SPEED_Q, SPEED_BIN), 3), "a_max": round(self._q(ha, ACC_Q, ACC_BIN), 3),
                      "n_speed": int(self.speed[gi].sum()), "n_acc": int(self.acc[gi].sum())}
        return out


# ---------------------------------------------------------------------------- receptor
def simulate(p0: np.ndarray, v0: np.ndarray, headings: np.ndarray, target_speed: np.ndarray,
             v_max: float, a_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Integra N planes. headings (N, T) en grados; target_speed (N,). Devuelve pos y vel (N, T+1, 2)."""
    N, T = headings.shape
    v_cap = max(float(v_max), float(np.hypot(*v0)))   # nunca se fuerza un frenazo instantáneo
    dv_max = float(a_max) * DT
    pos = np.empty((N, T + 1, 2))
    vel = np.empty((N, T + 1, 2))
    pos[:, 0], vel[:, 0] = p0, v0
    sp = np.minimum(target_speed, v_cap)[:, None]
    tx, ty = heading_to_velocity(sp * np.ones((1, T)), headings)
    for t in range(T):
        dv = np.stack([tx[:, t], ty[:, t]], axis=1) - vel[:, t]
        n = np.hypot(dv[:, 0], dv[:, 1])
        v = vel[:, t] + dv * np.minimum(1.0, dv_max / np.maximum(n, 1e-12))[:, None]
        speed = np.hypot(v[:, 0], v[:, 1])
        v *= np.minimum(1.0, v_cap / np.maximum(speed, 1e-12))[:, None]   # proyección: no aumenta |Δv|
        vel[:, t + 1] = v
        pos[:, t + 1] = pos[:, t] + v * DT
    return pos, vel


SECTORS = {0: "hacia la banda izquierda", 45: "adelante-izquierda", 90: "hacia adelante",
           135: "adelante-derecha", 180: "hacia la banda derecha", 225: "atrás-derecha",
           270: "hacia atrás", 315: "atrás-izquierda"}


def heading_name(theta: float) -> str:
    """Nombre de una dirección en el campo normalizado (90° = hacia la zona rival, 0° = izquierda del ataque)."""
    return SECTORS[int(round((theta % 360) / 45.0)) % 8 * 45]


@dataclass
class Plans:
    headings: np.ndarray            # (N, T)
    speeds: np.ndarray              # (N,)
    meta: list[dict] = field(default_factory=list)


def build_plans(dir0: float, s0: float, v_max: float, steps: int) -> Plans:
    """Familias: mantener dirección, recta/diagonal inmediata, quiebre tardío y curva de giro constante."""
    t = np.arange(steps) * DT
    horizon = steps * DT
    heads, speeds, meta = [], [], []
    paces = (("actual", float(np.clip(s0, MIN_PACE, v_max)), "ritmo actual"), ("max", float(v_max), "máxima velocidad plausible"))
    for pace, sp, pace_label in paces:
        heads.append(np.full(steps, dir0))
        speeds.append(sp)
        meta.append({"kind": "mantener", "heading": round(dir0 % 360, 1), "break_s": None, "rate": None, "pace": pace,
                     "label": f"Mantener la dirección actual ({dir0 % 360:.0f}°) · {pace_label}"})
        for theta in range(0, 360, HEADING_STEP):
            for tb in BREAK_TIMES:
                if tb > 0 and tb > horizon - 0.5 + 1e-9:
                    continue   # el quiebre debe dejar al menos 0.5 s de recorrido
                heads.append(np.where(t < tb - 1e-9, dir0, float(theta)))
                speeds.append(sp)
                if tb == 0:
                    kind = "recta" if theta % 90 == 0 else "diagonal"
                    label = f"{kind.capitalize()} {heading_name(theta)} ({theta}°) · {pace_label}"
                else:
                    kind = "quiebre"
                    label = f"Quiebre a {tb:g} s: {heading_name(theta)} ({theta}°) · {pace_label}"
                meta.append({"kind": kind, "heading": float(theta), "break_s": tb, "rate": None, "pace": pace, "label": label})
        for rate in CURVE_RATES:
            heads.append(dir0 + rate * t)
            speeds.append(sp)
            side = "derecha" if rate > 0 else "izquierda"
            meta.append({"kind": "curva", "heading": round((dir0 + rate * horizon) % 360, 1), "break_s": None, "rate": rate,
                         "pace": pace, "label": f"Curva suave hacia la {side} ({abs(rate):g}°/s) · {pace_label}"})
    return Plans(np.array(heads, dtype=np.float64), np.array(speeds, dtype=np.float64), meta)


def feasibility(pos: np.ndarray, los_x: float, depth0: float) -> tuple[np.ndarray, np.ndarray]:
    """(dentro del campo, profundidad válida). La ruta no puede terminar más atrás de la línea de golpeo
    que el punto de partida del receptor (ni detrás de ella si empezó por delante)."""
    x, y = pos[:, 1:, 0], pos[:, 1:, 1]
    in_field = ((x >= EDGE_MARGIN) & (x <= FIELD_LENGTH - EDGE_MARGIN) &
                (y >= EDGE_MARGIN) & (y <= FIELD_WIDTH - EDGE_MARGIN)).all(axis=1)
    depth_ok = (pos[:, -1, 0] - los_x) >= min(depth0, 0.0) - 1e-9
    return in_field, depth_ok


# ---------------------------------------------------------------------------- defensores
def inertial_paths(model: MotionModel, pa: PlayArrays, fi: int, ent: np.ndarray, groups: np.ndarray,
                   steps: int, feats_all: np.ndarray | None = None) -> np.ndarray:
    """Trayectorias (J, steps+1, 2) con el modelo calibrado, interpoladas a 0.1 s entre sus horizontes."""
    feats_all = kinematic_features(pa) if feats_all is None else feats_all
    feats = feats_all[ent, fi].astype(np.float64)
    ks = [k for k in HORIZONS if k < steps + HORIZONS[0]]
    disp = np.zeros((ent.size, len(ks) + 1, 2))
    for g in np.unique(groups):
        m = groups == g
        for i, k in enumerate(ks):
            disp[m, i + 1] = model.predict_disp(int(g), k, feats[m])
    grid = np.array([0] + ks, dtype=np.float64)
    tt = np.arange(steps + 1, dtype=np.float64)
    out = np.empty((ent.size, steps + 1, 2))
    x0, y0 = pa.x[ent, fi].astype(np.float64), pa.y[ent, fi].astype(np.float64)
    for j in range(ent.size):
        out[j, :, 0] = x0[j] + np.interp(tt, grid, disp[j, :, 0])
        out[j, :, 1] = y0[j] + np.interp(tt, grid, disp[j, :, 1])
    return out


def choose_reactive(pa: PlayArrays, fi: int, rec: int, ent: np.ndarray, roles: dict) -> np.ndarray:
    """Índices (dentro de `ent`) de los defensores de cobertura más cercanos al receptor en t0."""
    d = np.hypot(pa.x[ent, fi] - pa.x[rec, fi], pa.y[ent, fi] - pa.y[rec, fi])
    cover = np.array([roles.get(int(pa.ids[e]), ("",))[0] == "Coverage" for e in ent])
    pool = np.flatnonzero(cover) if cover.any() else np.arange(ent.size)
    return pool[np.argsort(d[pool])][:N_REACTIVE]


def reactive_paths(base: np.ndarray, v0: np.ndarray, reactive: np.ndarray, rec: np.ndarray,
                   v_max: np.ndarray, a_max: np.ndarray, delay_steps: int) -> np.ndarray:
    """Defensores (N, J, T+1, 2). Los reactivos siguen la inercia hasta el retraso y luego persiguen la
    posición actual del receptor con aceleración y velocidad limitadas."""
    N, J, T1 = rec.shape[0], base.shape[0], base.shape[1]
    out = np.broadcast_to(base, (N, J, T1, 2)).copy()
    T = T1 - 1
    if reactive.size == 0 or delay_steps >= T:
        return out
    r, d = reactive, delay_steps
    pos = out[:, r, d].copy()
    vel0 = (base[r, d] - base[r, d - 1]) / DT if d >= 1 else v0[r]
    vel = np.broadcast_to(vel0, (N, r.size, 2)).copy()
    vm, am = v_max[r][None, :], a_max[r][None, :]
    v_cap = np.maximum(vm, np.hypot(vel[..., 0], vel[..., 1]))
    for t in range(d, T):
        to = rec[:, t, None, :] - pos
        dist = np.hypot(to[..., 0], to[..., 1])
        desired = to / np.maximum(dist, 1e-9)[..., None] * vm[..., None]
        dv = desired - vel
        n = np.hypot(dv[..., 0], dv[..., 1])
        vel = vel + dv * np.minimum(1.0, am * DT / np.maximum(n, 1e-12))[..., None]
        speed = np.hypot(vel[..., 0], vel[..., 1])
        vel = vel * np.minimum(1.0, v_cap / np.maximum(speed, 1e-12))[..., None]
        pos = pos + vel * DT
        out[:, r, t + 1] = pos
    return out


def separation(rec: np.ndarray, defs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rec (N, T+1, 2), defs (N, J, T+1, 2) -> distancia al defensor más cercano (N, T+1) y su índice."""
    d = np.hypot(rec[:, None, :, 0] - defs[..., 0], rec[:, None, :, 1] - defs[..., 1])
    return d.min(axis=1), d.argmin(axis=1)


# ---------------------------------------------------------------------------- función objetivo
def window_start(horizon_s: float) -> float:
    return max(0.5, horizon_s - WINDOW_S)


def score(sep: np.ndarray, horizon_s: float, cap: float, open_thr: float) -> dict[str, np.ndarray]:
    t = np.arange(sep.shape[1]) * DT
    w = t >= window_start(horizon_s) - 1e-9
    reach = sep >= open_thr
    t_open = np.where(reach.any(axis=1), t[np.argmax(reach, axis=1)], np.nan)
    return {"J": np.minimum(sep[:, w], cap).mean(axis=1), "window_mean": sep[:, w].mean(axis=1),
            "s_min": sep[:, 1:].min(axis=1), "s_end": sep[:, -1], "s_max": sep.max(axis=1),
            "t_open": t_open, "t_max": t[np.argmax(sep, axis=1)]}


def contact_ok(sep: np.ndarray) -> np.ndarray:
    """Sin contacto: la ruta no se acerca a menos de CONTACT_YD de un defensor, o de la separación
    inicial si el receptor ya empezó más cerca."""
    return sep[:, 1:].min(axis=1) >= np.minimum(CONTACT_YD, sep[:, 0]) - 1e-9


def rank(m: dict[str, np.ndarray], ok: np.ndarray | None = None) -> np.ndarray:
    """Primero las rutas sin contacto; dentro de cada grupo, J (desc), separación mínima (desc) y
    tiempo para abrirse (asc)."""
    t_open = np.where(np.isnan(m["t_open"]), np.inf, m["t_open"])
    ok = np.ones(m["J"].shape[0], dtype=bool) if ok is None else ok
    return np.lexsort((t_open, -m["s_min"], -m["J"], ~ok))


def diverse_top(order: np.ndarray, pos: np.ndarray, k: int = 3) -> list[int]:
    picked: list[int] = []
    for i in order:
        if all(np.hypot(*(pos[i, -1] - pos[j, -1])) >= DIVERSITY_YD for j in picked):
            picked.append(int(i))
        if len(picked) == k:
            break
    return picked


# ---------------------------------------------------------------------------- orquestación
@dataclass
class Context:
    """Estado observado en t0 y parámetros; lo prepara services.py."""
    pa: PlayArrays
    fi: int
    rec: int                       # índice de entidad del receptor
    roles: dict
    model: MotionModel
    groups: np.ndarray             # código de grupo por entidad
    limits: dict[str, dict]
    rec_v_max: float
    rec_a_max: float
    los_x: float
    cap: float
    open_thr: float


def optimize(ctx: Context, horizon_s: float, defense_model: str) -> dict:
    pa, fi, rec = ctx.pa, ctx.fi, ctx.rec
    steps = int(round(horizon_s * FPS))
    p0 = np.array([pa.x[rec, fi], pa.y[rec, fi]], dtype=np.float64)
    s0, dir0 = float(pa.s[rec, fi]), float(pa.dir[rec, fi])
    v0 = np.array(heading_to_velocity(np.float64(s0), np.float64(dir0)))
    plans = build_plans(dir0, s0, ctx.rec_v_max, steps)
    pos, vel = simulate(p0, v0, plans.headings, plans.speeds, ctx.rec_v_max, ctx.rec_a_max)
    in_field, depth_ok = feasibility(pos, ctx.los_x, float(p0[0] - ctx.los_x))
    feasible = in_field & depth_ok

    F = pa.n_frames
    real_full = fi + steps < F
    real_rec = np.stack([pa.x[rec, fi:fi + steps + 1], pa.y[rec, fi:fi + steps + 1]], axis=-1).astype(np.float64)

    ent = np.flatnonzero(pa.side == SIDE_DEF)
    groups = ctx.groups[ent]
    feats_all = kinematic_features(pa)
    base = inertial_paths(ctx.model, pa, fi, ent, groups, steps, feats_all)
    reactive = choose_reactive(pa, fi, rec, ent, ctx.roles) if defense_model == "reactive" else np.array([], dtype=int)
    dv0 = np.stack(heading_to_velocity(pa.s[ent, fi].astype(np.float64), pa.dir[ent, fi].astype(np.float64)), axis=1)
    gl = [GROUPS[g] for g in groups]
    v_lim = np.array([ctx.limits[g]["v_max"] for g in gl])
    a_lim = np.array([ctx.limits[g]["a_max"] for g in gl])
    delay = int(round(REACTION_DELAY_S * FPS))

    cand = pos[feasible]
    allrec = np.concatenate([cand, real_rec[None]], axis=0) if real_full else cand
    defs = reactive_paths(base, dv0, reactive, allrec, v_lim, a_lim, delay)
    sep, who = separation(allrec, defs)
    m = score(sep, horizon_s, ctx.cap, ctx.open_thr)
    nc = cand.shape[0]
    cm = {k: v[:nc] for k, v in m.items()}
    ok = contact_ok(sep[:nc])
    order = rank(cm, ok) if nc else np.array([], dtype=int)
    top = diverse_top(order, cand) if nc else []
    return {
        "steps": steps, "p0": p0, "s0": s0, "dir0": dir0, "plans": plans, "pos": pos, "vel": vel,
        "feasible": feasible, "in_field": in_field, "depth_ok": depth_ok, "feasible_idx": np.flatnonzero(feasible),
        "metrics": cm, "contact_ok": ok, "order": order, "top": top, "sep": sep, "who": who, "defs": defs, "ent": ent,
        "reactive": reactive, "real_full": real_full, "real_rec": real_rec,
        "real_metrics": {k: v[nc] for k, v in m.items()} if real_full else None,
        "base": base,
    }


# ---------------------------------------------------------------------------- validación del modelo defensivo
class DefenseValidation:
    """Compara, en partidos reservados, la separación prevista por cada modelo defensivo (con el
    receptor siguiendo su ruta real) contra la separación real."""

    CHECK = (10, 20, 30)

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], list[float]] = {}

    def add(self, pa: PlayArrays, roles: dict, model: MotionModel, groups: np.ndarray, limits: dict) -> None:
        si, ti = pa.idx(pa.snap_frame), pa.idx(pa.throw_frame)
        if si is None or ti is None:
            return
        fi = max(LAG, si + 10)
        if fi >= ti:
            fi = max(LAG, si + 5)
        if fi >= ti:
            return
        steps = min(max(self.CHECK), pa.n_frames - 1 - fi)
        if steps < min(self.CHECK):
            return
        ent = np.flatnonzero(pa.side == SIDE_DEF)
        routes = [i for i in range(pa.ids.size) if roles.get(int(pa.ids[i]), ("",))[0] == "Pass Route"]
        if not routes or ent.size == 0:
            return
        g = groups[ent]
        base = inertial_paths(model, pa, fi, ent, g, steps)
        real_defs = np.stack([pa.x[ent, fi:fi + steps + 1], pa.y[ent, fi:fi + steps + 1]], axis=-1).astype(np.float64)
        dv0 = np.stack(heading_to_velocity(pa.s[ent, fi].astype(np.float64), pa.dir[ent, fi].astype(np.float64)), axis=1)
        gl = [GROUPS[x] for x in g]
        v_lim = np.array([limits[x]["v_max"] for x in gl])
        a_lim = np.array([limits[x]["a_max"] for x in gl])
        delay = int(round(REACTION_DELAY_S * FPS))
        for r in routes:
            rr = np.stack([pa.x[r, fi:fi + steps + 1], pa.y[r, fi:fi + steps + 1]], axis=-1).astype(np.float64)[None]
            true_sep, _ = separation(rr, real_defs[None])
            react = choose_reactive(pa, fi, r, ent, roles)
            for name, D in (("inertial", base[None]), ("reactive", reactive_paths(base, dv0, react, rr, v_lim, a_lim, delay))):
                pred_sep, _ = separation(rr, D)
                for k in self.CHECK:
                    if k > steps:
                        continue
                    pe = np.hypot(*(D[0, react, k] - real_defs[react, k]).T).mean() if react.size else np.nan
                    self.rows.setdefault((name, k), []).append((float(pred_sep[0, k] - true_sep[0, k]), float(pe)))

    def report(self) -> list[dict]:
        out = []
        for (name, k), vals in sorted(self.rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            a = np.array(vals)
            out.append({"model": name, "horizon_s": k / FPS, "n": int(len(a)),
                        "sep_mae": round(float(np.abs(a[:, 0]).mean()), 3), "sep_bias": round(float(a[:, 0].mean()), 3),
                        "near_def_pos_mae": round(float(np.nanmean(a[:, 1])), 3)})
        return out

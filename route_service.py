"""Servicios del Route Optimizer: validación de entradas, formato de la respuesta y explicaciones."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from . import routes as R
from .analysis import analyze_play
from .geometry import FPS
from .loader import SIDE_OFF
from .motion import GROUPS, LAG, pos_group
from .play import PlayArrays
from .services import clean, get_play, need_ready, play_meta, roles_of
from .store import MIN_PLAYS, Store

METRIC_KEYS = ("J", "window_mean", "s_min", "s_end", "s_max", "t_open", "t_max")
DEFENSE_LABEL = {
    "reactive": (f"los {R.N_REACTIVE} defensores de cobertura más cercanos persiguen al receptor tras "
                 f"{R.REACTION_DELAY_S:g} s de reacción; el resto sigue su inercia (modelo calibrado)"),
    "inertial": "todos los defensores siguen su inercia según el modelo calibrado y no reaccionan al receptor",
}


def uncertainty(store: Store, model: str, horizon_s: float) -> float | None:
    """Error absoluto medio de la separación prevista en partidos reservados, interpolado al horizonte."""
    rows = sorted((r for r in (store.route_validation or []) if r["model"] == model), key=lambda r: r["horizon_s"])
    if not rows:
        return None
    return float(np.interp(horizon_s, [r["horizon_s"] for r in rows], [r["sep_mae"] for r in rows]))


def route_info(store: Store) -> dict:
    need_ready(store)
    ref = store.sep_reference
    return clean({
        "limits": store.limits, "separation_reference": ref, "validation": store.route_validation,
        "horizons": list(R.ROUTE_HORIZONS), "defense_models": {k: DEFENSE_LABEL[k] for k in R.DEFENSE_MODELS},
        "parameters": {"dt_s": R.DT, "edge_margin_yd": R.EDGE_MARGIN, "reaction_delay_s": R.REACTION_DELAY_S,
                       "n_reactive": R.N_REACTIVE, "heading_step_deg": R.HEADING_STEP, "break_times_s": list(R.BREAK_TIMES),
                       "curve_rates_deg_s": list(R.CURVE_RATES), "window_s": R.WINDOW_S, "diversity_yd": R.DIVERSITY_YD,
                       "speed_quantile": R.SPEED_Q, "acc_quantile": R.ACC_Q},
        "objective": {
            "formula": "J = media_{t ∈ W} min(sep(t), cap);  sep(t) = min_j ‖r(t) − d_j(t)‖;  W = [max(0.5, H − 1), H]",
            "cap": ref["cap"], "open_thr": ref["open_thr"],
            "tiebreak": "Desempate: mayor separación mínima durante el recorrido; después, menor tiempo para alcanzar el umbral de apertura.",
        },
    })


def receiver_limits(store: Store, pid: int) -> tuple[float, float, str]:
    g = pos_group(store.positions.get(pid))
    lim = store.limits[g]
    t = store.player_table
    if pid in t.index and t.loc[pid, "plays"] >= MIN_PLAYS:
        return (float(t.loc[pid, "max_speed"]), float(lim["a_max"]),
                f"velocidad máxima propia observada en {int(t.loc[pid, 'plays'])} jugadas")
    return float(lim["v_max"]), float(lim["a_max"]), f"percentil {R.SPEED_Q * 100:g} del grupo {g} (muestra propia insuficiente)"


def _validate(store: Store, pa: PlayArrays, roles: dict, player_id: int, frame_id: int) -> tuple[int, int, int, int, list[int]]:
    si, ti = pa.idx(pa.snap_frame), pa.idx(pa.throw_frame)
    if ti is None:
        raise HTTPException(422, "Esta jugada no tiene evento de lanzamiento: el optimizador necesita un instante previo al pase.")
    if si is None:
        raise HTTPException(422, "Esta jugada no tiene evento de snap.")
    routes_ids = [int(pa.ids[i]) for i in range(pa.ids.size) if roles.get(int(pa.ids[i]), ("",))[0] == "Pass Route"]
    rec = pa.entity(player_id)
    if rec is None or player_id not in routes_ids:
        raise HTTPException(422, f"El jugador {player_id} no corre una ruta en esta jugada. Receptores disponibles: {routes_ids}.")
    lo, hi = max(LAG, si), ti - 1
    if hi < lo:
        raise HTTPException(422, "El lanzamiento ocurre demasiado pronto: no hay un instante previo con 0.5 s de historial.")
    fi = pa.idx(frame_id)
    if fi is None or fi < lo or fi > hi:
        raise HTTPException(422, f"El instante debe estar entre el snap y el lanzamiento: frames {int(pa.frames[lo])} a {int(pa.frames[hi])}.")
    return si, ti, rec, fi, routes_ids


def route_optimize(store: Store, game_id: int, play_id: int, player_id: int, frame_id: int, horizon_s: float,
                   defense_model: str, reveal: bool) -> dict:
    need_ready(store)
    if store.motion_model is None or not store.limits or not store.sep_reference:
        raise HTTPException(503, "El modelo de movimiento aún no está calibrado.")
    pa = get_play(store, game_id, play_id)
    meta = play_meta(store, game_id, play_id)
    roles = roles_of(store, game_id, play_id)
    si, ti, rec, fi, routes_ids = _validate(store, pa, roles, player_id, frame_id)
    v_max, a_max, v_src = receiver_limits(store, player_id)
    los_x = analyze_play(pa, meta, roles, store.names)["play"]["los_x"]
    groups = store.group_codes(pa)
    ctx = R.Context(pa=pa, fi=fi, rec=rec, roles=roles, model=store.motion_model, groups=groups, limits=store.limits,
                    rec_v_max=v_max, rec_a_max=a_max, los_x=los_x, cap=store.sep_reference["cap"],
                    open_thr=store.sep_reference["open_thr"])
    res = R.optimize(ctx, horizon_s, defense_model)
    plans, ent, fidx, cm = res["plans"], res["ent"], res["feasible_idx"], res["metrics"]
    nc = int(fidx.size)
    if nc == 0:
        raise HTTPException(422, "Ninguna ruta candidata cumple las restricciones (campo y profundidad) desde esta posición.")
    delta = uncertainty(store, defense_model, horizon_s)
    rank_of = {int(c): r for r, c in enumerate(res["order"])}
    def_ids = pa.ids[ent]

    def r2(a):
        return np.round(np.asarray(a, dtype=float), 2)

    def cand_row(c: int) -> dict:
        p = int(fidx[c])
        return {"id": p, "rank": rank_of[c] + 1, **plans.meta[p], "path": r2(res["pos"][p]),
                "contact_ok": bool(res["contact_ok"][c]),
                **{k: round(float(cm[k][c]), 3) for k in METRIC_KEYS}}

    def curve(row: int) -> dict:
        return {"sep": r2(res["sep"][row]), "nearest_id": [int(def_ids[j]) for j in res["who"][row]],
                "reactive_paths": {str(int(def_ids[j])): r2(res["defs"][row, j]) for j in res["reactive"]}}

    best_J = float(cm["J"][res["top"][0]])
    top = [{**cand_row(c), **curve(c), "similar_to_best": delta is not None and best_J - float(cm["J"][c]) <= delta}
           for c in res["top"]]
    real = {"available": bool(res["real_full"]), "path": r2(res["real_rec"]),
            "note": None if res["real_full"] else "La jugada termina antes del horizonte: la ruta real no se puede puntuar completa."}
    if res["real_full"]:
        real.update({k: round(float(res["real_metrics"][k]), 3) for k in METRIC_KEYS}, **curve(nc))
    reactive_ids = {int(def_ids[j]) for j in res["reactive"]}
    defenders = [{"id": int(pa.ids[e]), "name": store.names.get(int(pa.ids[e])),
                  "jersey": None if np.isnan(pa.jersey[e]) else int(pa.jersey[e]), "position": store.positions.get(int(pa.ids[e])),
                  "role": roles.get(int(pa.ids[e]), (None,))[0], "x": float(pa.x[e, fi]), "y": float(pa.y[e, fi]),
                  "s": float(pa.s[e, fi]), "dir": float(pa.dir[e, fi]), "reactive": int(pa.ids[e]) in reactive_ids,
                  "inertial_path": r2(res["base"][j])} for j, e in enumerate(ent)]
    offense = [{"id": int(pa.ids[i]), "name": store.names.get(int(pa.ids[i])), "jersey": None if np.isnan(pa.jersey[i]) else int(pa.jersey[i]),
                "x": float(pa.x[i, fi]), "y": float(pa.y[i, fi]), "s": float(pa.s[i, fi]), "dir": float(pa.dir[i, fi]),
                "is_route": int(pa.ids[i]) in routes_ids} for i in np.flatnonzero(pa.side == SIDE_OFF) if i != rec]
    vel = res["vel"][res["feasible"]]
    pts = res["pos"][res["feasible"]][:, 1:]
    dv = np.diff(vel, axis=1)
    compliance = {"max_speed": float(np.hypot(vel[..., 0], vel[..., 1]).max()),
                  "max_acc": float(np.hypot(dv[..., 0], dv[..., 1]).max() / R.DT),
                  "min_x": float(pts[..., 0].min()), "max_x": float(pts[..., 0].max()),
                  "min_y": float(pts[..., 1].min()), "max_y": float(pts[..., 1].max())}
    out = {
        "game_id": game_id, "play_id": play_id, "player_id": player_id, "name": store.names.get(player_id),
        "frame_id": frame_id, "horizon_s": horizon_s, "defense_model": defense_model,
        "defense_model_text": DEFENSE_LABEL[defense_model], "t0_from_snap_s": (fi - si) / FPS, "throw_in_s": (ti - fi) / FPS,
        "los_x": los_x, "state": {"x": float(res["p0"][0]), "y": float(res["p0"][1]), "speed": res["s0"], "dir": res["dir0"]},
        "limits": {"v_max": v_max, "a_max": a_max, "v_source": v_src, "edge_margin": R.EDGE_MARGIN,
                   "defenders": {g: store.limits[g] for g in sorted({GROUPS[x] for x in groups[ent]})}},
        "compliance": compliance,
        "objective": {"window": [R.window_start(horizon_s), horizon_s], "cap": ctx.cap, "open_thr": ctx.open_thr,
                      "uncertainty": delta, "contact_yd": R.CONTACT_YD},
        "counts": {"generated": int(res["pos"].shape[0]), "feasible": nc, "out_of_field": int((~res["in_field"]).sum()),
                   "behind_los": int((res["in_field"] & ~res["depth_ok"]).sum()),
                   "contact": int((~res["contact_ok"]).sum())},
        "candidates": [cand_row(c) for c in range(nc)], "top": top, "real": real, "defenders": defenders, "offense": offense,
        "receivers": [{"id": i, "name": store.names.get(i)} for i in routes_ids],
    }
    out["explanation"] = explanation(store, out, delta)
    out["warnings"] = warnings(store, defense_model, horizon_s)
    if reveal:
        out["reveal"] = verification(pa, fi, res["steps"], ent, res, out)
    return clean(out)


def explanation(store: Store, o: dict, delta: float | None) -> list[str]:
    best, top, real, c, obj = o["top"][0], o["top"], o["real"], o["counts"], o["objective"]
    lines = [f"Se generaron {c['generated']} rutas candidatas; {c['feasible']} cumplen las restricciones "
             f"({c['out_of_field']} salen del campo y {c['behind_los']} terminan demasiado atrás de la línea de golpeo)."]
    if c["contact"]:
        lines.append(f"{c['contact']} de ellas pasan a menos de {obj['contact_yd']:g} yd de un defensor previsto y quedan "
                     "detrás en el orden: el modelo no representa el contacto.")
    if not best["contact_ok"]:
        lines.append("Ninguna ruta evita el contacto bajo este modelo; se muestra la de mayor J, que exige revisión.")
    lines.append(f"Recomendada: «{best['label']}». Su separación útil es J = {best['J']:.2f} yd: la separación media con el "
                 f"defensor más cercano entre {obj['window'][0]:g} y {obj['window'][1]:g} s, con un tope de {obj['cap']:.1f} yd.")
    unc = f"del modelo defensivo a {o['horizon_s']:g} s (±{delta:.2f} yd)" if delta is not None else "del modelo defensivo"
    if real["available"]:
        d = best["J"] - real["J"]
        if delta is not None and d > delta:
            lines.append(f"Supera a la ruta real (J = {real['J']:.2f} yd) por {d:.2f} yd, más que la incertidumbre {unc}.")
        elif d > 0:
            lines.append(f"Supera a la ruta real (J = {real['J']:.2f} yd) por {d:.2f} yd, pero la diferencia no supera la "
                         f"incertidumbre {unc}: bajo este modelo, la ruta real ya era competitiva.")
        else:
            lines.append(f"Ninguna candidata supera a la ruta real (J = {real['J']:.2f} yd) bajo este modelo; "
                         "la recomendada es la mejor alternativa generada.")
    if len(top) > 1:
        gap = best["J"] - top[1]["J"]
        if delta is not None and gap <= delta:
            n_sim = sum(1 for t in top if t["similar_to_best"])
            lines.append(f"La segunda alternativa queda a {gap:.2f} yd. {n_sim} de las {len(top)} mejores están dentro de la "
                         "incertidumbre: conviene tratarlas como opciones equivalentes para el entrenamiento.")
        else:
            lines.append(f"Su ventaja sobre la segunda alternativa ({gap:.2f} yd) es mayor que la incertidumbre del modelo.")
    lines.append(f"Separación mínima durante el recorrido: {best['s_min']:.2f} yd; al final: {best['s_end']:.2f} yd.")
    if best["s_min"] < 1.0:
        lines.append("Atención: durante el recorrido pasa a menos de 1 yd de un defensor, donde el contacto es probable "
                     "y el modelo no lo representa.")
    if best["t_open"] is not None and not np.isnan(best["t_open"]):
        lines.append(f"Alcanza {obj['open_thr']:.1f} yd de separación (la mediana observada al lanzar en el dataset) "
                     f"a los {best['t_open']:.1f} s.")
    else:
        lines.append(f"No llega a {obj['open_thr']:.1f} yd de separación (mediana observada al lanzar) dentro del horizonte.")
    nid = best["nearest_id"][-1]
    lines.append(f"Al final del recorrido, el defensor más cercano sería {store.names.get(nid, nid)}.")
    if o["throw_in_s"] <= o["horizon_s"]:
        lines.append(f"En la jugada real, el lanzamiento ocurrió {o['throw_in_s']:.1f} s después del instante elegido, "
                     "dentro del horizonte simulado.")
    return lines


def warnings(store: Store, model: str, horizon_s: float) -> list[str]:
    w = ["Simulación hipotética: las rutas candidatas no ocurrieron en el partido y sirven como hipótesis de entrenamiento.",
         f"Modelo defensivo simplificado: {DEFENSE_LABEL[model]}. Los defensores reales pueden reaccionar de otra forma, "
         "cambiar de asignación, anticipar la ruta o recibir ayuda de otro defensor.",
         "Más separación no garantiza un pase completo: no se modelan la visión y la presión sobre el QB, el vuelo del balón, "
         "la técnica de recepción ni el contacto.",
         "La ruta recomendada es la mejor entre las candidatas generadas bajo este modelo, no la óptima frente a todas las "
         "respuestas defensivas posibles."]
    rows = [r for r in (store.route_validation or []) if r["model"] == model]
    if rows:
        r = min(rows, key=lambda r: abs(r["horizon_s"] - horizon_s))
        trend = "sobreestima" if r["sep_bias"] > 0 else "subestima"
        w.append(f"En {r['n']} casos de partidos reservados, este modelo {trend} la separación real a {r['horizon_s']:g} s "
                 f"en {abs(r['sep_bias']):.2f} yd de media (error absoluto medio {r['sep_mae']:.2f} yd).")
    return w


def verification(pa: PlayArrays, fi: int, steps: int, ent: np.ndarray, res: dict, o: dict) -> dict:
    """Control posterior con posiciones reales de los defensores. Se calcula después de elegir la ruta."""
    last = min(fi + steps, pa.n_frames - 1)
    n = last - fi + 1
    real_defs = np.stack([pa.x[ent, fi:last + 1], pa.y[ent, fi:last + 1]], axis=-1).astype(np.float64)

    def sep_vs_real(path) -> np.ndarray:
        p = np.asarray(path, dtype=float)[:n]
        return np.round(np.hypot(p[None, :, 0] - real_defs[..., 0], p[None, :, 1] - real_defs[..., 1]).min(axis=0), 2)

    out = {"steps_available": n - 1,
           "defender_paths": {str(int(pa.ids[e])): np.round(real_defs[j], 2) for j, e in enumerate(ent)},
           "top_sep": [sep_vs_real(t["path"]) for t in o["top"]], "real_sep": sep_vs_real(o["real"]["path"]),
           "note": "Los defensores reales reaccionaron a la ruta real, no a las alternativas: esta comparación es solo un control."}
    if res["real_full"]:
        pred = res["defs"][-1]   # defensores previstos frente a la ruta real
        err = np.hypot(pred[:, :n, 0] - real_defs[..., 0], pred[:, :n, 1] - real_defs[..., 1])
        out["defender_error"] = {"all_end": float(err[:, -1].mean()),
                                 "reactive_end": float(err[res["reactive"], -1].mean()) if res["reactive"].size else None}
    return out

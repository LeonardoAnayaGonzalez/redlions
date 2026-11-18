"""Estado de la aplicación: carga de tablas, preprocesamiento de todo el dataset y caché.

El procesamiento pesado (122 archivos de tracking) corre una sola vez en un hilo al
arrancar. Los endpoints de exploración de jugadas funcionan desde el primer momento;
los agregados (resumen, jugadores, modelos) esperan a `status == "ready"`.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import loader
from .analysis import SPEED_BINS, analyze_play
from .loader import SIDE_BALL
from .motion import (GROUPS, EvalAccumulator, FitAccumulator, MotionModel, build_samples, pos_group)
from .play import PlayArrays, split_plays
from .routes import DefenseValidation, LimitAccumulator
from .playbook import CoverageValidation, RushCurve
from .schema import SchemaError, resolve
from .stats import auc_score, brier, calibration_table, fit_logistic, log_loss, percentile_rank, sigmoid

MIN_PLAYS = 20
MIN_SEP_PLAYS = 15
TEST_EVERY = 4                      # 1 de cada 4 partidos queda reservado para validación
COMPLETION_FEATURES = ["sep", "dist_qb", "depth", "time_to_throw", "rush_min", "speed"]
OTHER_CODE = GROUPS.index("OTHER")


class Store:
    def __init__(self, max_games: int | None = None):
        self.max_games = max_games
        self.status = {"state": "idle", "done": 0, "total": 0, "message": "Sin iniciar", "seconds": 0.0}
        self.lock = threading.Lock()
        self._cache: OrderedDict[int, dict[int, PlayArrays]] = OrderedDict()
        self.cache_lock = threading.Lock()
        self.games = self.plays = self.players = self.scouting = None
        self.files: dict[int, object] = {}
        self.names: dict[int, str] = {}
        self.positions: dict[int, str] = {}
        self.capabilities: dict[str, dict] = {}
        self.quality: dict = {}
        self.pp = self.pp_hist = self.play_stats = self.recv = self.player_table = None
        self.motion_model: MotionModel | None = None
        self.motion_report: dict | None = None
        self.completion: dict | None = None
        self.train_games: list[int] = []
        self.test_games: list[int] = []
        self.limits: dict | None = None
        self.route_validation: list[dict] | None = None
        self.sep_reference: dict | None = None
        self.rush_curve = None
        self.rush_curve_n = None
        self.throw_window: tuple[float, float] | None = None
        self.yards_model: dict | None = None
        self.coverage_validation: list[dict] | None = None

    # ------------------------------------------------------------------ carga estática
    def load_static(self) -> None:
        self.games = loader.read_table("games.csv", "games")
        plays = loader.read_table("plays.csv", "plays")
        from .schema import ALIASES
        plays = plays.reindex(columns=list(ALIASES["plays"].keys()))
        plays["play_key"] = plays["game_id"].astype("int64") * 100000 + plays["play_id"].astype("int64")
        plays["play_type"] = plays["dropback"].fillna("SIN_DATO")
        self.plays = plays
        self.players = loader.read_table("players.csv", "players")
        self.scouting = loader.read_table("pffScoutingData.csv", "scouting")
        self.names = dict(zip(self.players["player_id"], self.players["name"]))
        self.positions = dict(zip(self.players["player_id"], self.players["position"]))
        self.files = loader.tracking_files()
        usable = [g for g in self.games["game_id"] if g in self.files]
        if self.max_games:
            usable = sorted(usable)[: self.max_games]
            self.games = self.games[self.games["game_id"].isin(usable)]
            self.plays = self.plays[self.plays["game_id"].isin(usable)]
        first = next(iter(self.files.values()), None)
        has_a = has_event = False
        if first is not None:
            head = pd.read_csv(first, nrows=0)
            m = resolve(list(head.columns), "tracking")
            has_a, has_event = "a" in m, "event" in m
        self.capabilities = {
            "tracking": {"available": first is not None, "reason": None if first else "No hay archivos en data/tracking."},
            "acceleration": {"available": has_a, "reason": None if has_a else "El tracking no incluye aceleración."},
            "events": {"available": has_event, "reason": None if has_event else "El tracking no incluye eventos (snap, lanzamiento)."},
            "pass_roles": {"available": True, "reason": None},
            "coverage": {"available": bool(self.plays["coverage"].notna().any()),
                         "reason": None if self.plays["coverage"].notna().any() else "plays.csv no trae coberturas."},
        }

    # ------------------------------------------------------------------ acceso a jugadas
    def game_ids(self) -> list[int]:
        return sorted(g for g in self.games["game_id"] if g in self.files)

    def get_game(self, game_id: int) -> dict[int, PlayArrays]:
        """Jugadas de un partido (caché LRU de 12 partidos)."""
        with self.cache_lock:
            if game_id in self._cache:
                self._cache.move_to_end(game_id)
                return self._cache[game_id]
        if game_id not in self.files:
            raise KeyError(game_id)
        pg = self.plays[self.plays["game_id"] == game_id]
        df, _ = loader.load_game_tracking(self.files[game_id], game_id, pg)
        out = {p.play_id: p for p in split_plays(df, game_id) if isinstance(p, PlayArrays)}
        with self.cache_lock:
            self._cache[game_id] = out
            while len(self._cache) > 12:
                self._cache.popitem(last=False)
        return out

    def roles_for_game(self, game_id: int) -> dict[int, dict[int, tuple[str, str]]]:
        sc = self.scouting[self.scouting["game_id"] == game_id]
        out: dict[int, dict[int, tuple[str, str]]] = {}
        for pid, sub in sc.groupby("play_id"):
            out[int(pid)] = {int(a): (b, c) for a, b, c in zip(sub["player_id"], sub["role"], sub["lined_up"])}
        return out

    def group_codes(self, pa: PlayArrays) -> np.ndarray:
        codes = np.full(pa.ids.size, -1, dtype=np.int64)
        for i, pid in enumerate(pa.ids):
            if pa.side[i] != SIDE_BALL:
                codes[i] = GROUPS.index(pos_group(self.positions.get(int(pid))))
        return codes

    # ------------------------------------------------------------------ preprocesamiento
    def start_background(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="precompute").start()

    def _run(self) -> None:
        t0 = time.time()
        try:
            self.status.update(state="loading", message="Leyendo tablas")
            if self.games is None:
                self.load_static()
            self._precompute(t0)
            self.status.update(state="ready", message="Listo", seconds=round(time.time() - t0, 1))
        except Exception as exc:  # noqa: BLE001 - se reporta al frontend
            self.status.update(state="error", message=f"{type(exc).__name__}: {exc}")

    def _precompute(self, t0: float) -> None:
        ids = self.game_ids()
        self.test_games = [g for i, g in enumerate(ids) if i % TEST_EVERY == TEST_EVERY - 1]
        self.train_games = [g for g in ids if g not in set(self.test_games)]
        test_set = set(self.test_games)
        self.status.update(done=0, total=len(ids) + len(self.test_games))

        acc = FitAccumulator()
        lim = LimitAccumulator()
        rcurve = RushCurve()
        pp_parts, hist_parts, play_rows, recv_rows = [], [], [], []
        q = {"rows_raw": 0, "rows_duplicate": 0, "rows_other_game": 0, "irregular_plays": 0, "nan_dir_players": 0,
             "tracking_plays": 0}
        tracked_keys: set[int] = set()
        skipped: list[dict] = []

        def load(gid):
            pg = self.plays[self.plays["game_id"] == gid]
            return loader.load_game_tracking(self.files[gid], gid, pg)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {ids[0]: pool.submit(load, ids[0])} if ids else {}
            for n, gid in enumerate(ids):
                if n + 1 < len(ids):
                    futures[ids[n + 1]] = pool.submit(load, ids[n + 1])
                df, gq = futures.pop(gid).result()
                for k in ("rows_raw", "rows_duplicate", "rows_other_game", "nan_dir_players"):
                    q[k] += gq.get(k, 0)
                roles = self.roles_for_game(gid)
                pg = self.plays[self.plays["game_id"] == gid].set_index("play_id")
                meta_all = pg.to_dict("index")
                samples = []
                for item in split_plays(df, gid):
                    if not isinstance(item, PlayArrays):
                        q["irregular_plays"] += 1
                        skipped.append({"game_id": gid, "play_id": item[0], "reason": item[1]})
                        continue
                    pa = item
                    meta = meta_all.get(pa.play_id)
                    if meta is None:
                        continue
                    q["tracking_plays"] += 1
                    tracked_keys.add(gid * 100000 + pa.play_id)
                    res = analyze_play(pa, meta, roles.get(pa.play_id, {}), self.names)
                    r = res["rows"]
                    n_rows = r["player_id"].size
                    part = pd.DataFrame({
                        "game_id": gid, "play_id": pa.play_id, "player_id": r["player_id"], "side": r["side"],
                        "team": r["team"], "role": r["role"], "lined_up": r["lined_up"],
                        "max_speed": r["max_speed"], "mean_speed": r["mean_speed"], "max_acc": r["max_acc"],
                        "dist": r["dist"], "duration": r["duration"], "sep_throw": r["sep_throw"]})
                    pp_parts.append(part)
                    hist_parts.append(r["hist"])
                    play_rows.append(res["play"])
                    recv_rows.extend(res["receivers"])
                    codes = self.group_codes(pa)
                    lim.add(pa, codes, pa.idx(pa.snap_frame) or 0)
                    rcurve.add(pa, roles.get(pa.play_id, {}))
                    if gid not in test_set:
                        s = build_samples(pa, codes, pa.idx(pa.snap_frame) or 0)
                        if s is not None:
                            samples.append(s)
                if samples:
                    acc.add(np.concatenate([s[0] for s in samples]), np.concatenate([s[1] for s in samples]),
                            np.concatenate([s[2] for s in samples]))
                self.status.update(done=n + 1, message=f"Procesando partidos {n + 1}/{len(ids)}",
                                   seconds=round(time.time() - t0, 1))

        self.pp = pd.concat(pp_parts, ignore_index=True)
        self.pp_hist = np.concatenate(hist_parts)
        self.play_stats = pd.DataFrame(play_rows)
        self.recv = pd.DataFrame(recv_rows)
        self._enrich_tables()

        self.motion_model = MotionModel.fit(acc)
        self.limits = lim.limits()
        seps = self.recv["sep"].to_numpy() if len(self.recv) else np.array([])
        self.sep_reference = {"open_thr": round(float(np.median(seps)), 3) if seps.size else 2.5,
                              "cap": round(float(np.quantile(seps, 0.95)), 3) if seps.size else 8.0, "n": int(seps.size)}
        dval = DefenseValidation()
        self.rush_curve, self.rush_curve_n = rcurve.curve()
        cval = CoverageValidation()
        cov_type = self.plays.set_index("play_key")["coverage_type"].to_dict()
        ps_los = dict(zip(self.play_stats["play_key"], self.play_stats["los_x"]))
        ev = EvalAccumulator()
        done = len(ids)
        for gid in self.test_games:
            df, _ = loader.load_game_tracking(self.files[gid], gid, self.plays[self.plays["game_id"] == gid])
            roles = self.roles_for_game(gid)
            for item in split_plays(df, gid):
                if isinstance(item, PlayArrays):
                    codes = self.group_codes(item)
                    s = build_samples(item, codes, item.idx(item.snap_frame) or 0)
                    if s is not None:
                        ev.add(self.motion_model, *s)
                    dval.add(item, roles.get(item.play_id, {}), self.motion_model, codes, self.limits)
                    key = gid * 100000 + item.play_id
                    de = np.flatnonzero(item.side == 1)
                    gl = [GROUPS[c] for c in codes[de]]
                    cval.add(item, roles.get(item.play_id, {}), cov_type.get(key), (np.array([self.limits[g]["v_max"] for g in gl]),
                             np.array([self.limits[g]["a_max"] for g in gl])), self.rush_curve, ps_los.get(key, np.nan))
            done += 1
            self.status.update(done=done, message=f"Validando modelo {done - len(ids)}/{len(self.test_games)}")
        self.motion_report = ev.report()
        self.motion_report["train_samples"] = int(acc.count.get((-1, 5), 0))
        self.route_validation = dval.report()
        self.coverage_validation = cval.report()
        tt = self.play_stats["time_to_throw"].dropna()
        self.throw_window = (round(float(tt.quantile(0.1)), 1), round(float(tt.quantile(0.9)), 1)) if len(tt) else (1.9, 4.0)

        self._fit_completion()
        self._fit_yards()
        self._build_player_table()
        self._finalize_quality(q, tracked_keys, skipped)

    def _enrich_tables(self) -> None:
        pl = self.plays.set_index(["game_id", "play_id"])
        pp = self.pp
        pp["position"] = pp["player_id"].map(self.positions).fillna("SIN_DATO")
        pp["pos_group"] = pp["position"].map(lambda p: pos_group(p))
        key = pd.MultiIndex.from_arrays([pp["game_id"], pp["play_id"]])
        pp["play_type"] = pl["play_type"].reindex(key).to_numpy()
        pp["pass_result"] = pl["pass_result"].reindex(key).to_numpy()
        pp["play_key"] = pp["game_id"].astype("int64") * 100000 + pp["play_id"].astype("int64")
        ps = self.play_stats
        ps["play_key"] = ps["game_id"].astype("int64") * 100000 + ps["play_id"].astype("int64")
        r = self.recv
        if len(r):
            r["name"] = r["player_id"].map(self.names)
            r["play_key"] = r["game_id"].astype("int64") * 100000 + r["play_id"].astype("int64")

    # ------------------------------------------------------------------ modelo de completitud
    def _fit_completion(self) -> None:
        info = {"enabled": False, "reason": None}
        self.completion = info
        r = self.recv
        if r is None or not len(r):
            info["reason"] = "No se identificaron lanzamientos con receptores en el tracking."
            return
        d = r[r["is_target"]].merge(self.play_stats[["play_key", "time_to_throw", "rush_min"]], on="play_key")
        d = d.merge(self.plays[["play_key", "pass_result"]], on="play_key")
        d = d[d["pass_result"].isin(["C", "I", "IN"])].copy()
        d["y"] = (d["pass_result"] == "C").astype(int)
        train = d[d["game_id"].isin(self.train_games)]
        test = d[d["game_id"].isin(self.test_games)]
        info.update(n_targets=int(len(d)), n_train=int(len(train)), n_test=int(len(test)))
        if len(train) < 300 or len(test) < 100 or train["y"].nunique() < 2:
            info["reason"] = f"Muestra insuficiente para entrenar y validar (train={len(train)}, test={len(test)})."
            return
        med = {c: float(train[c].median()) for c in COMPLETION_FEATURES}
        info["ranges"] = {c: [float(train[c].quantile(0.01)), float(train[c].quantile(0.99))] for c in COMPLETION_FEATURES}
        Xtr = self._comp_matrix(train, med)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        beta = fit_logistic(np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))], train["y"].to_numpy(float), l2=1.0)
        Xte = self._comp_matrix(test, med)
        p = sigmoid(np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ beta)
        y = test["y"].to_numpy(float)
        base = float(train["y"].mean())
        pb = np.full_like(y, base)
        m = {"auc": auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p),
             "log_loss_base": log_loss(y, pb), "brier_base": brier(y, pb)}
        info.update(
            features=COMPLETION_FEATURES, medians=med, mu=mu.tolist(), sd=sd.tolist(), beta=beta.tolist(),
            base_rate_train=base, base_rate_test=float(y.mean()), metrics={k: round(float(v), 4) for k, v in m.items()},
            calibration=calibration_table(y, p),
            coefficients=[{"feature": f, "beta_std": round(float(b), 4), "odds_ratio": round(float(np.exp(b)), 3)}
                          for f, b in zip(COMPLETION_FEATURES, beta[:-1])])
        info["enabled"] = bool(m["auc"] > 0.55 and m["log_loss"] < m["log_loss_base"])
        info["reason"] = None if info["enabled"] else "El modelo no supera al modelo base en los partidos de validación."

    def _fit_yards(self) -> None:
        """Yardas de la jugada si el pase se completa: OLS sobre profundidad y separación al lanzar."""
        r = self.recv
        self.yards_model = None
        if r is None or not len(r):
            return
        d = r[r["is_target"]].merge(self.plays[["play_key", "pass_result", "play_result"]], on="play_key")
        d = d[(d["pass_result"] == "C") & d["play_result"].notna()]
        tr, te = d[d["game_id"].isin(self.train_games)], d[d["game_id"].isin(self.test_games)]
        if len(tr) < 200 or len(te) < 50:
            return
        X = lambda x: np.c_[np.ones(len(x)), x["depth"].to_numpy(float), x["sep"].to_numpy(float)]  # noqa: E731
        coef, *_ = np.linalg.lstsq(X(tr), tr["play_result"].to_numpy(float), rcond=None)
        yt = te["play_result"].to_numpy(float)
        pred = X(te) @ coef
        self.yards_model = {"coef": [float(c) for c in coef], "n_train": int(len(tr)), "n_test": int(len(te)),
                            "mae": round(float(np.abs(pred - yt).mean()), 3),
                            "mae_base": round(float(np.abs(tr["play_result"].mean() - yt).mean()), 3),
                            "r2": round(float(1 - ((pred - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 3)}

    @staticmethod
    def _comp_matrix(d: pd.DataFrame, med: dict) -> np.ndarray:
        cols = []
        for c in COMPLETION_FEATURES:
            v = d[c].astype(float).fillna(med[c]).to_numpy()
            cols.append(np.clip(v, -5, 15) if c == "rush_min" else v)
        return np.column_stack(cols)

    def completion_probability(self, rows: pd.DataFrame) -> np.ndarray | None:
        info = self.completion
        if not info or not info.get("enabled"):
            return None
        X = self._comp_matrix(rows, info["medians"])
        Z = (X - np.array(info["mu"])) / np.array(info["sd"])
        return sigmoid(np.c_[Z, np.ones(len(Z))] @ np.array(info["beta"]))

    # ------------------------------------------------------------------ jugadores
    def _build_player_table(self) -> None:
        pp = self.pp
        g = pp.groupby("player_id")
        tbl = pd.DataFrame({
            "plays": g.size(), "max_speed": g["max_speed"].max(), "p90_speed": g["max_speed"].quantile(0.9),
            "mean_speed": g["mean_speed"].mean(), "p90_acc": g["max_acc"].quantile(0.9), "max_acc": g["max_acc"].max(),
            "dist_total": g["dist"].sum(), "mobility": g["dist"].sum() / g["duration"].sum(),
            "cv_speed": g["max_speed"].std() / g["max_speed"].mean(),
        })
        tbl["team"] = g["team"].agg(lambda s: s.mode().iat[0])
        tbl["position"] = tbl.index.map(lambda i: self.positions.get(int(i), "SIN_DATO"))
        tbl["pos_group"] = tbl["position"].map(pos_group)
        tbl["name"] = tbl.index.map(lambda i: self.names.get(int(i), f"Jugador {i}"))
        s = pp[pp["sep_throw"].notna()]
        s_open = s[s["role"] == "Pass Route"].groupby("player_id")["sep_throw"].agg(["size", "mean"])
        s_tight = s[s["role"] == "Coverage"].groupby("player_id")["sep_throw"].agg(["size", "mean"])
        tbl["sep_open_n"] = s_open["size"].reindex(tbl.index).fillna(0).astype(int)
        tbl["sep_open"] = s_open["mean"].reindex(tbl.index)
        tbl["sep_tight_n"] = s_tight["size"].reindex(tbl.index).fillna(0).astype(int)
        tbl["sep_tight"] = s_tight["mean"].reindex(tbl.index)
        tbl["sep_kind"] = np.where(tbl["sep_open_n"] >= tbl["sep_tight_n"], "open", "tight")
        tbl["sep_n"] = np.where(tbl["sep_kind"] == "open", tbl["sep_open_n"], tbl["sep_tight_n"])
        tbl["sep_value"] = np.where(tbl["sep_kind"] == "open", tbl["sep_open"], tbl["sep_tight"])
        tbl.loc[tbl["sep_n"] < MIN_SEP_PLAYS, "sep_value"] = np.nan
        has_acc = self.capabilities["acceleration"]["available"]
        raw = {"speed": tbl["p90_speed"], "acceleration": tbl["p90_acc"] if has_acc else pd.Series(np.nan, tbl.index),
               "separation": tbl["sep_value"].where(tbl["sep_kind"] == "open", -tbl["sep_value"]),
               "consistency": -tbl["cv_speed"], "mobility": tbl["mobility"]}
        for dim, series in raw.items():
            tbl[f"raw_{dim}"] = series
            tbl[f"pct_{dim}"] = np.nan
            tbl[f"pool_{dim}"] = 0
            for keys, sub in tbl.groupby(["pos_group", "sep_kind"] if dim == "separation" else ["pos_group"]):
                pool = sub[(sub["plays"] >= MIN_PLAYS) & sub[f"raw_{dim}"].notna()][f"raw_{dim}"].to_numpy()
                if pool.size < 10:
                    continue
                idx = sub.index[sub[f"raw_{dim}"].notna()]
                tbl.loc[idx, f"pct_{dim}"] = [percentile_rank(v, pool) for v in sub.loc[idx, f"raw_{dim}"]]
                tbl.loc[sub.index, f"pool_{dim}"] = pool.size
        dims = [f"pct_{d}" for d in raw]
        tbl["n_dims"] = tbl[dims].notna().sum(axis=1)
        tbl["composite"] = tbl[dims].mean(axis=1).where((tbl["n_dims"] >= 3) & (tbl["plays"] >= MIN_PLAYS))
        self.player_table = tbl

    # ------------------------------------------------------------------ calidad de datos
    def _finalize_quality(self, q: dict, tracked_keys: set[int], skipped: list[dict]) -> None:
        plays_csv = set(self.plays["play_key"])
        ps = self.play_stats
        passes = ps[ps["throw_frame"].notna()]
        tgt_candidates = self.plays[self.plays["pass_result"].isin(["C", "I", "IN"])]
        tgt_candidates = tgt_candidates[tgt_candidates["play_key"].isin(tracked_keys)]
        resolved = int(ps["target_id"].notna().sum())
        miss_cols = {c: int(self.plays[c].isna().sum()) for c in
                     ["yardline_side", "absolute_yardline", "formation", "defenders_box", "dropback", "coverage"]}
        in_games = set(self.games["game_id"])
        self.quality = {
            "games_csv": int(len(self.games)), "tracking_files": int(len(self.files)),
            "games_without_tracking": sorted(int(g) for g in in_games - set(self.files)),
            "tracking_without_game": sorted(int(g) for g in set(self.files) - in_games) if not self.max_games else [],
            "plays_csv": int(len(plays_csv)), "plays_tracked": int(len(tracked_keys & plays_csv)),
            "plays_without_tracking": int(len(plays_csv - tracked_keys)),
            "irregular_plays_skipped": q["irregular_plays"], "irregular_examples": skipped[:5],
            "tracking_rows": q["rows_raw"], "duplicate_rows_removed": q["rows_duplicate"],
            "rows_from_other_game": q["rows_other_game"], "players_missing_direction_rows": q["nan_dir_players"],
            "plays_with_pass_forward": int(len(passes)), "plays_without_pass_forward": int(len(ps) - len(passes)),
            "pass_attempts_with_target_candidate": int(len(tgt_candidates)), "targets_resolved": resolved,
            "target_resolution_rate": round(resolved / max(len(tgt_candidates), 1), 3),
            "plays_without_scouting_roles": int((ps["n_routes"] == 0).sum()),
            "speed_over_13_yds": int((self.pp["max_speed"] > 13).sum()),
            "scouting_duplicates": int(self.scouting.duplicated(["game_id", "play_id", "player_id"]).sum()),
            "players_duplicates": int(self.players["player_id"].duplicated().sum()),
            "plays_missing_values": miss_cols,
            "split": {"method": "por partido (1 de cada 4 reservado)", "train_games": len(self.train_games),
                      "test_games": len(self.test_games)},
        }
        self.capabilities["passes"] = {"available": bool(len(passes)),
                                       "reason": None if len(passes) else "No hay eventos de lanzamiento en el tracking."}
        self.capabilities["target_inference"] = {
            "available": resolved > 0,
            "reason": None if resolved else "No se pudo asociar el receptor objetivo desde la descripción."}
        comp = self.completion or {}
        self.capabilities["completion_model"] = {"available": bool(comp.get("enabled")), "reason": comp.get("reason")}
        mr = self.motion_report or {}
        hz = mr.get("horizons", [])
        better = bool(hz) and all(h["model"]["mae"] <= h["base"]["mae"] for h in hz)
        self.capabilities["motion_model"] = {
            "available": bool(hz), "reason": None if hz else "No hay muestras de validación.",
            "beats_baseline_all_horizons": better}
        self.capabilities["radar_acceleration"] = self.capabilities["acceleration"]
        ok = bool(hz) and bool(self.limits) and bool(len(passes)) and bool(self.route_validation)
        pb_ok = bool(self.completion and self.completion.get("enabled")) and self.yards_model is not None and self.rush_curve is not None
        self.capabilities["playbook"] = {"available": pb_ok, "reason": None if pb_ok else
                                         "La evaluación necesita el modelo de completitud, el de yardas y la curva de presión validados."}
        self.capabilities["route_optimizer"] = {
            "available": ok, "reason": None if ok else "Faltan lanzamientos, límites físicos o validación del modelo defensivo."}


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        import os
        mg = os.environ.get("NFL_MAX_GAMES")
        _store = Store(max_games=int(mg) if mg else None)
    return _store

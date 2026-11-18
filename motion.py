"""Predicción de posiciones futuras.

Modelo base: velocidad constante, x(t+h) = x(t) + vx*h, y(t+h) = y(t) + vy*h, con
(vx, vy) = (s*sin(dir), s*cos(dir)).

Modelo calibrado: regresión ridge (NumPy) por grupo posicional y horizonte que corrige
el desplazamiento del modelo base usando: velocidad, cambio de velocidad en los últimos
0.5 s, velocidad derivada de las posiciones recientes y aceleración reportada. Como `a`
es una magnitud sin signo, su signo se infiere de la tendencia reciente de la rapidez.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import FPS, heading_to_velocity
from .loader import SIDE_BALL
from .play import PlayArrays

HORIZONS = (5, 10, 15, 20, 25, 30)          # frames (0.5 s a 3.0 s)
LAG = 5                                       # historial usado: 0.5 s
FEATURES = ["vx", "vy", "dvx", "dvy", "pvx", "pvy", "ax", "ay"]
GROUPS = ["QB", "OL", "RB", "WR", "TE", "DL", "LB", "DB", "OTHER"]
POS_TO_GROUP = {
    "QB": "QB", "T": "OL", "G": "OL", "C": "OL", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR",
    "TE": "TE", "DE": "DL", "DT": "DL", "NT": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "S": "DB",
}
RIDGE_LAMBDA = 10.0
MIN_GROUP_SAMPLES = 500
STRIDE = 2
ERR_BIN = 0.05
ERR_MAX = 20.0
N_BINS = int(ERR_MAX / ERR_BIN)


def pos_group(position: str | None) -> str:
    return POS_TO_GROUP.get(str(position), "OTHER")


def kinematic_features(pa: PlayArrays) -> np.ndarray:
    """Tensor (E, F, 8) con NaN donde no hay historial suficiente."""
    E, F = pa.x.shape
    out = np.full((E, F, len(FEATURES)), np.nan, dtype=np.float32)
    if F <= LAG:
        return out
    vx, vy = heading_to_velocity(pa.s, pa.dir)
    out[:, :, 0], out[:, :, 1] = vx, vy
    out[:, LAG:, 2] = vx[:, LAG:] - vx[:, :-LAG]
    out[:, LAG:, 3] = vy[:, LAG:] - vy[:, :-LAG]
    out[:, LAG:, 4] = (pa.x[:, LAG:] - pa.x[:, :-LAG]) * FPS / LAG
    out[:, LAG:, 5] = (pa.y[:, LAG:] - pa.y[:, :-LAG]) * FPS / LAG
    trend = np.zeros_like(pa.s)
    trend[:, 3:] = np.sign(pa.s[:, 3:] - pa.s[:, :-3])
    signed_a = np.nan_to_num(pa.a) * trend
    rad = np.radians(pa.dir)
    out[:, :, 6] = signed_a * np.sin(rad)
    out[:, :, 7] = signed_a * np.cos(rad)
    out[:, :LAG, 2:] = np.nan
    return out


def steady_features(speed: float, direction_deg: float) -> np.ndarray:
    """Estado hipotético de movimiento estable (sin aceleración ni giros recientes)."""
    vx, vy = heading_to_velocity(np.float32(speed), np.float32(direction_deg))
    return np.array([vx, vy, 0, 0, vx, vy, 0, 0], dtype=np.float64)


def design(feats: np.ndarray) -> np.ndarray:
    feats = np.atleast_2d(feats).astype(np.float64)
    return np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)


def baseline_disp(feats: np.ndarray, k: int) -> np.ndarray:
    feats = np.atleast_2d(feats)
    return feats[:, :2].astype(np.float64) * (k / FPS)


def build_samples(pa: PlayArrays, groups: np.ndarray, start_idx: int, stride: int = STRIDE):
    """Muestras (features, desplazamientos reales por horizonte, grupo) para una jugada.

    `groups` son códigos de grupo por entidad (índice en GROUPS, -1 para el balón).
    """
    E, F = pa.x.shape
    start = max(start_idx, LAG)
    if start >= F:
        return None
    I = np.arange(start, F, stride)
    ent = np.flatnonzero((pa.side != SIDE_BALL) & (groups >= 0))
    if ent.size == 0 or I.size == 0:
        return None
    feats_all = kinematic_features(pa)
    feats = feats_all[ent][:, I].reshape(-1, len(FEATURES))
    grp = np.repeat(groups[ent], I.size)
    disp = np.full((feats.shape[0], len(HORIZONS), 2), np.nan, dtype=np.float32)
    for hi, k in enumerate(HORIZONS):
        valid = (I + k) < F
        J = np.minimum(I + k, F - 1)
        dx = pa.x[ent][:, J] - pa.x[ent][:, I]
        dy = pa.y[ent][:, J] - pa.y[ent][:, I]
        dx[:, ~valid] = np.nan
        dy[:, ~valid] = np.nan
        disp[:, hi, 0] = dx.reshape(-1)
        disp[:, hi, 1] = dy.reshape(-1)
    ok = np.isfinite(feats).all(axis=1)
    return feats[ok], disp[ok], grp[ok]


@dataclass
class FitAccumulator:
    n_feat: int = len(FEATURES) + 1
    xtx: dict = field(default_factory=dict)
    xty: dict = field(default_factory=dict)
    count: dict = field(default_factory=dict)

    def add(self, feats: np.ndarray, disp: np.ndarray, grp: np.ndarray) -> None:
        X = design(feats)
        for hi, k in enumerate(HORIZONS):
            resid = disp[:, hi, :].astype(np.float64) - baseline_disp(feats, k)
            valid = np.isfinite(resid).all(axis=1)
            for g in list(np.unique(grp[valid])) + [-1]:
                m = valid if g == -1 else valid & (grp == g)
                if not m.any():
                    continue
                key = (int(g), k)
                self.xtx[key] = self.xtx.get(key, 0) + X[m].T @ X[m]
                self.xty[key] = self.xty.get(key, 0) + X[m].T @ resid[m]
                self.count[key] = self.count.get(key, 0) + int(m.sum())


@dataclass
class MotionModel:
    coefs: dict = field(default_factory=dict)   # (grupo, k) -> (9, 2)
    counts: dict = field(default_factory=dict)

    @staticmethod
    def fit(acc: FitAccumulator) -> "MotionModel":
        model = MotionModel(counts=dict(acc.count))
        for key, xtx in acc.xtx.items():
            reg = RIDGE_LAMBDA * np.eye(xtx.shape[0])
            reg[-1, -1] = 0.0  # no se penaliza el intercepto
            model.coefs[key] = np.linalg.solve(xtx + reg, acc.xty[key])
        return model

    def _coef(self, group: int, k: int) -> np.ndarray:
        if self.counts.get((group, k), 0) >= MIN_GROUP_SAMPLES and (group, k) in self.coefs:
            return self.coefs[(group, k)]
        return self.coefs[(-1, k)]

    def predict_disp(self, group: int, k: int, feats: np.ndarray) -> np.ndarray:
        """Desplazamiento (N, 2) entre t y t+k frames para el modelo calibrado."""
        return baseline_disp(feats, k) + design(feats) @ self._coef(group, k)

    def coef_table(self) -> list[dict]:
        rows = []
        for hi, k in enumerate(HORIZONS):
            c = self.coefs.get((-1, k))
            if c is None:
                continue
            rows.append({"horizon_s": k / FPS, "n": self.counts.get((-1, k), 0),
                         "coef_x": {n: round(float(v), 4) for n, v in zip(FEATURES + ["intercepto"], c[:, 0])},
                         "coef_y": {n: round(float(v), 4) for n, v in zip(FEATURES + ["intercepto"], c[:, 1])}})
        return rows


class EvalAccumulator:
    """Acumula errores de validación sin guardar todas las muestras."""

    def __init__(self, seed: int = 7, scatter_per_game: int = 25):
        H = len(HORIZONS)
        self.n = np.zeros(H)
        self.sum = np.zeros((2, H))                 # [base, modelo]
        self.sumsq = np.zeros((2, H))
        self.hist = np.zeros((2, H, N_BINS + 1))
        self.diff_sum = np.zeros(H)
        self.diff_sq = np.zeros(H)
        self.by_group: dict[tuple[int, int], np.ndarray] = {}   # (grupo, hi) -> [n, sum_base, sum_model]
        self.scatter: list[list[dict]] = [[] for _ in range(H)]
        self.rng = np.random.default_rng(seed)
        self.scatter_per_game = scatter_per_game

    def add(self, model: MotionModel, feats, disp, grp) -> None:
        for hi, k in enumerate(HORIZONS):
            real = disp[:, hi, :].astype(np.float64)
            valid = np.isfinite(real).all(axis=1)
            if not valid.any():
                continue
            f, r, g = feats[valid], real[valid], grp[valid]
            pb = baseline_disp(f, k)
            pm = np.empty_like(pb)
            for gi in np.unique(g):
                m = g == gi
                pm[m] = model.predict_disp(int(gi), k, f[m])
            eb = np.hypot(*(pb - r).T)
            em = np.hypot(*(pm - r).T)
            self.n[hi] += eb.size
            for mi, e in enumerate((eb, em)):
                self.sum[mi, hi] += e.sum()
                self.sumsq[mi, hi] += (e ** 2).sum()
                bins = np.minimum((e / ERR_BIN).astype(int), N_BINS)
                self.hist[mi, hi] += np.bincount(bins, minlength=N_BINS + 1)
            d = eb - em
            self.diff_sum[hi] += d.sum()
            self.diff_sq[hi] += (d ** 2).sum()
            for gi in np.unique(g):
                m = g == gi
                acc = self.by_group.setdefault((int(gi), hi), np.zeros(3))
                acc += [m.sum(), eb[m].sum(), em[m].sum()]
            pick = self.rng.choice(eb.size, size=min(self.scatter_per_game, eb.size), replace=False)
            for j in pick:
                self.scatter[hi].append({"real": round(float(np.hypot(*r[j])), 3),
                                         "base": round(float(np.hypot(*pb[j])), 3),
                                         "model": round(float(np.hypot(*pm[j])), 3)})

    @staticmethod
    def _quantile(hist: np.ndarray, q: float) -> float:
        total = hist.sum()
        if total == 0:
            return float("nan")
        cum = np.cumsum(hist)
        i = int(np.searchsorted(cum, q * total))
        return float((min(i, N_BINS - 1) + 0.5) * ERR_BIN)

    def report(self) -> dict:
        horizons = []
        for hi, k in enumerate(HORIZONS):
            n = self.n[hi]
            if n == 0:
                continue
            row = {"horizon_s": k / FPS, "n": int(n)}
            for mi, name in enumerate(("base", "model")):
                mae = self.sum[mi, hi] / n
                rmse = float(np.sqrt(self.sumsq[mi, hi] / n))
                row[name] = {"mae": round(float(mae), 4), "rmse": round(rmse, 4),
                             "median": round(self._quantile(self.hist[mi, hi], 0.5), 3),
                             "p90": round(self._quantile(self.hist[mi, hi], 0.9), 3),
                             "p95": round(self._quantile(self.hist[mi, hi], 0.95), 3)}
            mean_d = self.diff_sum[hi] / n
            var_d = max(self.diff_sq[hi] / n - mean_d ** 2, 0.0)
            se = float(np.sqrt(var_d / n))
            row["improvement"] = {"mae_abs": round(float(mean_d), 4),
                                  "mae_pct": round(float(100 * mean_d / (self.sum[0, hi] / n)), 2),
                                  "ci95": [round(float(mean_d - 1.96 * se), 4), round(float(mean_d + 1.96 * se), 4)]}
            horizons.append(row)
        by_group = []
        for (gi, hi), (n, sb, sm) in sorted(self.by_group.items()):
            by_group.append({"group": GROUPS[gi], "horizon_s": HORIZONS[hi] / FPS, "n": int(n),
                             "mae_base": round(float(sb / n), 4), "mae_model": round(float(sm / n), 4)})
        edges = [round(i * 0.25, 2) for i in range(0, 17)]
        step = int(0.25 / ERR_BIN)
        dist = []
        for hi, k in enumerate(HORIZONS):
            if self.n[hi] == 0:
                continue
            for mi, name in enumerate(("base", "model")):
                h = self.hist[mi, hi]
                counts = [float(h[i * step:(i + 1) * step].sum()) for i in range(16)]
                counts[-1] += float(h[16 * step:].sum())
                dist.append({"horizon_s": k / FPS, "model": name, "edges": edges, "counts": counts})
        return {"horizons": horizons, "by_group": by_group, "distribution": dist,
                "scatter": [{"horizon_s": HORIZONS[i] / FPS, "points": pts} for i, pts in enumerate(self.scatter)]}


def project_path(model: MotionModel, group: int, feats: np.ndarray, x0: float, y0: float, max_k: int) -> dict:
    """Trayectorias proyectadas (base y calibrada) en los horizontes disponibles hasta `max_k`."""
    feats = np.atleast_2d(feats)
    base, fit = [], []
    for k in HORIZONS:
        if k > max_k:
            break
        db = baseline_disp(feats, k)[0]
        dm = model.predict_disp(group, k, feats)[0]
        base.append({"t": k / FPS, "x": float(x0 + db[0]), "y": float(y0 + db[1])})
        fit.append({"t": k / FPS, "x": float(x0 + dm[0]), "y": float(y0 + dm[1])})
    return {"base": base, "model": fit}

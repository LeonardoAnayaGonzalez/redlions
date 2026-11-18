"""Funciones estadísticas implementadas con NumPy/pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_rank(value: float, pool: np.ndarray) -> float:
    """Percentil (0-100) de `value` dentro de `pool` (empates cuentan la mitad)."""
    pool = pool[np.isfinite(pool)]
    if pool.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(100.0 * ((pool < value).sum() + 0.5 * (pool == value).sum()) / pool.size)


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Área bajo la curva ROC (estadístico de Mann-Whitney con rangos promedio)."""
    y = np.asarray(y).astype(int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60) -> np.ndarray:
    """Regresión logística con penalización L2 (Newton-Raphson). X ya incluye la columna de unos al final."""
    beta = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) + reg @ beta
        w = p * (1 - p)
        hess = (X * w[:, None]).T @ X + reg + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    """Calibración por cuantiles de probabilidad predicha."""
    if y.size < bins * 5:
        return []
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"pred": round(float(p[chunk].mean()), 4), "observed": round(float(y[chunk].mean()), 4), "n": int(chunk.size)})
    return out


def histogram(values: np.ndarray, edges: list[float]) -> list[dict]:
    """Histograma con la última barra abierta (>= último borde)."""
    values = values[np.isfinite(values)]
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else np.inf
        n = int(((values >= lo) & (values < hi)).sum())
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
        out.append({"bin": label, "lo": lo, "count": n})
    return out

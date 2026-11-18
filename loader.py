"""Carga de datos desde `data/` (solo lectura) con adaptación de esquema, limpieza y normalización."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import FPS, normalize_orientation
from .schema import SchemaError, resolve

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", ROOT / "data"))

SIDE_OFF, SIDE_DEF, SIDE_BALL = 0, 1, 2


def read_table(filename: str, table: str) -> pd.DataFrame:
    """Lee un CSV de `data/` y devuelve solo columnas conocidas con nombres internos."""
    path = DATA_DIR / filename
    if not path.exists():
        raise SchemaError(f"No existe el archivo requerido: {path}")
    head = pd.read_csv(path, nrows=0)
    mapping = resolve(list(head.columns), table)
    df = pd.read_csv(path, usecols=list(mapping.values()))
    return df.rename(columns={v: k for k, v in mapping.items()})


def tracking_files() -> dict[int, Path]:
    """Mapa game_id -> archivo de tracking (el id sale del nombre y se valida al leer)."""
    out: dict[int, Path] = {}
    folder = DATA_DIR / "tracking"
    if not folder.exists():
        return out
    for p in sorted(folder.glob("*.csv")):
        m = re.search(r"(\d{8,})", p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


def load_game_tracking(path: Path, game_id: int, plays_game: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Lee el tracking de un partido, lo adapta y normaliza (el ataque siempre va hacia +x)."""
    head = pd.read_csv(path, nrows=0)
    mapping = resolve(list(head.columns), "tracking")
    df = pd.read_csv(path, usecols=list(mapping.values()))
    df = df.rename(columns={v: k for k, v in mapping.items()})
    quality = {"rows_raw": int(len(df))}

    if "a" not in df:
        df["a"] = 0.0
    if "o" not in df:
        df["o"] = np.nan
    if "event" not in df:
        df["event"] = np.nan
    if "dis" not in df:
        df["dis"] = np.nan
    foreign = df["game_id"] != game_id
    quality["rows_other_game"] = int(foreign.sum())
    df = df[~foreign]

    ball = df["team"].astype(str).str.lower().eq("football") | df["player_id"].isna()
    df["player_id"] = df["player_id"].fillna(-1).astype("int64")
    possession = plays_game.set_index("play_id")["possession"]
    poss = df["play_id"].map(possession)
    df["side"] = np.where(ball, SIDE_BALL, np.where(df["team"] == poss, SIDE_OFF, SIDE_DEF)).astype("int8")
    df = df[poss.notna() | ball]  # jugadas sin metadatos de posesión no son analizables

    dup = df.duplicated(["play_id", "player_id", "frame"])
    quality["rows_duplicate"] = int(dup.sum())
    df = df[~dup]

    left = (df["direction"].astype(str).str.lower() == "left").to_numpy()
    x, y, d, o = normalize_orientation(
        df["x"].to_numpy(np.float32), df["y"].to_numpy(np.float32),
        df["dir"].to_numpy(np.float32), df["o"].to_numpy(np.float32), left)
    df["x"], df["y"], df["dir"], df["o"] = x.astype(np.float32), y.astype(np.float32), d.astype(np.float32), o.astype(np.float32)
    df["flipped"] = left
    for c in ("s", "a", "dis"):
        df[c] = df[c].astype(np.float32)
    df = df.sort_values(["play_id", "player_id", "frame"], kind="stable").reset_index(drop=True)
    if df["dis"].isna().all():  # reconstruir la distancia por frame si no viene en el archivo
        step = np.hypot(df.groupby(["play_id", "player_id"])["x"].diff(), df.groupby(["play_id", "player_id"])["y"].diff())
        df["dis"] = step.fillna(0).astype(np.float32)
    quality["nan_dir_players"] = int(df.loc[df["side"] != SIDE_BALL, "dir"].isna().sum())
    return df, quality


def seconds(frames: int | float) -> float:
    return float(frames) / FPS

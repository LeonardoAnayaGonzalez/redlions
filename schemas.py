"""Modelos Pydantic de entrada."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .motion import HORIZONS
from .geometry import FPS

ALLOWED_HORIZONS = [k / FPS for k in HORIZONS]


def _check_horizon(v: float) -> float:
    if not any(abs(v - h) < 1e-6 for h in ALLOWED_HORIZONS):
        raise ValueError(f"horizon_s debe ser uno de {ALLOWED_HORIZONS}")
    return v


class PredictRequest(BaseModel):
    game_id: int
    play_id: int
    player_id: int
    frame_id: int = Field(ge=1, description="frameId del instante observado")
    horizon_s: float = Field(description="Horizonte de predicción en segundos")
    reveal: bool = Field(False, description="Si es true, devuelve la posición real posterior y el error")

    _h = field_validator("horizon_s")(_check_horizon)


ROUTE_HORIZONS = [1.0, 1.5, 2.0, 2.5, 3.0]


class RouteRequest(BaseModel):
    game_id: int
    play_id: int
    player_id: int
    frame_id: int = Field(ge=1, description="frameId del instante previo al lanzamiento")
    horizon_s: float = Field(2.0, description="Duración simulada de la ruta (s)")
    defense_model: Literal["reactive", "inertial"] = "reactive"
    reveal: bool = Field(False, description="Si es true, añade un control con posiciones reales; no cambia la selección")

    @field_validator("horizon_s")
    @classmethod
    def _route_horizon(cls, v: float) -> float:
        if not any(abs(v - h) < 1e-6 for h in ROUTE_HORIZONS):
            raise ValueError(f"horizon_s debe ser uno de {ROUTE_HORIZONS}")
        return v
class SimulateRequest(BaseModel):
    game_id: int
    play_id: int
    player_id: int
    frame_id: int = Field(ge=1)
    dx: float = Field(0.0, ge=-30, le=30, description="Desplazamiento inicial en x (yardas, +x = hacia la zona rival)")
    dy: float = Field(0.0, ge=-27, le=27, description="Desplazamiento inicial en y (yardas, +y = banda superior)")
    speed: float | None = Field(None, ge=0, le=12, description="Velocidad hipotética (yd/s). Máximo observado: ~9.6")
    direction_deg: float | None = Field(None, ge=0, le=360, description="Dirección hipotética (0 = banda superior, 90 = zona rival)")
    horizon_s: float = 2.0

    _h = field_validator("horizon_s")(_check_horizon)



class PBPlayer(BaseModel):
    slot: str = Field(min_length=1, max_length=12)
    pos: Literal["QB", "OL", "WR", "TE", "RB"]
    x: float = Field(ge=0, le=120)
    y: float = Field(ge=0, le=53.34)
    route: list[tuple[float, float]] = Field(default_factory=list, max_length=8)
    route_name: str | None = Field(None, max_length=40)
    settle: bool = False
    name: str | None = Field(None, max_length=60)


class PBWeights(BaseModel):
    w1: float = Field(1.0, ge=0, le=2)
    w2: float = Field(0.5, ge=0, le=2)
    w3: float = Field(0.25, ge=0, le=2)
    fd_bonus: float = Field(2.0, ge=0, le=10, description="Yardas extra por alcanzar el primer down")


class PBEvaluateRequest(BaseModel):
    game_id: int
    play_id: int
    players: list[PBPlayer] = Field(min_length=1, max_length=11)
    weights: PBWeights = Field(default_factory=PBWeights)


class PBGenerateRequest(PBEvaluateRequest):
    n_samples: int = Field(300, ge=50, le=1500)
    locked: list[str] = Field(default_factory=list, max_length=11)
    seed: int = Field(7, ge=0, le=10_000)

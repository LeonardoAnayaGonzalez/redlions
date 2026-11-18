"""Geometría y unidades del campo.

Unidades del dataset: posiciones en yardas (x: 0-120 incluye zonas de anotación,
y: 0-53.33), velocidad en yardas/segundo, aceleración en yardas/segundo^2,
ángulos en grados. 10 frames por segundo.
Convención de `dir` (verificada con los datos): 0 grados = +y, sentido horario,
es decir vx = s*sin(dir) y vy = s*cos(dir).
"""
from __future__ import annotations

import numpy as np

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0
FPS = 10.0


def heading_to_velocity(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.radians(direction_deg)
    return speed * np.sin(rad), speed * np.cos(rad)


def velocity_to_heading(vx: float, vy: float) -> tuple[float, float]:
    """(velocidad, dirección en grados) a partir de componentes."""
    speed = float(np.hypot(vx, vy))
    return speed, float(np.degrees(np.arctan2(vx, vy)) % 360.0)


def normalize_orientation(x, y, direction, orientation, left_mask):
    """Hace que el ataque vaya siempre hacia +x. Gira 180 grados las jugadas hacia la izquierda."""
    x = np.where(left_mask, FIELD_LENGTH - x, x)
    y = np.where(left_mask, FIELD_WIDTH - y, y)
    direction = np.where(left_mask, (direction + 180.0) % 360.0, direction)
    orientation = np.where(left_mask, (orientation + 180.0) % 360.0, orientation)
    return x, y, direction, orientation

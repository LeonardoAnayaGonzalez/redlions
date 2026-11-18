"""Capa de adaptación: traduce los nombres reales de columnas a nombres internos.

No se asume que las columnas existan: se buscan por alias (insensible a mayúsculas
y guiones bajos) y se valida que estén las obligatorias. Lo faltante se reporta
como capacidad no disponible con su razón.
"""
from __future__ import annotations

import re

# nombre interno -> alias posibles en el archivo
ALIASES: dict[str, dict[str, list[str]]] = {
    "tracking": {
        "game_id": ["gameId"], "play_id": ["playId"], "player_id": ["nflId", "playerId"],
        "frame": ["frameId", "frame"], "x": ["x"], "y": ["y"], "s": ["s", "speed"],
        "a": ["a", "acceleration"], "dis": ["dis", "distance"], "o": ["o", "orientation"],
        "dir": ["dir", "direction"], "event": ["event"], "team": ["team", "club"],
        "direction": ["playDirection"], "jersey": ["jerseyNumber"],
    },
    "plays": {
        "game_id": ["gameId"], "play_id": ["playId"], "description": ["playDescription"],
        "quarter": ["quarter"], "down": ["down"], "yards_to_go": ["yardsToGo"],
        "possession": ["possessionTeam"], "defense": ["defensiveTeam"],
        "yardline_side": ["yardlineSide"], "yardline_number": ["yardlineNumber"],
        "game_clock": ["gameClock"], "pre_home": ["preSnapHomeScore"],
        "pre_away": ["preSnapVisitorScore"], "pass_result": ["passResult"],
        "play_result": ["playResult"], "absolute_yardline": ["absoluteYardlineNumber"],
        "formation": ["offenseFormation"], "personnel_o": ["personnelO"],
        "defenders_box": ["defendersInBox"], "personnel_d": ["personnelD"],
        "dropback": ["dropBackType"], "play_action": ["pff_playAction"],
        "coverage": ["pff_passCoverage"], "coverage_type": ["pff_passCoverageType"],
    },
    "games": {
        "game_id": ["gameId"], "season": ["season"], "week": ["week"], "date": ["gameDate"],
        "time": ["gameTimeEastern"], "home": ["homeTeamAbbr"], "away": ["visitorTeamAbbr"],
    },
    "players": {
        "player_id": ["nflId"], "height": ["height"], "weight": ["weight"],
        "birth": ["birthDate"], "college": ["collegeName"], "position": ["officialPosition"],
        "name": ["displayName"],
    },
    "scouting": {
        "game_id": ["gameId"], "play_id": ["playId"], "player_id": ["nflId"],
        "role": ["pff_role"], "lined_up": ["pff_positionLinedUp"],
    },
}

REQUIRED: dict[str, list[str]] = {
    "tracking": ["game_id", "play_id", "player_id", "frame", "x", "y", "s", "dir", "team", "direction"],
    "plays": ["game_id", "play_id", "possession"],
    "games": ["game_id"],
    "players": ["player_id", "name"],
    "scouting": ["game_id", "play_id", "player_id", "role"],
}


class SchemaError(Exception):
    """Falta una columna obligatoria en un archivo de datos."""


def _key(name: str) -> str:
    return re.sub(r"[_\s]", "", name).lower()


def resolve(columns: list[str], table: str) -> dict[str, str]:
    """Devuelve {nombre_interno: nombre_real} para las columnas encontradas."""
    by_key = {_key(c): c for c in columns}
    mapping: dict[str, str] = {}
    for internal, aliases in ALIASES[table].items():
        for alias in aliases:
            if _key(alias) in by_key:
                mapping[internal] = by_key[_key(alias)]
                break
    missing = [c for c in REQUIRED[table] if c not in mapping]
    if missing:
        raise SchemaError(f"Tabla '{table}': faltan columnas obligatorias {missing}. Columnas reales: {list(columns)}")
    return mapping

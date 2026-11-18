"""API de NFL score analysis. Ejecuta: .venv\\Scripts\\uvicorn backend.main:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from . import playbook_service as pbsvc
from . import route_service as rsvc
from . import services as svc
from .loader import ROOT
from .schemas import PBEvaluateRequest, PBGenerateRequest, PredictRequest, RouteRequest, SimulateRequest
from .store import get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    try:
        store.status.update(state="loading", message="Leyendo tablas")
        store.load_static()
        store.start_background()
    except Exception as exc:  # noqa: BLE001 - el estado de error se expone en /api/status
        store.status.update(state="error", message=f"{type(exc).__name__}: {exc}")
    yield


app = FastAPI(title="NFL score analysis", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    s = get_store()
    return svc.clean({**s.status, "capabilities": s.capabilities})


@app.get("/api/summary")
def summary(game_id: int | None = None, team: str | None = None, player_id: int | None = None,
            play_type: str | None = None, pass_result: str | None = None) -> dict:
    return svc.summary(get_store(), game_id, team, player_id, play_type, pass_result)


@app.get("/api/games")
def games() -> dict:
    s = get_store()
    if s.games is None:
        raise HTTPException(503, "Las tablas aún no se han cargado.")
    return {"items": svc.games_list(s), "teams": svc.teams(s)}


@app.get("/api/plays")
def plays(game_id: int | None = None, team: str | None = None, pass_result: str | None = None,
          play_type: str | None = None, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)) -> dict:
    return svc.plays_list(get_store(), game_id, team, pass_result, play_type, limit, offset)


@app.get("/api/plays/{game_id}/{play_id}")
def play(game_id: int, play_id: int) -> dict:
    return svc.play_detail(get_store(), game_id, play_id)


@app.get("/api/plays/{game_id}/{play_id}/narrative")
def play_narrative(game_id: int, play_id: int, player_id: int | None = None) -> dict:
    return svc.narrative(get_store(), game_id, play_id, player_id)


@app.get("/api/plays/{game_id}/{play_id}/pass")
def play_pass(game_id: int, play_id: int) -> dict:
    return svc.pass_analysis(get_store(), game_id, play_id)


@app.get("/api/players")
def players(q: str | None = None, position: str | None = None, team: str | None = None,
            min_plays: int = Query(20, ge=1), limit: int = Query(50, ge=1, le=500),
            sort: Literal["composite", "speed", "max_speed", "acceleration", "distance", "plays", "mobility"] = "composite") -> dict:
    return svc.players_list(get_store(), q, position, team, min_plays, sort, limit)


@app.get("/api/players/compare")
def players_compare(a: int, b: int) -> dict:
    return svc.players_compare(get_store(), a, b)


@app.get("/api/players/{player_id}")
def player(player_id: int) -> dict:
    return svc.player_detail(get_store(), player_id)


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    return svc.predict(get_store(), req.game_id, req.play_id, req.player_id, req.frame_id, req.horizon_s, req.reveal)


@app.post("/api/simulate")
def simulate(req: SimulateRequest) -> dict:
    return svc.simulate(get_store(), req.game_id, req.play_id, req.player_id, req.frame_id, req.dx, req.dy,
                        req.speed, req.direction_deg, req.horizon_s)


@app.get("/api/routes/info")
def routes_info() -> dict:
    return rsvc.route_info(get_store())


@app.post("/api/routes/optimize")
def routes_optimize(req: RouteRequest) -> dict:
    return rsvc.route_optimize(get_store(), req.game_id, req.play_id, req.player_id, req.frame_id, req.horizon_s,
                              req.defense_model, req.reveal)


@app.get("/api/playbook/meta")
def playbook_meta() -> dict:
    return pbsvc.meta(get_store())


@app.get("/api/playbook/scenario/{game_id}/{play_id}")
def playbook_scenario(game_id: int, play_id: int) -> dict:
    return pbsvc.scenario(get_store(), game_id, play_id)


def _pb_players(req: PBEvaluateRequest) -> list[dict]:
    return [{**p.model_dump(), "route": [list(w) for w in p.route]} for p in req.players]


@app.post("/api/playbook/evaluate")
def playbook_evaluate(req: PBEvaluateRequest) -> dict:
    return pbsvc.evaluate(get_store(), req.game_id, req.play_id, _pb_players(req), req.weights.model_dump())


@app.post("/api/playbook/generate")
def playbook_generate(req: PBGenerateRequest) -> dict:
    return pbsvc.generate(get_store(), req.game_id, req.play_id, _pb_players(req), req.weights.model_dump(),
                          req.n_samples, req.locked, req.seed)


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    return svc.model_metrics(get_store())


@app.get("/api/demo")
def demo() -> dict:
    return svc.demo(get_store())


_dist = ROOT / "dashboard" / "dist"
if _dist.exists():  # el mismo servidor entrega el build del dashboard
    app.mount("/", StaticFiles(directory=_dist, html=True), name="dashboard")

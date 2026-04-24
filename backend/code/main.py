from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import logging
import time

logging.basicConfig(level = logging.INFO, format = "%(asctime)s  %(levelname)-8s  %(message)s", datefmt = "%H:%M:%S",)
log = logging.getLogger("playcaller")

from points import PointsPredictor
from injury import InjuryRiskPredictor
from trade  import TradeEvaluator

app = FastAPI(title = "PlayCaller API", version = "1.0.0",)

app.add_middleware(CORSMiddleware, allow_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ], allow_credentials = True, allow_methods = ["*"], allow_headers = ["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()

    body = await request.body()
    if body:
        log.info("→ %s %s  body: %s", request.method, request.url.path, body.decode())
    else:
        log.info("→ %s %s", request.method, request.url.path)

    async def receive():
        return {"type": "http.request", "body": body}
    request._receive = receive

    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    log.info("← %s %s  status=%d  %.1fms",
             request.method, request.url.path, response.status_code, elapsed)
    return response

fantasy_model = PointsPredictor()
injury_model  = InjuryRiskPredictor()
trade_model   = TradeEvaluator()

@app.on_event("startup")
def load_all_models():
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BACKEND_DIR)
    MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
    DATA_PATH = os.path.join(BACKEND_DIR, "final_weekly_stats.csv")

    if not os.path.isdir(MODEL_DIR):
        MODEL_DIR = os.path.join(BACKEND_DIR, "models")
    if not os.path.isfile(DATA_PATH):
        DATA_PATH = os.path.join(BACKEND_DIR, "weekly_stats_enhanced.csv")

    log.info("MODEL_DIR : %s", MODEL_DIR)
    log.info("DATA_PATH : %s", DATA_PATH)

    if not os.path.isdir(MODEL_DIR):
        log.warning("models/ not found — run points.py / injury.py / trade.py first")
        return

    try:
        fantasy_model.load_model(MODEL_DIR)
    except Exception as e:
        log.error(e)

    try:
        injury_model.load_model(MODEL_DIR)
    except Exception as e:
        log.error(e)

    try:
        trade_model.load(model_dir = MODEL_DIR, data_path = DATA_PATH)
    except Exception as e:
        log.error(e)

    log.info("API: http://localhost:8000 Docs: http://localhost:8000/docs")


class PlayerRequest(BaseModel):
    player_name: str
    season:  Optional[int] = None
    week:    Optional[int] = None

    def clean_season(self): return self.season if self.season else None
    def clean_week(self):   return self.week   if self.week   else None


class TradeRequest(BaseModel):
    giving:    List[str]
    receiving: List[str]

@app.get("/health")
def health():
    return {"status": "ok", "models": "loaded"}


@app.post("/predict")
def predict(req: PlayerRequest):
    log.info("[predict] player_name=%r  season=%s  week=%s",
             req.player_name, req.season, req.week)

    if fantasy_model.model is None:
        raise HTTPException(status_code = 503, detail = "Fantasy model not loaded yet.")

    result = fantasy_model.predict_for_player(
        req.player_name, season = req.clean_season(), week = req.clean_week()
    )
    if result is None:
        log.warning("[predict] NOT FOUND: %r", req.player_name)
        raise HTTPException(status_code = 404,
                            detail = f"Player '{req.player_name}' not found.")

    log.info("[predict] → %s  predicted_points=%s  range=%s  source=ML_MODEL",
             result["player"], result["predicted_points"], result["range"])

    result["_source"] = "ml_model"
    return result


@app.post("/injury")
def injury(req: PlayerRequest):
    log.info("[injury] player_name=%r", req.player_name)

    if injury_model.gb_model is None:
        raise HTTPException(status_code = 503, detail = "Injury model not loaded yet.")

    result = injury_model.predict_for_player(req.player_name, season = req.clean_season(), week = req.clean_week())
    if result is None or "error" in result:
        log.warning("[injury] NOT FOUND: %r", req.player_name)
        raise HTTPException(status_code = 404,
                            detail = f"Player '{req.player_name}' not found.")

    log.info("[injury] → %s  risk_tier=%s  risk_pct=%s  source=ML_MODEL",
             req.player_name, result["risk_tier"], result["risk_pct"])

    result["_source"] = "ml_model"
    return result


@app.post("/trade/evaluate")
def evaluate_trade(req: TradeRequest):
    log.info("[trade] giving=%s  receiving=%s", req.giving, req.receiving)

    if not req.giving or not req.receiving:
        raise HTTPException(status_code = 400, detail = "Both sides need at least one player.")
    if trade_model.profiler is None:
        raise HTTPException(status_code = 503, detail = "Trade model not loaded yet.")

    result = trade_model.evaluate(req.giving, req.receiving)

    log.info("[trade] → verdict=%s  net_value_swing=%s  source=ML_MODEL",
             result.get("verdict"), result.get("net_value_swing"))

    result["_source"] = "ml_model"
    return result


@app.get("/players/search/{query}")
def search_players(query: str, limit: int = 10):
    if trade_model.profiler is None:
        raise HTTPException(status_code = 503, detail = "Models still loading.")

    df = trade_model.profiler.df
    matches = (df[df["player_display_name"].str.contains(query, case = False, na = False)]["player_display_name"].unique().tolist())
    results = sorted(matches)[:limit]
    log.info("[search] query=%r  hits=%d", query, len(results))
    return {"results": results}


@app.post("/player/full")
def player_full(req: PlayerRequest):
    log.info("[player/full] player_name=%r", req.player_name)

    if fantasy_model.model is None:
        raise HTTPException(status_code = 503, detail = "Fantasy model not loaded.")

    projection  = fantasy_model.predict_for_player(req.player_name, season = req.clean_season(), week = req.clean_week())
    injury_risk = None

    if injury_model.gb_model is not None:
        r = injury_model.predict_for_player(req.player_name)
        if r and "error" not in r:
            injury_risk = r

    if projection is None:
        log.warning("[player/full] NOT FOUND: %r", req.player_name)
        raise HTTPException(status_code = 404, detail = f"Player '{req.player_name}' not found.")

    projection["_source"]  = "ml_model"
    if injury_risk:
        injury_risk["_source"] = "ml_model"

    log.info("[player/full] → %s  pts=%s  injury=%s  source=ML_MODEL",
             projection["player"],
             projection["predicted_points"],
             injury_risk["risk_tier"] if injury_risk else "n/a")

    return {"projection": projection, "injury": injury_risk}

if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 8000, reload = True)
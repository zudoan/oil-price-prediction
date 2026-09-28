# app/backend/main.py
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.backend.config import (
    STATIC_DIR, TEMPLATES_DIR, APP_NAME, VERSION, HOST, PORT
)
from app.backend.schemas import (
    MarketOverviewResponse, NextDayForecastResponse, SimulateRequest, SimulateResponse
)
from app.backend.engine import PetroleumEngine

engine: PetroleumEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print(f"🚀 Khởi động máy chủ {APP_NAME} v{VERSION}...")
    engine = PetroleumEngine.get_instance()
    yield
    print("🛑 Đóng máy chủ hoàn tất.")

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="Hệ thống Trí tuệ Nhân tạo Dự báo Giá Xăng dầu Singapore (Singapore MoPS Energy Intelligence)",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ==============================================================================
# UI ROUTES
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": APP_NAME,
            "version": VERSION,
            "hardware": engine.hardware_info if engine else "NVIDIA GPU RTX 4060 Ti"
        }
    )

# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": VERSION,
        "hardware": engine.hardware_info,
        "records": len(engine.df_feat),
        "models_active": ["Residual_GRU_final.keras", "Residual_LSTM_final.keras"]
    }

@app.get("/api/overview")
async def get_overview():
    return engine.get_market_overview()

@app.get("/api/forecast/latest")
async def get_latest_forecast():
    return engine.predict_next_day()

@app.post("/api/forecast/simulate")
async def simulate_forecast(req: SimulateRequest):
    return engine.simulate_scenario(
        shock_gasoline_pct=req.shock_gasoline_pct,
        shock_diesel_pct=req.shock_diesel_pct,
        vol_multiplier=req.volatility_multiplier
    )

@app.get("/api/historical")
async def get_historical(limit: int = Query(180, ge=1, le=1000)):
    return engine.get_historical_series(limit=limit)

@app.get("/api/crack-spreads")
async def get_crack_spreads(limit: int = Query(180, ge=1, le=1000)):
    return engine.get_crack_spreads(limit=limit)

@app.get("/api/metrics")
async def get_metrics():
    return engine.get_metrics_comparison()

@app.get("/api/vietnam/forecast")
async def get_vietnam_forecast(
    fx_rate: float = Query(25400.0, ge=20000.0, le=35000.0, description="Tỷ giá USD/VND"),
    env_tax: float = Query(None, description="Thuế bảo vệ môi trường điều chỉnh (VND)")
):
    return engine.get_vietnam_forecast(fx_rate=fx_rate, env_tax_override=env_tax)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=PORT, reload=True)

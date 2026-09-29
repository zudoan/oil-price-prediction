# app/backend/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ProductOverview(BaseModel):
    product: str
    label: str
    current_price: float
    predicted_next_price: float
    delta_predicted: float
    delta_pct: float
    trend: str  # "bullish", "bearish", "neutral"
    mae: float
    mape: float
    r2: float
    sparkline: List[float]

class MarketOverviewResponse(BaseModel):
    last_updated: str
    market_status: str
    total_records: int
    system_engine: str
    hardware: str
    products: Dict[str, ProductOverview]

class NextDayPredictionItem(BaseModel):
    product: str
    label: str
    current_price: float
    predicted_price: float
    delta: float
    delta_pct: float
    confidence_lower: float
    confidence_upper: float
    signal: str
    confidence_score: float

class NextDayForecastResponse(BaseModel):
    prediction_date: str
    model_used: str
    latency_ms: float
    predictions: List[NextDayPredictionItem]

class TrajectoryPoint(BaseModel):
    day_index: int
    date: str
    prices: Dict[str, float]
    lower_bounds: Dict[str, float]
    upper_bounds: Dict[str, float]

class MultiHorizonItem(BaseModel):
    product: str
    label: str
    current_price: float
    predicted_price: float
    delta: float
    delta_pct: float
    confidence_lower: float
    confidence_upper: float
    signal: str
    confidence_score: float
    mae: float
    mape: float
    r2: float

class MultiHorizonForecastResponse(BaseModel):
    horizon: int
    horizon_label: str
    target_date: str
    business_context: str
    model_used: str
    latency_ms: float
    predictions: List[MultiHorizonItem]
    trajectory: List[TrajectoryPoint]

class VietnamForecastItem(BaseModel):
    product_code: str
    vietnam_name: str
    singapore_usd_bbl: float
    singapore_predicted_usd_bbl: float
    current_retail_vnd: float
    predicted_retail_vnd: float
    delta_vnd: float
    cycle_delta_vnd: float
    cycle_signal: str
    action_class: str
    components: Dict[str, float]
    sparkline_vnd: List[float]

class VietnamForecastResponse(BaseModel):
    horizon: int
    horizon_label: str
    last_date: str
    next_adjustment_date: str
    usd_vnd_rate: float
    products: List[VietnamForecastItem]
    executive_summary: str

class HistoricalPoint(BaseModel):
    date: str
    actual_MG95: Optional[float] = None
    pred_MG95: Optional[float] = None
    actual_MG92: Optional[float] = None
    pred_MG92: Optional[float] = None
    actual_DO_0001: Optional[float] = None
    pred_DO_0001: Optional[float] = None
    actual_DO_005: Optional[float] = None
    pred_DO_005: Optional[float] = None

class CrackSpreadPoint(BaseModel):
    date: str
    spread_MG95_MG92: float
    spread_DO0001_DO005: float
    spread_GAS_OIL: float
    zscore_premium: float
    zscore_quality: float
    zscore_crack: float

class ModelMetricItem(BaseModel):
    model: str
    product: str
    mae: float
    rmse: float
    r2: float
    mape: float

class MetricsSummaryResponse(BaseModel):
    models_comparison: List[Dict[str, Any]]
    detailed_metrics: List[ModelMetricItem]
    multi_horizon_metrics: Optional[List[Dict[str, Any]]] = None

class SimulateRequest(BaseModel):
    shock_gasoline_pct: float = Field(0.0, ge=-20.0, le=20.0)
    shock_diesel_pct: float = Field(0.0, ge=-20.0, le=20.0)
    volatility_multiplier: float = Field(1.0, ge=0.5, le=3.0)

class SimulateResponse(BaseModel):
    scenario: str
    predictions: List[NextDayPredictionItem]

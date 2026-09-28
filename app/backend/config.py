# app/backend/config.py
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "app" / "frontend" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "frontend" / "templates"

# Files
DATA_FILE = DATA_DIR / "price_petroleum.xlsx"
SCALER_X_FILE = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_FILE = MODELS_DIR / "scaler_y.pkl"
MODEL_GRU_FILE = MODELS_DIR / "Residual_GRU_final.keras"
MODEL_LSTM_FILE = MODELS_DIR / "Residual_LSTM_final.keras"

# Server Settings
HOST = "0.0.0.0"
PORT = 8000
APP_NAME = "PetroForecast AI · Singapore Energy Intelligence"
VERSION = "2.0.0"

# Target Columns
TARGET_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
PRODUCT_LABELS = {
    'MG95': 'MOGAS 95 Unleaded (Gasoline)',
    'MG92': 'MOGAS 92 Unleaded (Gasoline)',
    'DO_0001': 'Gasoil 10ppm (Diesel 0.001%)',
    'DO_005': 'Gasoil 500ppm (Diesel 0.05%)'
}

VIETNAM_PRODUCT_LABELS = {
    'MG95': 'Xăng RON 95-III (Cao cấp)',
    'MG92': 'Xăng E5 RON 92-II (Sinh học)',
    'DO_0001': 'Dầu Diesel 0.001S-V (Euro 5)',
    'DO_005': 'Dầu Diesel 0.05S-II (Tiêu chuẩn)'
}

DEFAULT_USD_VND = 25400.0
BARREL_TO_LITER = 158.9873
LOOKBACK = 15

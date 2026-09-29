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
MODEL_MULTI_HORIZON_FILE = MODELS_DIR / "Residual_MultiHorizon_final.keras"

# Server Settings
HOST = "0.0.0.0"
PORT = 8000
APP_NAME = "PetroForecast AI · Singapore & Vietnam Energy Intelligence"
VERSION = "2.1.0"

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
MAX_HORIZON = 20
SUPPORTED_HORIZONS = [1, 3, 7, 20]

HORIZON_CONFIGS = {
    1: {
        "days": 1,
        "label": "T+1 (1 Ngày)",
        "name": "Phiên Kế Tiếp",
        "desc": "Dự báo giá chốt phiên ngày mai cho giao dịch và khớp lệnh",
        "badge": "1 Ngày · Intraday"
    },
    3: {
        "days": 3,
        "label": "T+3 (3 Ngày)",
        "name": "Lướt Sóng Ngắn Hạn",
        "desc": "Chiến lược quản trị vị thế và hedging chu kỳ thanh toán T+3",
        "badge": "3 Ngày · Short Swing"
    },
    7: {
        "days": 7,
        "label": "T+7 (7 Ngày)",
        "name": "Kỳ Điều Hành Liên Bộ",
        "desc": "Dự báo định lượng trước kỳ điều chỉnh giá xăng dầu Thứ Năm hàng tuần (NĐ 80/2023)",
        "badge": "7 Ngày · Kỳ Điều Hành"
    },
    20: {
        "days": 20,
        "label": "T+20 (20 Ngày)",
        "name": "Chu Kỳ Tháng",
        "desc": "Hoạch định ngân sách 1 tháng giao dịch và chiến lược dự trữ tồn kho",
        "badge": "20 Ngày · Chu Kỳ Tháng"
    }
}

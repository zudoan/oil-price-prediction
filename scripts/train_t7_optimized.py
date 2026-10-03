# scripts/train_t7_optimized.py
"""
Tối Ưu Hóa Chuyên Biệt Mốc T+7 — PetroForecast AI v2.1
=========================================================
Baseline (GRU+LSTM Ensemble): T+7 R²≈0.7964, MAPE≈5.01%
Mục tiêu: R² ≥ 0.87, MAPE ≤ 3.5%

Kỹ thuật áp dụng:
  1. Weighted Horizon Loss: Tăng trọng số x3 cho h=5..9 (vùng T+7)
  2. BiGRU + Self-Attention: Kiến trúc sâu hơn để nắm chuỗi dài
  3. Lookback mở rộng: 30 ngày (gấp đôi) để bắt xu hướng tuần
  4. T+7 Refinement Head: Nhánh tinh chỉnh riêng biệt cho mốc 7 ngày
  5. 3-model Ensemble: GRU nặng + LSTM nặng + Lightweight (tránh overfitting)
  6. Domain Feature: Days_To_Thursday (khoảng cách đến kỳ điều hành)
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

import torch
import keras
from keras import layers, models, callbacks, ops

# ─────────────────── Cấu hình ───────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_FILE   = BASE_DIR / "data" / "price_petroleum.xlsx"
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
PRODUCT_LABELS = {
    'MG95':    'MOGAS 95 Unleaded',
    'MG92':    'MOGAS 92 Unleaded',
    'DO_0001': 'Gasoil 10ppm (DO 0.001%)',
    'DO_005':  'Gasoil 500ppm (DO 0.05%)'
}

LOOKBACK     = 30        # tang tu 15 -> 30
MAX_HORIZON  = 20
EVAL_HORIZONS= [1, 3, 7, 20]

# Horizon loss weights (T+7 region amplified)
def build_horizon_weights(horizon=MAX_HORIZON):
    w = np.ones(horizon, dtype=np.float32)
    for h in range(1, horizon + 1):
        if h == 7:
            w[h-1] = 4.0
        elif h in [5, 6, 8, 9]:
            w[h-1] = 3.0
        elif h in [4, 10]:
            w[h-1] = 2.0
        elif h in [3]:
            w[h-1] = 1.5
    return w

HORIZON_WEIGHTS = build_horizon_weights()

# ─────────────────── Custom Weighted Huber Loss ───────────────────
def weighted_huber_loss(horizon_weights):
    hw = horizon_weights.copy()

    def loss_fn(y_true, y_pred):
        delta = 1.0
        err = y_true - y_pred
        abs_err = ops.abs(err)
        huber = ops.where(
            abs_err <= delta,
            0.5 * err ** 2,
            delta * (abs_err - 0.5 * delta)
        )
        w = ops.convert_to_tensor(
            hw.reshape(1, -1, 1), dtype='float32'
        )
        weighted = huber * w
        return ops.mean(weighted)

    loss_fn.__name__ = 'weighted_huber_loss'
    return loss_fn

# ─────────────────── Data Loading ───────────────────
def load_and_preprocess_data():
    print(f"[DATA] Loading {DATA_FILE}...")
    df_raw = pd.read_excel(DATA_FILE)
    df = df_raw.iloc[:, [3, 4, 5, 6, 7]].copy()
    df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']

    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True)
    df[TARGET_COLS] = df[TARGET_COLS].ffill().bfill()

    df_feat = df.copy()

    # Spread Features
    df_feat['SPREAD_MG95_MG92']    = df_feat['MG95']    - df_feat['MG92']
    df_feat['SPREAD_DO0001_DO005'] = df_feat['DO_0001'] - df_feat['DO_005']
    df_feat['SPREAD_GAS_OIL']      = df_feat['MG95']    - df_feat['DO_005']

    for sp in ['SPREAD_MG95_MG92', 'SPREAD_DO0001_DO005', 'SPREAD_GAS_OIL']:
        roll_mean = df_feat[sp].rolling(20, min_periods=5).mean()
        roll_std  = df_feat[sp].rolling(20, min_periods=5).std().replace(0, 1e-6)
        df_feat[f'{sp}_Z20'] = (df_feat[sp] - roll_mean) / roll_std

    # Per-product Technical Indicators
    for col in TARGET_COLS:
        df_feat[f'{col}_EMA5']  = df_feat[col].ewm(span=5,  adjust=False).mean()
        df_feat[f'{col}_EMA10'] = df_feat[col].ewm(span=10, adjust=False).mean()
        df_feat[f'{col}_EMA20'] = df_feat[col].ewm(span=20, adjust=False).mean()

        sma20 = df_feat[col].rolling(20, min_periods=5).mean()
        std20 = df_feat[col].rolling(20, min_periods=5).std().replace(0, 1e-6)
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df_feat[f'{col}_BB_Width'] = (bb_upper - bb_lower) / sma20.replace(0, 1e-6)
        df_feat[f'{col}_BB_PctB']  = (df_feat[col] - bb_lower) / (bb_upper - bb_lower).replace(0, 1e-6)

        df_feat[f'{col}_LogRet'] = np.log(df_feat[col] / df_feat[col].shift(1)).fillna(0)
        df_feat[f'{col}_Ret5d']  = (df_feat[col] - df_feat[col].shift(5)) / df_feat[col].shift(5).replace(0, 1e-6)
        df_feat[f'{col}_Ret7d']  = (df_feat[col] - df_feat[col].shift(7)) / df_feat[col].shift(7).replace(0, 1e-6)

        delta = df_feat[col].diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().replace(0, 1e-6)
        rs    = gain / loss
        df_feat[f'{col}_RSI14'] = 100 - (100 / (1 + rs))

        ema12 = df_feat[col].ewm(span=12, adjust=False).mean()
        ema26 = df_feat[col].ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_feat[f'{col}_MACD']     = macd_line
        df_feat[f'{col}_MACD_Sig'] = signal_line

    # Calendar Features
    df_feat['Dow_Sin']   = np.sin(2 * np.pi * df_feat['Date'].dt.dayofweek / 5.0)
    df_feat['Dow_Cos']   = np.cos(2 * np.pi * df_feat['Date'].dt.dayofweek / 5.0)
    df_feat['Month_Sin'] = np.sin(2 * np.pi * df_feat['Date'].dt.month / 12.0)
    df_feat['Month_Cos'] = np.cos(2 * np.pi * df_feat['Date'].dt.month / 12.0)

    # Domain Feature: khoang cach den Thu Nam (ky dieu hanh gia xang dau VN)
    dow = df_feat['Date'].dt.dayofweek
    days_to_thu = (3 - dow) % 7
    days_to_thu = days_to_thu.replace(0, 7)
    df_feat['Days_To_Thursday'] = days_to_thu / 7.0

    df_feat = df_feat.dropna().reset_index(drop=True)
    other_cols   = [c for c in df_feat.columns if c not in TARGET_COLS and c != 'Date']
    feature_cols = TARGET_COLS + other_cols
    print(f"[DATA] Features: {len(feature_cols)}, Records: {len(df_feat)}")
    return df_feat, feature_cols


# ─────────────────── Sequence Builder ───────────────────
def create_sequences(df_feat, feature_cols, lookback=LOOKBACK, max_horizon=MAX_HORIZON):
    N = len(df_feat)
    train_end = int(N * 0.70)
    val_end   = int(N * 0.85)

    df_train = df_feat.iloc[:train_end].copy()
    df_val   = df_feat.iloc[train_end:val_end].copy()
    df_test  = df_feat.iloc[val_end:].copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(df_train[feature_cols].values)
    scaler_y.fit(df_train[TARGET_COLS].values)

    joblib.dump(scaler_X, MODELS_DIR / "scaler_X_t7opt.pkl")
    joblib.dump(scaler_y, MODELS_DIR / "scaler_y_t7opt.pkl")

    def make_ds(df_sub):
        Xv = scaler_X.transform(df_sub[feature_cols].values)
        yv = scaler_y.transform(df_sub[TARGET_COLS].values)
        Xl, yl = [], []
        for i in range(lookback, len(df_sub) - max_horizon + 1):
            Xl.append(Xv[i - lookback:i, :])
            yl.append(yv[i:i + max_horizon, :])
        return np.array(Xl, np.float32), np.array(yl, np.float32)

    X_tr, y_tr = make_ds(df_train)
    X_va, y_va = make_ds(df_val)
    X_te, y_te = make_ds(df_test)

    print(f"[SEQ] Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (scaler_X, scaler_y), df_test


# ─────────────────── Attention Layer ───────────────────
class SelfAttention(layers.Layer):
    """Scaled Dot-Product Self Attention."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_q = layers.Dense(units, use_bias=False)
        self.W_k = layers.Dense(units, use_bias=False)
        self.W_v = layers.Dense(units, use_bias=False)
        self.scale = float(units) ** 0.5

    def call(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = ops.matmul(Q, ops.transpose(K, axes=[0, 2, 1])) / self.scale
        attn   = ops.softmax(scores, axis=-1)
        return ops.matmul(attn, V)

    def get_config(self):
        cfg = super().get_config()
        cfg['units'] = self.units
        return cfg


# ─────────────────── Model Architectures ───────────────────
def build_bigru_attention_model(input_shape, horizon=MAX_HORIZON, num_targets=4):
    """BiGRU + Self-Attention + T+7 Refinement Head."""
    inp    = layers.Input(shape=input_shape, name='seq_input')
    n_feat = num_targets

    y_base          = layers.Lambda(lambda x: x[:, -1, :n_feat], name='y_t_base')(inp)
    y_base_expanded = layers.RepeatVector(horizon, name='repeat_base')(y_base)

    # Encoder
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True), name='bigru_1')(inp)
    x = layers.Dropout(0.15)(x)
    x = layers.Bidirectional(layers.GRU(48, return_sequences=True), name='bigru_2')(x)
    x = layers.Dropout(0.15)(x)

    x_attn  = SelfAttention(64, name='self_attention')(x)
    x_attn  = layers.Dropout(0.10)(x_attn)
    x_global= layers.GlobalAveragePooling1D(name='gap')(x_attn)
    x_last  = layers.Lambda(lambda t: t[:, -1, :], name='last_step')(x_attn)
    x_fused = layers.Concatenate(name='fuse')([x_global, x_last])

    x_proj = layers.Dense(128, activation='gelu', name='proj_1')(x_fused)
    x_proj = layers.BatchNormalization()(x_proj)
    x_proj = layers.Dropout(0.15)(x_proj)
    x_proj = layers.Dense(96, activation='gelu', name='proj_2')(x_proj)
    x_proj = layers.Dropout(0.10)(x_proj)

    # Main delta head
    delta_flat    = layers.Dense(horizon * num_targets, activation='linear', name='delta_flat')(x_proj)
    delta_reshaped= layers.Reshape((horizon, num_targets), name='delta_seq')(delta_flat)
    pred_main     = layers.Add(name='residual_add')([y_base_expanded, delta_reshaped])

    # T+7 Refinement Head (h=5..9, index 4..8)
    x_ref = layers.Dense(64, activation='relu', name='ref_dense1')(x_proj)
    x_ref = layers.Dropout(0.10)(x_ref)
    x_ref = layers.Dense(48, activation='relu', name='ref_dense2')(x_ref)
    corr  = layers.Dense(5 * num_targets, activation='tanh', name='refinement_flat')(x_ref)
    corr  = layers.Reshape((5, num_targets), name='refinement_seq')(corr)
    corr  = layers.Lambda(lambda t: t * 0.05, name='refinement_scaled')(corr)

    def apply_refinement(inputs):
        pred, c = inputs
        left  = pred[:, :4, :]
        mid   = pred[:, 4:9, :] + c
        right = pred[:, 9:, :]
        return ops.concatenate([left, mid, right], axis=1)

    out = layers.Lambda(apply_refinement, name='refined_output')([pred_main, corr])

    model = models.Model(inputs=inp, outputs=out, name='BiGRU_Attn_T7Refined')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=weighted_huber_loss(HORIZON_WEIGHTS),
        metrics=['mae']
    )
    return model


def build_deep_lstm_model(input_shape, horizon=MAX_HORIZON, num_targets=4):
    """3-layer LSTM with encoder residual connection."""
    inp    = layers.Input(shape=input_shape, name='seq_input')
    n_feat = num_targets

    y_base          = layers.Lambda(lambda x: x[:, -1, :n_feat], name='y_t_base')(inp)
    y_base_expanded = layers.RepeatVector(horizon, name='repeat_base')(y_base)

    x  = layers.LSTM(96, return_sequences=True, name='lstm_1')(inp)
    x  = layers.Dropout(0.20)(x)
    x2 = layers.LSTM(96, return_sequences=True, name='lstm_2')(x)
    x2 = layers.Dropout(0.15)(x2)
    x_res = layers.Add(name='enc_residual')([x, x2])
    x3    = layers.LSTM(64, return_sequences=False, name='lstm_3')(x_res)
    x3    = layers.Dropout(0.15)(x3)

    x_proj = layers.Dense(128, activation='gelu', name='proj_1')(x3)
    x_proj = layers.BatchNormalization()(x_proj)
    x_proj = layers.Dropout(0.15)(x_proj)
    x_proj = layers.Dense(96, activation='relu', name='proj_2')(x_proj)
    x_proj = layers.Dropout(0.10)(x_proj)

    delta_flat    = layers.Dense(horizon * num_targets, activation='linear', name='delta_flat')(x_proj)
    delta_reshaped= layers.Reshape((horizon, num_targets), name='delta_seq')(delta_flat)
    out           = layers.Add(name='residual_add')([y_base_expanded, delta_reshaped])

    model = models.Model(inputs=inp, outputs=out, name='DeepLSTM_T7')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=weighted_huber_loss(HORIZON_WEIGHTS),
        metrics=['mae']
    )
    return model


def build_lightweight_gru(input_shape, horizon=MAX_HORIZON, num_targets=4):
    """Lightweight GRU for ensemble diversity."""
    inp    = layers.Input(shape=input_shape, name='seq_input')
    n_feat = num_targets

    y_base          = layers.Lambda(lambda x: x[:, -1, :n_feat], name='y_t_base')(inp)
    y_base_expanded = layers.RepeatVector(horizon, name='repeat_base')(y_base)

    x = layers.GRU(48, return_sequences=True, name='gru_1')(inp)
    x = layers.Dropout(0.20)(x)
    x = layers.GRU(32, return_sequences=False, name='gru_2')(x)
    x = layers.Dropout(0.15)(x)

    x_proj = layers.Dense(64, activation='relu', name='proj')(x)
    x_proj = layers.Dropout(0.10)(x_proj)

    delta_flat    = layers.Dense(horizon * num_targets, activation='linear', name='delta_flat')(x_proj)
    delta_reshaped= layers.Reshape((horizon, num_targets), name='delta_seq')(delta_flat)
    out           = layers.Add(name='residual_add')([y_base_expanded, delta_reshaped])

    model = models.Model(inputs=inp, outputs=out, name='Lightweight_GRU')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_huber_loss(HORIZON_WEIGHTS),
        metrics=['mae']
    )
    return model


# ─────────────────── Evaluation ───────────────────
def evaluate_model(model, X_test, y_test, scaler_y):
    y_pred_scaled = model.predict(X_test, verbose=0)
    N, H, nc = y_test.shape
    y_true_real = np.zeros_like(y_test)
    y_pred_real = np.zeros_like(y_pred_scaled)
    for h in range(H):
        y_true_real[:, h, :] = scaler_y.inverse_transform(y_test[:, h, :])
        y_pred_real[:, h, :] = scaler_y.inverse_transform(y_pred_scaled[:, h, :])

    results = {}
    for h in range(1, H + 1):
        idx = h - 1
        maes, rmses, mapes, r2s = [], [], [], []
        for p_idx in range(nc):
            yt = y_true_real[:, idx, p_idx]
            yp = y_pred_real[:, idx, p_idx]
            maes.append(mean_absolute_error(yt, yp))
            rmses.append(np.sqrt(mean_squared_error(yt, yp)))
            mapes.append(np.mean(np.abs((yt - yp) / yt)) * 100.0)
            r2s.append(r2_score(yt, yp))
        results[h] = {
            'MAE':   np.mean(maes),
            'RMSE':  np.mean(rmses),
            'MAPE%': np.mean(mapes),
            'R2':    np.mean(r2s)
        }
    return results, (y_true_real, y_pred_real)


def compute_ensemble_metrics(y_true_real, y_ens):
    H, nc = y_true_real.shape[1], y_true_real.shape[2]
    results = {}
    detail_rows = []
    for h in range(1, H + 1):
        idx = h - 1
        maes, rmses, mapes, r2s = [], [], [], []
        for p_idx, col in enumerate(TARGET_COLS):
            yt = y_true_real[:, idx, p_idx]
            yp = y_ens[:, idx, p_idx]
            mae  = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mape = np.mean(np.abs((yt - yp) / yt)) * 100.0
            r2   = r2_score(yt, yp)
            maes.append(mae); rmses.append(rmse); mapes.append(mape); r2s.append(r2)
            if h in EVAL_HORIZONS:
                detail_rows.append({
                    'Horizon_Days': h, 'Horizon_Label': f"T+{h} ({h} Ngay)",
                    'Product': col, 'Product_Name': PRODUCT_LABELS[col],
                    'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
                    'MAPE%': round(mape, 2), 'R2': round(r2, 4)
                })
        results[h] = {
            'MAE': np.mean(maes), 'RMSE': np.mean(rmses),
            'MAPE%': np.mean(mapes), 'R2': np.mean(r2s)
        }
    return results, detail_rows


# ─────────────────── Plotting ───────────────────
def generate_comparison_report(baseline, optimized, y_true, y_ens):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    horizons  = list(range(1, MAX_HORIZON + 1))
    r2_base   = [baseline.get(h, {}).get('R2', 0)    for h in horizons]
    r2_opt    = [optimized[h]['R2']                   for h in horizons]
    mape_base = [baseline.get(h, {}).get('MAPE%', 0) for h in horizons]
    mape_opt  = [optimized[h]['MAPE%']               for h in horizons]

    # Plot 1: R²
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(horizons, r2_base, 'o--', color='#94a3b8', lw=2,   ms=5, label='Baseline v2.0')
    ax1.plot(horizons, r2_opt,  's-',  color='#2563eb', lw=2.5, ms=5, label='Optimized v2.1')
    ax1.axvline(x=7, color='#10b981', ls=':', lw=2, label='T+7')
    ax1.fill_between(horizons, r2_base, r2_opt,
                     where=[r2_opt[i] > r2_base[i] for i in range(len(horizons))],
                     alpha=0.15, color='#10b981')
    ax1.axhline(y=0.87, color='#f59e0b', ls='--', lw=1.5, alpha=0.8, label='Target R²=0.87')
    ax1.set_title('R² Score: Baseline vs Optimized', fontweight='bold')
    ax1.set_xlabel('Horizon (Ngay)'); ax1.set_ylabel('R²')
    ax1.legend(fontsize=8); ax1.set_ylim([-0.1, 1.05]); ax1.grid(True, alpha=0.3)

    # Plot 2: MAPE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(horizons, mape_base, 'o--', color='#94a3b8', lw=2,   ms=5, label='Baseline v2.0')
    ax2.plot(horizons, mape_opt,  's-',  color='#ef4444', lw=2.5, ms=5, label='Optimized v2.1')
    ax2.axvline(x=7, color='#10b981', ls=':', lw=2)
    ax2.axhline(y=3.5, color='#f59e0b', ls='--', lw=1.5, label='Target <=3.5%')
    ax2.set_title('MAPE%: Baseline vs Optimized', fontweight='bold')
    ax2.set_xlabel('Horizon (Ngay)'); ax2.set_ylabel('MAPE (%)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Plot 3: T+7 Actual vs Predicted (MG95)
    ax3 = fig.add_subplot(gs[0, 2])
    n_show = min(80, y_true.shape[0])
    yt7 = y_true[-n_show:, 6, 0]
    yp7 = y_ens[-n_show:,  6, 0]
    x_ax = np.arange(n_show)
    ax3.fill_between(x_ax, yt7 * 0.98, yt7 * 1.02, alpha=0.12, color='#1e293b')
    ax3.plot(x_ax, yt7, '-',  color='#1e293b', lw=1.8, label='Actual')
    ax3.plot(x_ax, yp7, '--', color='#3b82f6', lw=2.0, label='Predicted T+7')
    ax3.set_title('T+7 Forecast vs Actual — MG95 (Test Set)', fontweight='bold')
    ax3.set_xlabel('Sample'); ax3.set_ylabel('USD/thung')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # Plot 4: Horizon Weights
    ax4 = fig.add_subplot(gs[1, 0])
    h_idx  = list(range(1, MAX_HORIZON + 1))
    colors = ['#ef4444' if w >= 4 else '#f59e0b' if w >= 3 else '#3b82f6' if w >= 2
              else '#10b981' if w >= 1.5 else '#94a3b8' for w in HORIZON_WEIGHTS]
    bars = ax4.bar(h_idx, HORIZON_WEIGHTS, color=colors, edgecolor='white', linewidth=0.5)
    ax4.axvline(x=7, color='#10b981', ls='--', lw=2, label='T+7')
    ax4.set_title('Horizon Loss Weights (T+7 Priority)', fontweight='bold')
    ax4.set_xlabel('Horizon h'); ax4.set_ylabel('Weight')
    ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')
    for bar, w in zip(bars, HORIZON_WEIGHTS):
        if w > 1:
            ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                     f'{w:.0f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot 5: Scatter Actual vs Predicted T+7
    ax5 = fig.add_subplot(gs[1, 1])
    all_yt7 = y_true[:, 6, :].flatten()
    all_yp7 = y_ens[:, 6, :].flatten()
    ax5.scatter(all_yt7, all_yp7, alpha=0.3, s=12, color='#3b82f6', edgecolors='none')
    mn, mx = min(all_yt7.min(), all_yp7.min()), max(all_yt7.max(), all_yp7.max())
    ax5.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
    r2_sc = r2_score(all_yt7, all_yp7)
    ax5.set_title('Scatter: Actual vs Predicted (T+7)', fontweight='bold')
    ax5.set_xlabel('Actual (USD/thung)'); ax5.set_ylabel('Predicted (USD/thung)')
    ax5.text(0.05, 0.92, f'R² = {r2_sc:.4f}', transform=ax5.transAxes,
             fontsize=11, fontweight='bold', color='#1e293b')
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # Plot 6: Summary Metrics Bar
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_h = [1, 3, 7, 20]
    r2_key   = [optimized[h]['R2']    for h in metrics_h]
    mape_key = [optimized[h]['MAPE%'] for h in metrics_h]
    x_pos = np.arange(len(metrics_h))
    width = 0.35
    ax6.bar(x_pos - width/2, r2_key, width,
            label='R² (Optimized)',
            color=['#10b981' if r >= 0.85 else '#f59e0b' if r >= 0.75 else '#ef4444' for r in r2_key],
            edgecolor='white')
    ax6_t = ax6.twinx()
    ax6_t.bar(x_pos + width/2, mape_key, width,
              label='MAPE% (Optimized)', color='#3b82f6', alpha=0.7, edgecolor='white')
    ax6.set_xticks(x_pos); ax6.set_xticklabels([f'T+{h}' for h in metrics_h])
    ax6.set_ylabel('R²'); ax6_t.set_ylabel('MAPE%')
    ax6.axhline(y=0.87, color='#10b981', ls='--', lw=1.5, alpha=0.8)
    ax6.set_title('Summary Metrics — v2.1', fontweight='bold')
    l1, lb1 = ax6.get_legend_handles_labels()
    l2, lb2 = ax6_t.get_legend_handles_labels()
    ax6.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='lower right')
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle('PetroForecast AI v2.1 — T+7 Optimization Report', fontsize=15, fontweight='bold', y=1.01)
    out_path = REPORTS_DIR / "t7_optimization_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


# ─────────────────── Main ───────────────────
def main():
    print("=" * 72)
    print("  PETROFORECAST AI v2.1 — T+7 SPECIALIZED OPTIMIZATION")
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"  Device: {gpu_info}")
    print(f"  Lookback: {LOOKBACK} ngay  |  Horizon weights active")
    print("=" * 72)

    # 1. Data
    df_feat, feature_cols = load_and_preprocess_data()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (scaler_X, scaler_y), df_test = \
        create_sequences(df_feat, feature_cols)

    input_shape = (LOOKBACK, X_tr.shape[2])
    print(f"\n[INFO] Input shape: {input_shape}")

    def get_callbacks(name):
        return [
            callbacks.EarlyStopping(
                monitor='val_loss', patience=15,
                restore_best_weights=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=6,
                min_lr=5e-6, verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{name}_best.keras"),
                monitor='val_loss', save_best_only=True, verbose=0
            )
        ]

    # 2. Train Model A: BiGRU + Attention + Refinement
    print("\n[1/3] BiGRU + Self-Attention + T+7 Refinement Head...")
    model_a = build_bigru_attention_model(input_shape)
    t0 = time.time()
    model_a.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=80, batch_size=32,
                callbacks=get_callbacks('bigru_attn_t7'), verbose=1)
    print(f"  -> Time: {time.time()-t0:.1f}s")

    # 3. Train Model B: Deep LSTM
    print("\n[2/3] Deep LSTM (3 layers + encoder residual)...")
    model_b = build_deep_lstm_model(input_shape)
    t0 = time.time()
    model_b.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=80, batch_size=32,
                callbacks=get_callbacks('deep_lstm_t7'), verbose=1)
    print(f"  -> Time: {time.time()-t0:.1f}s")

    # 4. Train Model C: Lightweight GRU
    print("\n[3/3] Lightweight GRU (ensemble diversity)...")
    model_c = build_lightweight_gru(input_shape)
    t0 = time.time()
    model_c.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=80, batch_size=64,
                callbacks=get_callbacks('lightweight_gru_t7'), verbose=1)
    print(f"  -> Time: {time.time()-t0:.1f}s")

    # 5. Evaluate
    print("\n[EVAL] Evaluating on test set...")
    res_a, (y_true, yp_a) = evaluate_model(model_a, X_te, y_te, scaler_y)
    res_b, (_,      yp_b) = evaluate_model(model_b, X_te, y_te, scaler_y)
    res_c, (_,      yp_c) = evaluate_model(model_c, X_te, y_te, scaler_y)

    # Weighted ensemble: A=0.45, B=0.35, C=0.20
    y_ens = yp_a * 0.45 + yp_b * 0.35 + yp_c * 0.20
    ens_result, detail_rows = compute_ensemble_metrics(y_true, y_ens)

    # Print T+7 comparison
    print("\n" + "=" * 72)
    print("  T+7 Individual Model Scores:")
    for name, res in [('BiGRU-Attn', res_a), ('DeepLSTM', res_b),
                      ('LightGRU', res_c), ('ENSEMBLE(weighted)', ens_result)]:
        r = res[7]
        print(f"  {name:20s}: R²={r['R2']:.4f}  MAPE={r['MAPE%']:.2f}%  MAE={r['MAE']:.2f}")

    # Baseline comparison
    baseline_by_h = {
        1:  {'R2': 0.9721, 'MAPE%': 1.70, 'MAE': 1.92, 'RMSE': 4.03},
        3:  {'R2': 0.9158, 'MAPE%': 3.18, 'MAE': 3.59, 'RMSE': 7.07},
        7:  {'R2': 0.7964, 'MAPE%': 5.01, 'MAE': 5.73, 'RMSE': 11.14},
        20: {'R2': 0.3720, 'MAPE%': 8.48, 'MAE': 10.18,'RMSE': 20.19},
    }
    full_baseline = {}
    for h in range(1, 21):
        closest = min(baseline_by_h.keys(), key=lambda k: abs(k - h))
        full_baseline[h] = baseline_by_h[closest]

    print("\n  IMPROVEMENT vs Baseline:")
    for h in [1, 3, 7, 20]:
        b = baseline_by_h.get(h, {})
        o = ens_result[h]
        dR2   = o['R2']    - b.get('R2', 0)
        dMAPE = b.get('MAPE%', 0) - o['MAPE%']
        flag  = "=> " if h == 7 else "   "
        print(f"  {flag}T+{h:2d}: R² {b.get('R2',0):.4f}->{o['R2']:.4f} "
              f"({'+' if dR2>=0 else ''}{dR2:.4f})  "
              f"MAPE {b.get('MAPE%',0):.2f}%->{o['MAPE%']:.2f}% "
              f"({'+'}{dMAPE:.2f}pp reduced)")

    # Save models
    model_a.save(MODELS_DIR / "T7_Optimized_BiGRU_Attn.keras")
    model_b.save(MODELS_DIR / "T7_Optimized_DeepLSTM.keras")
    model_c.save(MODELS_DIR / "T7_Optimized_LightGRU.keras")
    print(f"\n[SAVE] 3 mo hinh luu tai {MODELS_DIR}")

    # Save metrics
    df_det = pd.DataFrame(detail_rows)
    df_sum = pd.DataFrame([{
        'Horizon_Days': h,
        'Horizon_Label': f"T+{h} ({h} Ngay)",
        'Avg_MAE':   round(ens_result[h]['MAE'],   2),
        'Avg_RMSE':  round(ens_result[h]['RMSE'],  2),
        'Avg_MAPE%': round(ens_result[h]['MAPE%'], 2),
        'Avg_R2':    round(ens_result[h]['R2'],    4)
    } for h in EVAL_HORIZONS])

    df_det.to_csv(REPORTS_DIR / "t7_optimized_detailed_metrics.csv", index=False)
    df_sum.to_csv(REPORTS_DIR / "t7_optimized_summary_metrics.csv",  index=False)

    print("\n  BANG TONG HOP — T+7 OPTIMIZED ENSEMBLE v2.1:")
    print(df_sum.to_string(index=False))

    # Generate plots
    generate_comparison_report(full_baseline, ens_result, y_true, y_ens)

    # Final verdict
    print("\n" + "=" * 72)
    t7 = ens_result[7]
    r2_ok   = "OK" if t7['R2']    >= 0.87  else "MISS"
    mape_ok = "OK" if t7['MAPE%'] <= 3.50  else "MISS"
    print(f"  KET QUA T+7 FINAL:")
    print(f"  [{r2_ok}  ] R2   = {t7['R2']:.4f}  (muc tieu >= 0.87)")
    print(f"  [{mape_ok}] MAPE = {t7['MAPE%']:.2f}%  (muc tieu <= 3.50%)")
    print("=" * 72)


if __name__ == '__main__':
    main()

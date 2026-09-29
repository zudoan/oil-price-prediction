# scripts/train_multi_horizon.py
"""
Huấn luyện Mô hình Multi-Horizon Residual Deep Learning (Keras 3 + PyTorch CUDA)
Hỗ trợ dự báo đồng thời các mốc: 1 ngày, 3 ngày, 7 ngày, 20 ngày.
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

# Hardware detection
import torch
import keras
from keras import layers, models, callbacks

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "price_petroleum.xlsx"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs_v2"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
PRODUCT_LABELS = {
    'MG95': 'MOGAS 95 Unleaded',
    'MG92': 'MOGAS 92 Unleaded',
    'DO_0001': 'Gasoil 10ppm (DO 0.001%)',
    'DO_005': 'Gasoil 500ppm (DO 0.05%)'
}

LOOKBACK = 15
MAX_HORIZON = 20
EVAL_HORIZONS = [1, 3, 7, 20]

def load_and_preprocess_data():
    print(f"Loading data from {DATA_FILE}...")
    df_raw = pd.read_excel(DATA_FILE)
    df = df_raw.iloc[:, [3, 4, 5, 6, 7]].copy()
    df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']
    
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True)
    df[TARGET_COLS] = df[TARGET_COLS].ffill().bfill()

    # Feature Engineering (giữ chuẩn đồng nhất 44 features)
    df_feat = df.copy()
    df_feat['SPREAD_MG95_MG92'] = df_feat['MG95'] - df_feat['MG92']
    df_feat['SPREAD_DO0001_DO005'] = df_feat['DO_0001'] - df_feat['DO_005']
    df_feat['SPREAD_GAS_OIL'] = df_feat['MG95'] - df_feat['DO_005']

    for sp in ['SPREAD_MG95_MG92', 'SPREAD_DO0001_DO005', 'SPREAD_GAS_OIL']:
        roll_mean = df_feat[sp].rolling(20, min_periods=5).mean()
        roll_std = df_feat[sp].rolling(20, min_periods=5).std().replace(0, 1e-6)
        df_feat[f'{sp}_Z20'] = (df_feat[sp] - roll_mean) / roll_std

    for col in TARGET_COLS:
        df_feat[f'{col}_EMA5'] = df_feat[col].ewm(span=5, adjust=False).mean()
        df_feat[f'{col}_EMA10'] = df_feat[col].ewm(span=10, adjust=False).mean()
        df_feat[f'{col}_EMA20'] = df_feat[col].ewm(span=20, adjust=False).mean()
        sma20 = df_feat[col].rolling(20, min_periods=5).mean()
        std20 = df_feat[col].rolling(20, min_periods=5).std().replace(0, 1e-6)
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df_feat[f'{col}_BB_Width'] = (bb_upper - bb_lower) / sma20.replace(0, 1e-6)
        df_feat[f'{col}_BB_PctB'] = (df_feat[col] - bb_lower) / (bb_upper - bb_lower).replace(0, 1e-6)
        df_feat[f'{col}_LogRet'] = np.log(df_feat[col] / df_feat[col].shift(1)).fillna(0)
        df_feat[f'{col}_Ret5d'] = (df_feat[col] - df_feat[col].shift(5)) / df_feat[col].shift(5).replace(0, 1e-6)
        delta = df_feat[col].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().replace(0, 1e-6)
        rs = gain / loss
        df_feat[f'{col}_RSI14'] = 100 - (100 / (1 + rs))
        ema12 = df_feat[col].ewm(span=12, adjust=False).mean()
        ema26 = df_feat[col].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_feat[f'{col}_MACD'] = macd_line
        df_feat[f'{col}_MACD_Sig'] = signal_line

    df_feat['Dow_Sin'] = np.sin(2 * np.pi * df_feat['Date'].dt.dayofweek / 5.0)
    df_feat['Dow_Cos'] = np.cos(2 * np.pi * df_feat['Date'].dt.dayofweek / 5.0)
    df_feat['Month_Sin'] = np.sin(2 * np.pi * df_feat['Date'].dt.month / 12.0)
    df_feat['Month_Cos'] = np.cos(2 * np.pi * df_feat['Date'].dt.month / 12.0)
    df_feat = df_feat.dropna().reset_index(drop=True)

    other_cols = [c for c in df_feat.columns if c not in TARGET_COLS and c != 'Date']
    feature_cols = TARGET_COLS + other_cols
    print(f"Total features: {len(feature_cols)}, Total records: {len(df_feat)}")
    return df_feat, feature_cols

def create_multi_horizon_sequences(df_feat, feature_cols, lookback=LOOKBACK, max_horizon=MAX_HORIZON):
    N = len(df_feat)
    train_end = int(N * 0.70)
    val_end = int(N * 0.85)

    df_train = df_feat.iloc[:train_end].copy()
    df_val = df_feat.iloc[train_end:val_end].copy()
    df_test = df_feat.iloc[val_end:].copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(df_train[feature_cols].values)
    scaler_y.fit(df_train[TARGET_COLS].values)

    joblib.dump(scaler_X, MODELS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODELS_DIR / "scaler_y.pkl")

    def make_dataset(df_sub):
        X_vals = scaler_X.transform(df_sub[feature_cols].values)
        y_vals = scaler_y.transform(df_sub[TARGET_COLS].values)
        
        X_list, y_multi_list = [], []
        # Each sample needs lookback inputs and max_horizon future steps
        for i in range(lookback, len(df_sub) - max_horizon + 1):
            X_seq = X_vals[i - lookback:i, :]
            y_future = y_vals[i:i + max_horizon, :]  # shape: (max_horizon, 4)
            X_list.append(X_seq)
            y_multi_list.append(y_future)
        return np.array(X_list, dtype=np.float32), np.array(y_multi_list, dtype=np.float32)

    X_train, y_train = make_dataset(df_train)
    X_val, y_val = make_dataset(df_val)
    X_test, y_test = make_dataset(df_test)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes  : X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes : X={X_test.shape}, y={y_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (scaler_X, scaler_y), df_test

def build_residual_multi_horizon_model(input_shape, horizon=MAX_HORIZON, num_targets=4, rnn_type='GRU'):
    """
    Kiến trúc Multi-Horizon Residual Deep Learning:
    - Input: Sequence 15 ngày quá khứ
    - Baseline: Giá ngày cuối cùng t (tại index -1, các cột target đầu tiên)
    - RNN Encoder: Học biểu diễn xu hướng đa biến & crack spreads
    - Dense Multi-Step Head: Dự đoán delta vector cho từng bước h = 1..H
    - Residual Skip Connection: pred[t+h] = y[t] + delta[t+h]
    """
    inp = layers.Input(shape=input_shape, name='seq_input')
    
    # Target columns are the first num_targets columns in normalized input
    # Shape: (batch, num_targets)
    y_base = layers.Lambda(lambda x: x[:, -1, :num_targets], name='y_t_base')(inp)
    # Expand to (batch, horizon, num_targets)
    y_base_expanded = layers.RepeatVector(horizon, name='repeat_base')(y_base)

    if rnn_type == 'GRU':
        x = layers.GRU(64, return_sequences=True, name='gru_1')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(48, return_sequences=False, name='gru_2')(x)
        x = layers.Dropout(0.2)(x)
    else:
        x = layers.LSTM(64, return_sequences=True, name='lstm_1')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(48, return_sequences=False, name='lstm_2')(x)
        x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu', name='dense_proj')(x)
    x = layers.Dropout(0.1)(x)
    
    # Dự đoán delta cho toàn bộ horizon * num_targets
    delta_flat = layers.Dense(horizon * num_targets, activation='linear', name='delta_flat')(x)
    delta_reshaped = layers.Reshape((horizon, num_targets), name='delta_seq')(delta_flat)

    # Residual addition: y[t+h] = y[t] + delta[t+h]
    out = layers.Add(name='residual_add')([y_base_expanded, delta_reshaped])

    model = models.Model(inputs=inp, outputs=out, name=f'Residual_MH_{rnn_type}')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model

def evaluate_multi_horizon(model, X_test, y_test, scaler_y, horizons=EVAL_HORIZONS):
    """
    Đánh giá chi tiết mô hình tại các mốc T+1, T+3, T+7, T+20 và tính tổng hợp.
    """
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    N_test, H, num_cols = y_test.shape
    y_true_real = np.zeros_like(y_test)
    y_pred_real = np.zeros_like(y_pred_scaled)

    for h in range(H):
        y_true_real[:, h, :] = scaler_y.inverse_transform(y_test[:, h, :])
        y_pred_real[:, h, :] = scaler_y.inverse_transform(y_pred_scaled[:, h, :])

    results_by_h = {}
    detailed_rows = []

    for h in range(1, H + 1):
        idx = h - 1
        mae_prods = []
        rmse_prods = []
        mape_prods = []
        r2_prods = []

        for p_idx, col in enumerate(TARGET_COLS):
            y_t = y_true_real[:, idx, p_idx]
            y_p = y_pred_real[:, idx, p_idx]
            
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100.0
            r2 = r2_score(y_t, y_p)

            mae_prods.append(mae)
            rmse_prods.append(rmse)
            mape_prods.append(mape)
            r2_prods.append(r2)

            if h in horizons:
                detailed_rows.append({
                    'Horizon_Days': h,
                    'Horizon_Label': f"T+{h} ({h} Ngày)",
                    'Product': col,
                    'Product_Name': PRODUCT_LABELS[col],
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                    'MAPE%': round(mape, 2),
                    'R2': round(r2, 4)
                })

        results_by_h[h] = {
            'MAE': np.mean(mae_prods),
            'RMSE': np.mean(rmse_prods),
            'MAPE%': np.mean(mape_prods),
            'R2': np.mean(r2_prods)
        }

    return results_by_h, detailed_rows, (y_true_real, y_pred_real)

def main():
    print("=" * 70)
    print("  PETROFORECAST AI — MULTI-HORIZON RESIDUAL DEEP LEARNING (T+3, T+7, T+20)")
    print(f"  PyTorch CUDA GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}")
    print("=" * 70)

    df_feat, feature_cols = load_and_preprocess_data()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (scaler_X, scaler_y), df_test = create_multi_horizon_sequences(
        df_feat, feature_cols, LOOKBACK, MAX_HORIZON
    )

    # 1. Train Residual Multi-Horizon GRU
    print("\n[1/3] Huấn luyện Residual Multi-Horizon GRU...")
    model_gru = build_residual_multi_horizon_model(X_tr.shape[1:], horizon=MAX_HORIZON, rnn_type='GRU')
    
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    t0 = time.time()
    hist_gru = model_gru.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=60,
        batch_size=32,
        callbacks=cb,
        verbose=1
    )
    t_gru = time.time() - t0
    print(f"  Thời gian huấn luyện GRU: {t_gru:.1f}s")

    # 2. Train Residual Multi-Horizon LSTM
    print("\n[2/3] Huấn luyện Residual Multi-Horizon LSTM...")
    model_lstm = build_residual_multi_horizon_model(X_tr.shape[1:], horizon=MAX_HORIZON, rnn_type='LSTM')
    t0 = time.time()
    hist_lstm = model_lstm.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=60,
        batch_size=32,
        callbacks=cb,
        verbose=1
    )
    t_lstm = time.time() - t0
    print(f"  Thời gian huấn luyện LSTM: {t_lstm:.1f}s")

    # 3. Evaluate GRU
    print("\n[3/3] Đánh giá trên tập Test...")
    res_h_gru, det_gru, (y_true, y_pred_gru) = evaluate_multi_horizon(model_gru, X_te, y_te, scaler_y)
    res_h_lstm, det_lstm, (_, y_pred_lstm) = evaluate_multi_horizon(model_lstm, X_te, y_te, scaler_y)

    # Ensemble Blending (50% GRU + 50% LSTM)
    y_pred_ens = 0.5 * y_pred_gru + 0.5 * y_pred_lstm
    
    # Calculate ensemble metrics
    detailed_ens = []
    summary_ens = []
    for h in range(1, MAX_HORIZON + 1):
        idx = h - 1
        maes, rmses, mapes, r2s = [], [], [], []
        for p_idx, col in enumerate(TARGET_COLS):
            yt = y_true[:, idx, p_idx]
            yp = y_pred_ens[:, idx, p_idx]
            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mape = np.mean(np.abs((yt - yp) / yt)) * 100.0
            r2 = r2_score(yt, yp)
            maes.append(mae); rmses.append(rmse); mapes.append(mape); r2s.append(r2)

            if h in EVAL_HORIZONS:
                detailed_ens.append({
                    'Horizon_Days': h,
                    'Horizon_Label': f"T+{h} ({h} Ngày)",
                    'Product': col,
                    'Product_Name': PRODUCT_LABELS[col],
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                    'MAPE%': round(mape, 2),
                    'R2': round(r2, 4)
                })

        if h in EVAL_HORIZONS:
            summary_ens.append({
                'Horizon_Days': h,
                'Horizon_Label': f"T+{h} ({h} Ngày)",
                'Avg_MAE': round(np.mean(maes), 2),
                'Avg_RMSE': round(np.mean(rmses), 2),
                'Avg_MAPE%': round(np.mean(mapes), 2),
                'Avg_R2': round(np.mean(r2s), 4)
            })

    # Save models
    model_gru.save(MODELS_DIR / "Residual_MultiHorizon_final.keras")
    model_lstm.save(MODELS_DIR / "Residual_MultiHorizon_LSTM.keras")
    print(f"✅ Đã lưu mô hình tại: {MODELS_DIR / 'Residual_MultiHorizon_final.keras'}")

    # Save Metrics CSV
    df_det = pd.DataFrame(detailed_ens)
    df_sum = pd.DataFrame(summary_ens)
    df_det.to_csv(REPORTS_DIR / "multi_horizon_detailed_metrics.csv", index=False)
    df_sum.to_csv(REPORTS_DIR / "multi_horizon_summary_metrics.csv", index=False)
    print("\n" + "=" * 70)
    print("📊 BẢNG TỔNG HỢP HIỆU NĂNG THEO MỐC DỰ BÁO (MULTI-HORIZON BENCHMARK)")
    print("=" * 70)
    print(df_sum.to_string(index=False))
    print("=" * 70)

    # 4. Generate Visual Plots
    generate_plots(res_h_gru, y_true, y_pred_ens, df_test)

def generate_plots(res_h_gru, y_true, y_pred, df_test):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Suy thoái sai số theo độ dài chân trời dự báo (Horizon Decay Analysis)
    horizons = list(range(1, MAX_HORIZON + 1))
    mape_list = [res_h_gru[h]['MAPE%'] for h in horizons]
    r2_list = [res_h_gru[h]['R2'] for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(horizons, mape_list, marker='o', color='#2563eb', linewidth=2.5, markersize=6)
    ax1.axvline(x=3, color='#f59e0b', linestyle='--', label='T+3 (3 Ngày Swing)')
    ax1.axvline(x=7, color='#10b981', linestyle='--', label='T+7 (Kỳ Điều Hành)')
    ax1.axvline(x=20, color='#ef4444', linestyle='--', label='T+20 (Chu Kỳ Tháng)')
    ax1.set_title('Sai Số Tương Đối Bình Quân (MAPE %) Theo Chân Trời Dự Báo', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Horizon (Số ngày làm việc tiếp theo)')
    ax1.set_ylabel('MAPE (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(horizons, r2_list, marker='s', color='#10b981', linewidth=2.5, markersize=6)
    ax2.axvline(x=3, color='#f59e0b', linestyle='--')
    ax2.axvline(x=7, color='#10b981', linestyle='--')
    ax2.axvline(x=20, color='#ef4444', linestyle='--')
    ax2.set_title('Hệ Số Xác Định (R² Score) Theo Chân Trời Dự Báo', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Horizon (Số ngày làm việc tiếp theo)')
    ax2.set_ylabel('R² Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "multi_horizon_decay_curve.png", dpi=150)
    plt.close()

    # Plot 2: Trajectory Fan Chart (So sánh thực tế vs Dự báo dải 20 ngày tại mẫu gần nhất)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sample_idx = -1  # mẫu cuối cùng
    x_steps = np.arange(1, MAX_HORIZON + 1)

    for i, col in enumerate(TARGET_COLS):
        ax = axes[i // 2, i % 2]
        yt_traj = y_true[sample_idx, :, i]
        yp_traj = y_pred[sample_idx, :, i]
        
        # Uncertainty band widening with sqrt(h)
        band = 1.96 * (res_h_gru[1]['RMSE'] * np.sqrt(x_steps) / 2.0)

        ax.plot(x_steps, yt_traj, 'o-', color='#1e293b', label='Thực tế (Actual)', linewidth=2)
        ax.plot(x_steps, yp_traj, 's--', color='#3b82f6', label='Dự báo AI (Residual MH)', linewidth=2.5)
        ax.fill_between(x_steps, yp_traj - band, yp_traj + band, color='#3b82f6', alpha=0.18, label='Khoảng tin cậy 95% (Fan Band)')

        ax.scatter([3], [yp_traj[2]], color='#f59e0b', s=100, zorder=5, label='Mốc 3 Ngày (T+3)')
        ax.scatter([7], [yp_traj[6]], color='#10b981', s=100, zorder=5, label='Mốc 7 Ngày (T+7)')
        ax.scatter([20], [yp_traj[19]], color='#ef4444', s=100, zorder=5, label='Mốc 20 Ngày (T+20)')

        ax.set_title(f"{PRODUCT_LABELS[col]} — Quỹ Đạo Dự Báo 20 Ngày", fontsize=12, fontweight='bold')
        ax.set_xlabel('Ngày làm việc trong tương lai')
        ax.set_ylabel('USD/thùng')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "multi_horizon_trajectories_sample.png", dpi=150)
    plt.close()
    print(f"📈 Đã tạo các biểu đồ phân tích tại thư mục: {REPORTS_DIR}")

if __name__ == '__main__':
    main()

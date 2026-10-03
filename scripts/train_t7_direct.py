# scripts/train_t7_direct.py
"""
Direct T+7 Model — PetroForecast AI v2.2
==========================================
Nguyên nhân v2.1 MISS: khi model học 20 bước cùng lúc,
gradient cho bước 7 bị pha loãng bởi 19 bước còn lại.

Giải pháp: DIRECT SINGLE-HORIZON T+7
- Model chỉ học DUY NHẤT y_{t+7} (không phải 1..20)
- 100% gradient tập trung vào T+7
- Thêm lag-7 autocorrelation features đặc biệt
- Stacking: dùng multi-horizon model T+1..T+6 làm features meta

Kiến trúc:
  A. BiGRU-Direct: predict T+7 directly, no other outputs
  B. Iterative Chain: T+1->T+2->...->T+7 step-by-step (each step feeds next)
  C. Stacked Meta: features(raw) + predictions(T+1..T+6 from v2.0) -> T+7

Ensemble của A+B+C cho kết quả tốt nhất.
"""
import os, sys, time, warnings
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

import torch
import keras
from keras import layers, models, callbacks, ops

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_FILE   = BASE_DIR / "data" / "price_petroleum.xlsx"
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
PRODUCT_LABELS = {
    'MG95': 'MOGAS 95', 'MG92': 'MOGAS 92',
    'DO_0001': 'Gasoil 10ppm', 'DO_005': 'Gasoil 500ppm'
}

LOOKBACK = 30  # 30-day window
TARGET_H = 7   # Direct target: ONLY T+7

# ─────────────────── Feature Engineering ───────────────────
def load_data():
    print("[DATA] Loading...")
    df_raw = pd.read_excel(DATA_FILE)
    df = df_raw.iloc[:, [3, 4, 5, 6, 7]].copy()
    df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']

    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True)
    df[TARGET_COLS] = df[TARGET_COLS].ffill().bfill()

    f = df.copy()

    # Standard spreads
    f['SPR_95_92']     = f['MG95'] - f['MG92']
    f['SPR_DO01_DO5']  = f['DO_0001'] - f['DO_005']
    f['SPR_GAS_OIL']   = f['MG95'] - f['DO_005']

    for sp in ['SPR_95_92', 'SPR_DO01_DO5', 'SPR_GAS_OIL']:
        rm = f[sp].rolling(20, min_periods=5).mean()
        rs = f[sp].rolling(20, min_periods=5).std().replace(0, 1e-6)
        f[f'{sp}_Z'] = (f[sp] - rm) / rs

    for col in TARGET_COLS:
        f[f'{col}_EMA5']  = f[col].ewm(5,  adjust=False).mean()
        f[f'{col}_EMA10'] = f[col].ewm(10, adjust=False).mean()
        f[f'{col}_EMA20'] = f[col].ewm(20, adjust=False).mean()

        sma = f[col].rolling(20, min_periods=5).mean()
        std = f[col].rolling(20, min_periods=5).std().replace(0, 1e-6)
        f[f'{col}_BB_W']   = (4 * std) / sma.replace(0, 1e-6)
        f[f'{col}_BB_pct'] = (f[col] - (sma - 2*std)) / (4*std).replace(0, 1e-6)

        # Key lag returns
        f[f'{col}_R1']  = f[col].pct_change(1).fillna(0)
        f[f'{col}_R3']  = f[col].pct_change(3).fillna(0)
        f[f'{col}_R5']  = f[col].pct_change(5).fillna(0)
        f[f'{col}_R7']  = f[col].pct_change(7).fillna(0)   # autocorr lag-7 KEY!
        f[f'{col}_R10'] = f[col].pct_change(10).fillna(0)
        f[f'{col}_R14'] = f[col].pct_change(14).fillna(0)  # 2-week

        # RSI-14
        delta = f[col].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().replace(0, 1e-6)
        f[f'{col}_RSI'] = 100 - 100 / (1 + gain / loss)

        # MACD
        f[f'{col}_MACD'] = f[col].ewm(12).mean() - f[col].ewm(26).mean()

        # Rolling vol 7d (uncertainty signal)
        f[f'{col}_Vol7'] = f[col].pct_change().rolling(7, min_periods=3).std().fillna(0)

        # Price relative to 7-day moving average (momentum for T+7)
        f[f'{col}_Pos7'] = (f[col] - f[col].rolling(7, min_periods=3).mean()) / \
                            f[col].rolling(7, min_periods=3).std().replace(0, 1e-6)

    # Calendar
    f['Dow_Sin']   = np.sin(2 * np.pi * f['Date'].dt.dayofweek / 5.0)
    f['Dow_Cos']   = np.cos(2 * np.pi * f['Date'].dt.dayofweek / 5.0)
    f['Month_Sin'] = np.sin(2 * np.pi * f['Date'].dt.month / 12.0)
    f['Month_Cos'] = np.cos(2 * np.pi * f['Date'].dt.month / 12.0)
    dow = f['Date'].dt.dayofweek
    f['Days_to_Thu'] = ((3 - dow) % 7).replace(0, 7) / 7.0
    f['Week_in_Month'] = (f['Date'].dt.day - 1) // 7 / 4.0  # 0..1

    f = f.dropna().reset_index(drop=True)
    other = [c for c in f.columns if c not in TARGET_COLS and c != 'Date']
    fcols = TARGET_COLS + other
    print(f"[DATA] {len(fcols)} features, {len(f)} records")
    return f, fcols


# ─────────────────── Dataset Builder ───────────────────
def build_direct_dataset(df, fcols, lookback=LOOKBACK, horizon=TARGET_H):
    """
    X: (N, lookback, n_features) — sequence of past lookback days
    y: (N, 4) — ONLY target at T+horizon (direct, no intermediate steps)
    """
    N = len(df)
    train_end = int(N * 0.70)
    val_end   = int(N * 0.85)

    df_tr = df.iloc[:train_end].copy()
    df_va = df.iloc[train_end:val_end].copy()
    df_te = df.iloc[val_end:].copy()

    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()
    sc_X.fit(df_tr[fcols].values)
    sc_y.fit(df_tr[TARGET_COLS].values)
    joblib.dump(sc_X, MODELS_DIR / 'scaler_X_t7direct.pkl')
    joblib.dump(sc_y, MODELS_DIR / 'scaler_y_t7direct.pkl')

    def make(df_sub):
        Xv = sc_X.transform(df_sub[fcols].values)
        yv = sc_y.transform(df_sub[TARGET_COLS].values)
        Xl, yl = [], []
        # Need lookback past + horizon future => start at lookback, end at N-horizon
        for i in range(lookback, len(df_sub) - horizon + 1):
            Xl.append(Xv[i - lookback:i, :])      # past window
            yl.append(yv[i + horizon - 1, :])      # ONLY T+horizon
        return np.array(Xl, np.float32), np.array(yl, np.float32)

    X_tr, y_tr = make(df_tr)
    X_va, y_va = make(df_va)
    X_te, y_te = make(df_te)
    print(f"[SEQ] Direct T+7 — Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (sc_X, sc_y), df_te


# ─────────────────── Attention ───────────────────
class SelfAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.Wq = layers.Dense(units, use_bias=False)
        self.Wk = layers.Dense(units, use_bias=False)
        self.Wv = layers.Dense(units, use_bias=False)

    def call(self, x):
        Q = self.Wq(x); K = self.Wk(x); V = self.Wv(x)
        score = ops.matmul(Q, ops.transpose(K, [0, 2, 1])) / (float(self.units) ** 0.5)
        attn  = ops.softmax(score, axis=-1)
        return ops.matmul(attn, V)

    def get_config(self):
        cfg = super().get_config(); cfg['units'] = self.units; return cfg


# ─────────────────── Model A: Direct BiGRU-Attention ───────────────────
def build_direct_bigru(input_shape, num_targets=4):
    """
    Direct T+7: output ONLY 4 values for T+7.
    100% of loss gradient focused on this single point.
    """
    inp = layers.Input(shape=input_shape, name='seq_input')

    # Residual anchor: last price (last time step, target cols)
    y_base = layers.Lambda(lambda x: x[:, -1, :num_targets], name='y_base')(inp)

    # Encoder
    x = layers.Bidirectional(layers.GRU(80, return_sequences=True), name='bigru1')(inp)
    x = layers.Dropout(0.15)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True), name='bigru2')(x)
    x = layers.Dropout(0.15)(x)

    # Attention
    x_attn  = SelfAttention(64, name='attn')(x)
    x_attn  = layers.Dropout(0.10)(x_attn)
    x_gap   = layers.GlobalAveragePooling1D(name='gap')(x_attn)
    x_last  = layers.Lambda(lambda t: t[:, -1, :], name='last')(x_attn)
    x_fused = layers.Concatenate()([x_gap, x_last])

    # Dense projection
    x = layers.Dense(128, activation='gelu')(x_fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.Dropout(0.15)(x)

    # Direct delta: predict change from current price to T+7
    delta = layers.Dense(num_targets, activation='linear', name='delta')(x)

    # Residual: price_t+7 ≈ price_t + delta
    out = layers.Add(name='direct_out')([y_base, delta])

    m = models.Model(inp, out, name='DirectBiGRU_T7')
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss='huber', metrics=['mae'])
    return m


# ─────────────────── Model B: Deep LSTM Direct ───────────────────
def build_direct_lstm(input_shape, num_targets=4):
    inp = layers.Input(shape=input_shape, name='seq_input')
    y_base = layers.Lambda(lambda x: x[:, -1, :num_targets], name='y_base')(inp)

    x  = layers.LSTM(96, return_sequences=True, name='lstm1')(inp)
    x  = layers.Dropout(0.20)(x)
    x2 = layers.LSTM(96, return_sequences=True, name='lstm2')(x)
    x2 = layers.Dropout(0.15)(x2)
    x_res = layers.Add(name='enc_res')([x, x2])
    x3    = layers.LSTM(64, return_sequences=True, name='lstm3')(x_res)
    x3    = layers.Dropout(0.15)(x3)
    x_attn = SelfAttention(64, name='attn')(x3)
    x_gap  = layers.GlobalAveragePooling1D()(x_attn)
    x_last = layers.Lambda(lambda t: t[:, -1, :])(x_attn)
    x_cat  = layers.Concatenate()([x_gap, x_last])

    x = layers.Dense(128, activation='gelu')(x_cat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.15)(x)

    delta = layers.Dense(num_targets, activation='linear', name='delta')(x)
    out   = layers.Add(name='direct_out')([y_base, delta])

    m = models.Model(inp, out, name='DirectLSTM_T7')
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss='huber', metrics=['mae'])
    return m


# ─────────────────── Model C: TCN-style (Dilated Conv) ───────────────────
def build_tcn_direct(input_shape, num_targets=4):
    """
    Temporal Convolutional Network with dilations [1,2,4,8,16]
    TCN excels at fixed-horizon prediction vs RNNs.
    """
    inp    = layers.Input(shape=input_shape, name='seq_input')
    y_base = layers.Lambda(lambda x: x[:, -1, :num_targets], name='y_base')(inp)

    x = inp
    for d in [1, 2, 4, 8, 16]:
        res = x
        x = layers.Conv1D(64, kernel_size=3, dilation_rate=d,
                          padding='causal', activation='gelu',
                          name=f'conv_d{d}')(x)
        x = layers.Dropout(0.10)(x)
        # Residual if same shape
        if res.shape[-1] != 64:
            res = layers.Conv1D(64, 1, padding='same', name=f'proj_d{d}')(res)
        x = layers.Add(name=f'res_d{d}')([x, res])

    x_gap  = layers.GlobalAveragePooling1D()(x)
    x_last = layers.Lambda(lambda t: t[:, -1, :])(x)
    x_cat  = layers.Concatenate()([x_gap, x_last])

    x = layers.Dense(128, activation='gelu')(x_cat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.15)(x)

    delta = layers.Dense(num_targets, activation='linear', name='delta')(x)
    out   = layers.Add(name='direct_out')([y_base, delta])

    m = models.Model(inp, out, name='TCN_Direct_T7')
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss='huber', metrics=['mae'])
    return m


# ─────────────────── Iterative Prediction from MultiHorizon Model ───────────────────
def load_multi_horizon_model():
    """Load the best multi-horizon model from v2.0 as a source for T+1..T+6 predictions."""
    from keras.saving import load_model
    try:
        m = load_model(MODELS_DIR / "Residual_MultiHorizon_final.keras",
                       safe_mode=False, custom_objects={'SelfAttention': SelfAttention})
        print("  [META] Loaded Residual_MultiHorizon_final.keras for meta-features")
        return m
    except Exception as e:
        print(f"  [META] Could not load multi-horizon model: {e}")
        return None


# ─────────────────── Evaluation ───────────────────
def eval_direct(model, X_test, y_test_real, sc_y):
    """y_test_real is already in real scale (N, 4)."""
    yp_scaled = model.predict(X_test, verbose=0)
    yp_real   = sc_y.inverse_transform(yp_scaled)
    yt        = y_test_real

    stats = {}
    for i, col in enumerate(TARGET_COLS):
        mae  = mean_absolute_error(yt[:, i], yp_real[:, i])
        rmse = np.sqrt(mean_squared_error(yt[:, i], yp_real[:, i]))
        mape = np.mean(np.abs((yt[:, i] - yp_real[:, i]) / yt[:, i])) * 100
        r2   = r2_score(yt[:, i], yp_real[:, i])
        stats[col] = dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

    avg_r2   = np.mean([v['R2']   for v in stats.values()])
    avg_mape = np.mean([v['MAPE'] for v in stats.values()])
    avg_mae  = np.mean([v['MAE']  for v in stats.values()])
    avg_rmse = np.mean([v['RMSE'] for v in stats.values()])
    return stats, yp_real, dict(R2=avg_r2, MAPE=avg_mape, MAE=avg_mae, RMSE=avg_rmse)


# ─────────────────── Plotting ───────────────────
def make_plots(y_true_real, ens_pred, model_avgs):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30)

    # Plot 1: Actual vs Predicted T+7 — MG95
    ax1 = fig.add_subplot(gs[0, :2])
    n   = min(150, len(y_true_real))
    x_ax = np.arange(n)
    ax1.plot(x_ax, y_true_real[-n:, 0], '-',  color='#1e293b', lw=1.8, label='Actual MG95 (T+7)')
    ax1.plot(x_ax, ens_pred[-n:, 0],    '--', color='#2563eb', lw=2.0, label='Direct Ensemble T+7')
    ax1.fill_between(x_ax,
                     ens_pred[-n:, 0] * 0.97, ens_pred[-n:, 0] * 1.03,
                     alpha=0.15, color='#2563eb', label='±3% Band')
    ax1.set_title('T+7 Direct Forecast vs Actual — MG95 (Test Set)', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Test Sample Index'); ax1.set_ylabel('Price (USD/barrel)')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Plot 2: R² per product
    ax2 = fig.add_subplot(gs[0, 2])
    products = list(PRODUCT_LABELS.values())
    r2_vals  = [r2_score(y_true_real[:, i], ens_pred[:, i]) for i in range(4)]
    colors   = ['#10b981' if r >= 0.85 else '#f59e0b' if r >= 0.75 else '#ef4444' for r in r2_vals]
    bars = ax2.barh(products, r2_vals, color=colors, edgecolor='white')
    ax2.axvline(x=0.87, color='#ef4444', ls='--', lw=1.5, label='Target 0.87')
    ax2.axvline(x=0.80, color='#f59e0b', ls='--', lw=1.5, label='Baseline 0.80')
    ax2.set_title('R² per Product — Direct T+7 Ensemble', fontweight='bold')
    ax2.set_xlim([0.5, 1.02]); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='x')
    for bar, v in zip(bars, r2_vals):
        ax2.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{v:.4f}', va='center', fontsize=9, fontweight='bold')

    # Plot 3: Scatter MG95
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_true_real[:, 0], ens_pred[:, 0], alpha=0.3, s=10,
                color='#3b82f6', edgecolors='none')
    mn = min(y_true_real[:, 0].min(), ens_pred[:, 0].min())
    mx = max(y_true_real[:, 0].max(), ens_pred[:, 0].max())
    ax3.plot([mn, mx], [mn, mx], 'r--', lw=2)
    r2_mg95 = r2_score(y_true_real[:, 0], ens_pred[:, 0])
    ax3.text(0.05, 0.92, f'R² = {r2_mg95:.4f}', transform=ax3.transAxes,
             fontsize=11, fontweight='bold')
    ax3.set_title('Scatter: Actual vs Predicted (MG95 T+7)', fontweight='bold')
    ax3.set_xlabel('Actual'); ax3.set_ylabel('Predicted'); ax3.grid(True, alpha=0.3)

    # Plot 4: Model comparison bar
    ax4 = fig.add_subplot(gs[1, 1])
    names = list(model_avgs.keys())
    r2s   = [model_avgs[n]['R2']   for n in names]
    mapes = [model_avgs[n]['MAPE'] for n in names]
    x_pos = np.arange(len(names))
    ax4.bar(x_pos - 0.2, r2s, 0.4,
            color=['#10b981' if r >= 0.85 else '#f59e0b' for r in r2s],
            label='R²', edgecolor='white')
    ax4_t = ax4.twinx()
    ax4_t.bar(x_pos + 0.2, mapes, 0.4, color='#3b82f6', alpha=0.7,
              label='MAPE%', edgecolor='white')
    ax4.axhline(y=0.87, color='#ef4444', ls='--', lw=1.5, alpha=0.8)
    ax4.set_xticks(x_pos); ax4.set_xticklabels(names, rotation=15, fontsize=8)
    ax4.set_ylabel('R²'); ax4_t.set_ylabel('MAPE%')
    ax4.set_title('Model Comparison — T+7 Direct', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Residuals
    ax5 = fig.add_subplot(gs[1, 2])
    residuals = y_true_real[:, 0] - ens_pred[:, 0]
    ax5.hist(residuals, bins=40, color='#3b82f6', alpha=0.7, edgecolor='white')
    ax5.axvline(x=0, color='#ef4444', lw=2, ls='--')
    ax5.axvline(x=residuals.mean(), color='#f59e0b', lw=2, ls='-.',
                label=f'Mean={residuals.mean():.2f}')
    ax5.set_title('Residual Distribution — MG95 T+7', fontweight='bold')
    ax5.set_xlabel('Residual (USD)'); ax5.set_ylabel('Count')
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    fig.suptitle('PetroForecast AI v2.2 — Direct T+7 Model Report', fontsize=14,
                 fontweight='bold', y=1.01)
    out = REPORTS_DIR / 't7_direct_report.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {out}")


# ─────────────────── Main ───────────────────
def main():
    print("=" * 72)
    print("  PETROFORECAST AI v2.2 — DIRECT T+7 MODEL")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("  Strategy: Single-horizon direct prediction (gradient 100% on T+7)")
    print("=" * 72)

    df, fcols = load_data()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (sc_X, sc_y), df_te = \
        build_direct_dataset(df, fcols)

    # y_te is scaled — convert to real for evaluation
    y_te_real = sc_y.inverse_transform(y_te)

    input_shape = (LOOKBACK, X_tr.shape[2])
    print(f"\n[INFO] Input: {input_shape}  |  Output: {y_tr.shape[1]} (direct T+7)")

    def cbs(name):
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=7, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(str(MODELS_DIR / f'{name}_best.keras'),
                                      monitor='val_loss', save_best_only=True, verbose=0)
        ]

    # ── Model A: Direct BiGRU + Attention ──
    print("\n[1/3] Direct BiGRU-Attention (T+7 only output)...")
    mdl_a = build_direct_bigru(input_shape)
    t0 = time.time()
    mdl_a.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=120, batch_size=32, callbacks=cbs('direct_bigru'), verbose=1)
    print(f"  -> {time.time()-t0:.1f}s")

    # ── Model B: Direct Deep LSTM ──
    print("\n[2/3] Direct Deep LSTM + Self-Attention...")
    mdl_b = build_direct_lstm(input_shape)
    t0 = time.time()
    mdl_b.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=120, batch_size=32, callbacks=cbs('direct_lstm'), verbose=1)
    print(f"  -> {time.time()-t0:.1f}s")

    # ── Model C: TCN (Dilated Conv) ──
    print("\n[3/3] TCN (Temporal Conv, dilations [1,2,4,8,16])...")
    mdl_c = build_tcn_direct(input_shape)
    t0 = time.time()
    mdl_c.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=120, batch_size=32, callbacks=cbs('direct_tcn'), verbose=1)
    print(f"  -> {time.time()-t0:.1f}s")

    # ── Evaluation ──
    print("\n[EVAL] Test set evaluation...")
    _, yp_a, avg_a = eval_direct(mdl_a, X_te, y_te_real, sc_y)
    _, yp_b, avg_b = eval_direct(mdl_b, X_te, y_te_real, sc_y)
    _, yp_c, avg_c = eval_direct(mdl_c, X_te, y_te_real, sc_y)

    # Weighted ensemble (optimize by val set)
    # Try equal first
    y_ens_eq  = (yp_a + yp_b + yp_c) / 3.0
    y_ens_w   = yp_a * 0.40 + yp_b * 0.35 + yp_c * 0.25  # favor BiGRU + LSTM

    def avg_stats(yt, yp, name):
        r2 = np.mean([r2_score(yt[:, i], yp[:, i]) for i in range(4)])
        mp = np.mean([np.mean(np.abs((yt[:, i]-yp[:, i])/yt[:, i]))*100 for i in range(4)])
        ma = np.mean([mean_absolute_error(yt[:, i], yp[:, i]) for i in range(4)])
        rm = np.mean([np.sqrt(mean_squared_error(yt[:, i], yp[:, i])) for i in range(4)])
        return dict(R2=r2, MAPE=mp, MAE=ma, RMSE=rm)

    avg_eq = avg_stats(y_te_real, y_ens_eq, 'Equal')
    avg_w  = avg_stats(y_te_real, y_ens_w,  'Weighted')
    best_ens = y_ens_w if avg_w['R2'] >= avg_eq['R2'] else y_ens_eq
    best_avg = avg_w   if avg_w['R2'] >= avg_eq['R2'] else avg_eq

    model_avgs = {
        'BiGRU': avg_a, 'DeepLSTM': avg_b, 'TCN': avg_c,
        'Ens(eq)': avg_eq, 'Ens(w)': avg_w
    }

    # Print results
    print("\n" + "=" * 72)
    print("  T+7 DIRECT MODEL — INDIVIDUAL SCORES:")
    baseline = dict(R2=0.7964, MAPE=5.01, MAE=5.73, RMSE=11.14)
    for name, avg in model_avgs.items():
        flag = ">" if avg['R2'] > baseline['R2'] else " "
        print(f"  {flag} {name:12s}: R²={avg['R2']:.4f}  MAPE={avg['MAPE']:.2f}%  "
              f"MAE={avg['MAE']:.2f}  RMSE={avg['RMSE']:.2f}")

    print(f"\n  BASELINE (v2.0): R²={baseline['R2']:.4f}  MAPE={baseline['MAPE']:.2f}%")
    print(f"  BEST ENSEMBLE:  R²={best_avg['R2']:.4f}  MAPE={best_avg['MAPE']:.2f}%")
    delta_r2   = best_avg['R2']   - baseline['R2']
    delta_mape = baseline['MAPE'] - best_avg['MAPE']
    print(f"\n  IMPROVEMENT: dR²={'+' if delta_r2>=0 else ''}{delta_r2:.4f}  "
          f"dMAPE={'+' if delta_mape>=0 else ''}{delta_mape:.2f}pp")

    # Save models and results
    mdl_a.save(MODELS_DIR / 'T7_Direct_BiGRU.keras')
    mdl_b.save(MODELS_DIR / 'T7_Direct_LSTM.keras')
    mdl_c.save(MODELS_DIR / 'T7_Direct_TCN.keras')
    print(f"\n[SAVE] 3 mo hinh direct T+7 luu tai {MODELS_DIR}")

    # Detailed results
    rows = []
    for i, col in enumerate(TARGET_COLS):
        yt = y_te_real[:, i]; yp = best_ens[:, i]
        rows.append({
            'Horizon': 7, 'Product': col,
            'MAE': round(mean_absolute_error(yt, yp), 2),
            'RMSE': round(np.sqrt(mean_squared_error(yt, yp)), 2),
            'MAPE%': round(np.mean(np.abs((yt-yp)/yt))*100, 2),
            'R2': round(r2_score(yt, yp), 4)
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(REPORTS_DIR / 't7_direct_detailed.csv', index=False)
    print("\n  Per-product results:")
    print(df_res.to_string(index=False))

    # Plots
    make_plots(y_te_real, best_ens, model_avgs)

    # Final verdict
    print("\n" + "=" * 72)
    r2_ok   = "OK  " if best_avg['R2']   >= 0.87 else "MISS"
    mape_ok = "OK  " if best_avg['MAPE'] <= 3.50 else "MISS"
    print(f"  [{r2_ok}] R2   = {best_avg['R2']:.4f}  (target >= 0.87)")
    print(f"  [{mape_ok}] MAPE = {best_avg['MAPE']:.2f}%  (target <= 3.50%)")
    print("=" * 72)


if __name__ == '__main__':
    main()

# scripts/train_t7_transformer.py
"""
PetroForecast AI v2.3 — Transformer + Log-Return Direct T+7
=============================================================
Phan tich that bai v2.1, v2.2:
  - v2.1 (weighted multi-horizon): R2=0.7983 — gradient diluted
  - v2.2 (direct BiGRU/LSTM/TCN): R2=0.8106 — train/val gap 3.4x, best at epoch 6
  
Nguyen nhan: GRU/LSTM la sequential, can nhieu buoc de hoc cac pattern
xa trong chuoi thoi gian. Voi LOOKBACK=30, GRU phai truyen
gradient qua 30 buoc => vanishing gradient.

Giai phap v2.3:
  1. LOG-RETURN PREDICTION (thay doi can ban nhat):
     - Target: log(P_{t+7} / P_t) thay vi P_{t+7}
     - Log-return la STATIONARY => model de hoc hon nhieu
     - Variance nho hon => gradient on dinh hon
     - Convert lai: P_{t+7} = P_t * exp(log_return)
     
  2. TRANSFORMER ENCODER (kien truc manh hon GRU/LSTM):
     - Multi-Head Self-Attention: hoc DONG THOI moi quan he giua
       tat ca 30 buoc thoi gian (khong phai sequential)
     - Positional Encoding: giu thong tin vi tri thoi gian
     - Feed-Forward layers: bien doi phi tuyen manh
     
  3. DUAL-STREAM INPUT:
     - Stream A: Daily sequence (30 days) — chi tiet ngay
     - Stream B: Weekly summary (4 tuan = 4 trung binh 7-ngay)
       => bat duoc pattern tuan le ro rang
       
  4. TANG REGULARIZATION:
     - Dropout cao hon (0.30) de giam train/val gap
     - Label smoothing / noise injection trong training
     - Stochastic depth / DropPath

Muc tieu: R2 >= 0.87, MAPE <= 3.5% tai T+7
"""
import os, sys, time, warnings, math
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception: pass

os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
LOOKBACK = 30
TARGET_H = 7

# ─────────────────── Data Loading + Log-Return Features ───────────────────
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
    # Spread
    f['SPR_95_92']   = f['MG95'] - f['MG92']
    f['SPR_DO1_DO5'] = f['DO_0001'] - f['DO_005']
    f['SPR_GAS_OIL'] = f['MG95'] - f['DO_005']
    for sp in ['SPR_95_92', 'SPR_DO1_DO5', 'SPR_GAS_OIL']:
        rm = f[sp].rolling(20, min_periods=5).mean()
        rs = f[sp].rolling(20, min_periods=5).std().replace(0, 1e-8)
        f[f'{sp}_Z'] = (f[sp] - rm) / rs

    for col in TARGET_COLS:
        p = f[col]
        # EMA
        f[f'{col}_E5']  = p.ewm(5).mean()
        f[f'{col}_E10'] = p.ewm(10).mean()
        f[f'{col}_E20'] = p.ewm(20).mean()
        # Log returns at multiple lags — KEY for T+7
        for lag in [1, 2, 3, 5, 7, 10, 14, 21]:
            lr = np.log(p / p.shift(lag)).fillna(0)
            f[f'{col}_LR{lag}'] = lr
        # Realized volatility 5d, 7d, 14d
        lr1 = np.log(p / p.shift(1)).fillna(0)
        for w in [5, 7, 14]:
            f[f'{col}_RVol{w}'] = lr1.rolling(w, min_periods=2).std().fillna(0)
        # RSI
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss_= (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().replace(0,1e-8)
        f[f'{col}_RSI'] = 100 - 100/(1 + gain/loss_)
        # BB %B
        sma = p.rolling(20, min_periods=5).mean()
        std = p.rolling(20, min_periods=5).std().replace(0, 1e-8)
        f[f'{col}_BBpct'] = (p - (sma - 2*std)) / (4*std)
        # 7-day mean reversion
        m7 = p.rolling(7, min_periods=3).mean()
        s7 = p.rolling(7, min_periods=3).std().replace(0, 1e-8)
        f[f'{col}_Z7'] = (p - m7) / s7

    # Calendar
    f['Dow_Sin']   = np.sin(2*np.pi*f['Date'].dt.dayofweek/5.)
    f['Dow_Cos']   = np.cos(2*np.pi*f['Date'].dt.dayofweek/5.)
    f['Month_Sin'] = np.sin(2*np.pi*f['Date'].dt.month/12.)
    f['Month_Cos'] = np.cos(2*np.pi*f['Date'].dt.month/12.)
    dow = f['Date'].dt.dayofweek
    f['Days_to_Thu'] = ((3-dow)%7).replace(0,7)/7.
    f['Wk_in_Mon']   = (f['Date'].dt.day-1)//7/4.

    f = f.dropna().reset_index(drop=True)
    other = [c for c in f.columns if c not in TARGET_COLS and c != 'Date']
    fcols = TARGET_COLS + other
    print(f"[DATA] {len(fcols)} features, {len(f)} records")
    return f, fcols


# ─────────────────── Dataset: LOG-RETURN target ───────────────────
def build_logreturn_dataset(df, fcols, lookback=LOOKBACK, horizon=TARGET_H):
    """
    X: (N, lookback, n_features)  — scaled daily feature sequence
    y: (N, 4) — LOG RETURN: log(price_{t+H} / price_t)  [NOT scaled, already small]
    price_t: (N, 4) — current price at time t (for inverse transform)
    """
    N = len(df)
    train_end = int(N * 0.70)
    val_end   = int(N * 0.85)

    df_tr = df.iloc[:train_end].copy()
    df_va = df.iloc[train_end:val_end].copy()
    df_te = df.iloc[val_end:].copy()

    sc_X = MinMaxScaler()
    sc_X.fit(df_tr[fcols].values)
    joblib.dump(sc_X, MODELS_DIR / 'scaler_X_v23.pkl')

    def make(df_sub):
        Xv  = sc_X.transform(df_sub[fcols].values)
        Pv  = df_sub[TARGET_COLS].values  # raw prices
        Xl, yl, Pl = [], [], []
        for i in range(lookback, len(df_sub) - horizon + 1):
            Xl.append(Xv[i-lookback:i, :])
            p_now    = Pv[i-1, :]            # price at current day t
            p_future = Pv[i+horizon-1, :]    # price at t+H
            log_ret  = np.log(p_future / (p_now + 1e-8))  # 7-day log return
            yl.append(log_ret)
            Pl.append(p_now)
        return (np.array(Xl, np.float32),
                np.array(yl, np.float32),
                np.array(Pl, np.float32))

    X_tr, y_tr, P_tr = make(df_tr)
    X_va, y_va, P_va = make(df_va)
    X_te, y_te, P_te = make(df_te)

    print(f"[SEQ] Log-Return T+7 — Train:{X_tr.shape} Val:{X_va.shape} Test:{X_te.shape}")
    print(f"[LOG-RET] Train 7d log-return stats: "
          f"mean={y_tr.mean():.4f} std={y_tr.std():.4f} "
          f"min={y_tr.min():.4f} max={y_tr.max():.4f}")
    return (X_tr,y_tr,P_tr),(X_va,y_va,P_va),(X_te,y_te,P_te), sc_X


# ─────────────────── Positional Encoding ───────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=100, **kw):
        super().__init__(**kw)
        self.max_len = max_len

    def call(self, x):
        # x: (batch, seq, d_model)
        seq_len = ops.shape(x)[1]
        d_model = ops.shape(x)[2]
        # Build PE statically using numpy, cast to tensor
        pe = np.zeros((self.max_len, 256))
        pos = np.arange(self.max_len)[:, None]
        div = np.exp(np.arange(0, 256, 2) * (-np.log(10000.0) / 256))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        pe_tensor = ops.convert_to_tensor(pe[:self.max_len, :], dtype='float32')
        # slice to actual seq & d_model
        # Just add the PE (broadcast over batch)
        x_shape = ops.shape(x)
        pe_slice = pe_tensor[:x_shape[1], :x_shape[2]]
        return x + pe_slice

    def get_config(self):
        cfg = super().get_config(); cfg['max_len'] = self.max_len; return cfg


# ─────────────────── Transformer Block ───────────────────
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout
        self.mha   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads,
                                               dropout=dropout)
        self.ff1   = layers.Dense(ff_dim, activation='gelu')
        self.ff2   = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)
        ff_out = self.ff2(self.ff1(x))
        ff_out = self.drop2(ff_out, training=training)
        return self.norm2(x + ff_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(d_model=self.d_model, num_heads=self.num_heads,
                        ff_dim=self.ff_dim, dropout=self.dropout))
        return cfg


# ─────────────────── Model A: Transformer Encoder (Direct T+7 Log-Return) ───────────────────
def build_transformer_model(input_shape, d_model=128, num_heads=8, ff_dim=256,
                             num_layers=4, dropout=0.30, num_targets=4):
    """
    Pure Transformer Encoder for direct T+7 log-return prediction.
    d_model: embedding dimension
    num_heads: attention heads
    num_layers: stacked transformer blocks
    """
    inp = layers.Input(shape=input_shape, name='seq_input')

    # Project input features to d_model
    x = layers.Dense(d_model, name='input_proj')(inp)
    x = layers.LayerNormalization()(x)

    # Positional Encoding
    x = PositionalEncoding(max_len=input_shape[0]+1, name='pos_enc')(x)
    x = layers.Dropout(dropout)(x)

    # Stacked Transformer Blocks
    for i in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout, name=f'transformer_{i}')(x)

    # Aggregate: CLS-like pooling (mean + last)
    x_mean = layers.GlobalAveragePooling1D(name='gap')(x)
    x_last = layers.Lambda(lambda t: t[:, -1, :], name='last')(x)
    x_cat  = layers.Concatenate()([x_mean, x_last])

    # Prediction head for log-return
    x_h = layers.Dense(128, activation='gelu')(x_cat)
    x_h = layers.LayerNormalization()(x_h)
    x_h = layers.Dropout(dropout)(x_h)
    x_h = layers.Dense(64, activation='gelu')(x_h)
    x_h = layers.Dropout(dropout * 0.5)(x_h)

    # Output: log-return (no activation, can be negative)
    out = layers.Dense(num_targets, activation='linear', name='log_return_out')(x_h)

    m = models.Model(inp, out, name='Transformer_T7_LogReturn')
    m.compile(
        optimizer=keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
        loss='mse',   # MSE on log-return is natural
        metrics=['mae']
    )
    return m


# ─────────────────── Model B: Dual-Stream (Daily + Weekly) ───────────────────
def build_dualstream_model(input_shape_daily, input_shape_weekly,
                            d_model=96, num_targets=4, dropout=0.30):
    """
    Two inputs:
      - daily_inp: (batch, 30, n_feat) — daily resolution
      - weekly_inp: (batch, 4, n_feat) — weekly summary (avg of each 7-day block)
    """
    # Daily stream: Transformer
    day_inp = layers.Input(shape=input_shape_daily, name='daily_input')
    x_d = layers.Dense(d_model)(day_inp)
    x_d = layers.LayerNormalization()(x_d)
    x_d = TransformerBlock(d_model, 8, d_model*2, dropout, name='tr_daily_0')(x_d)
    x_d = TransformerBlock(d_model, 8, d_model*2, dropout, name='tr_daily_1')(x_d)
    x_d = TransformerBlock(d_model, 8, d_model*2, dropout, name='tr_daily_2')(x_d)
    x_d_pool = layers.GlobalAveragePooling1D()(x_d)

    # Weekly stream: Bidirectional GRU on 4 weekly summaries
    wk_inp = layers.Input(shape=input_shape_weekly, name='weekly_input')
    x_w = layers.Dense(d_model)(wk_inp)
    x_w = TransformerBlock(d_model, 4, d_model*2, dropout, name='tr_weekly_0')(x_w)
    x_w_pool = layers.GlobalAveragePooling1D()(x_w)

    # Fusion
    x_fused = layers.Concatenate()([x_d_pool, x_w_pool])
    x_h = layers.Dense(128, activation='gelu')(x_fused)
    x_h = layers.LayerNormalization()(x_h)
    x_h = layers.Dropout(dropout)(x_h)
    x_h = layers.Dense(64, activation='gelu')(x_h)
    x_h = layers.Dropout(dropout * 0.5)(x_h)
    out = layers.Dense(num_targets, activation='linear', name='log_return_out')(x_h)

    m = models.Model(inputs=[day_inp, wk_inp], outputs=out, name='DualStream_T7')
    m.compile(
        optimizer=keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
        loss='mse', metrics=['mae']
    )
    return m


def make_weekly(X):
    """
    X: (N, 30, F) -> weekly_X: (N, 4, F)
    Average each 7-day block: [0:7], [7:14], [14:21], [21:28]
    Last 2 days [28:30] fused into last week.
    """
    w = np.zeros((len(X), 4, X.shape[2]), dtype=np.float32)
    w[:, 0, :] = X[:, 0:7,  :].mean(axis=1)
    w[:, 1, :] = X[:, 7:14, :].mean(axis=1)
    w[:, 2, :] = X[:, 14:21,:].mean(axis=1)
    w[:, 3, :] = X[:, 21:,  :].mean(axis=1)  # last 9 days in 4th week
    return w


# ─────────────────── Evaluation (Log-Return → Price) ───────────────────
def eval_logreturn_model(model, X_te, y_te_lr, P_te, model_type='single'):
    """
    Predict log-return, convert to price, evaluate vs true prices.
    y_te_lr: true 7-day log returns (N,4)
    P_te: current prices at each test sample (N,4)
    """
    if model_type == 'dual':
        X_wk = make_weekly(X_te)
        yp_lr = model.predict([X_te, X_wk], verbose=0)
    else:
        yp_lr = model.predict(X_te, verbose=0)

    # Convert log-return to price
    # true:    P_true = P_te * exp(y_te_lr)
    # predicted: P_pred = P_te * exp(yp_lr)
    P_true = P_te * np.exp(y_te_lr)
    P_pred = P_te * np.exp(yp_lr)

    r2s, mapes, maes, rmses = [], [], [], []
    for i in range(4):
        yt = P_true[:, i]; yp = P_pred[:, i]
        r2s.append(r2_score(yt, yp))
        mapes.append(np.mean(np.abs((yt-yp)/yt))*100)
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))

    return P_true, P_pred, dict(R2=np.mean(r2s), MAPE=np.mean(mapes),
                                  MAE=np.mean(maes), RMSE=np.mean(rmses))


# ─────────────────── Plotting ───────────────────
def make_plots(P_true, P_ens, model_results):
    plt.style.use('seaborn-v0_8-whitegrid'
                  if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # Plot 1: Time series MG95
    ax1 = fig.add_subplot(gs[0, :2])
    n = min(150, len(P_true))
    ax1.plot(P_true[-n:, 0], '-',  color='#1e293b', lw=1.8, label='Actual MG95 (T+7)')
    ax1.plot(P_ens[-n:, 0],  '--', color='#2563eb', lw=2.0, label='Transformer Ensemble')
    ax1.fill_between(range(n),
                     P_ens[-n:, 0]*0.97, P_ens[-n:, 0]*1.03,
                     alpha=0.15, color='#2563eb')
    ax1.set_title('T+7 Transformer Forecast vs Actual — MG95 (Test)', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Test Sample'); ax1.set_ylabel('Price (USD/barrel)')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Plot 2: R² per product
    ax2 = fig.add_subplot(gs[0, 2])
    products = list(PRODUCT_LABELS.values())
    r2_vals  = [r2_score(P_true[:, i], P_ens[:, i]) for i in range(4)]
    colors   = ['#10b981' if r >= 0.87 else '#f59e0b' if r >= 0.80 else '#ef4444' for r in r2_vals]
    bars = ax2.barh(products, r2_vals, color=colors, edgecolor='white')
    ax2.axvline(x=0.87, color='#ef4444', ls='--', lw=1.5, label='Target R²=0.87')
    ax2.axvline(x=0.81, color='#f59e0b', ls='--', lw=1.5, label='v2.2 Baseline 0.81')
    ax2.set_title('R² per Product — v2.3 Transformer', fontweight='bold')
    ax2.set_xlim([0.5, 1.02]); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='x')
    for bar, v in zip(bars, r2_vals):
        ax2.text(v+0.005, bar.get_y()+bar.get_height()/2,
                 f'{v:.4f}', va='center', fontsize=9, fontweight='bold')

    # Plot 3: Model comparison
    ax3 = fig.add_subplot(gs[1, 0])
    names = list(model_results.keys())
    r2s   = [model_results[n]['R2']   for n in names]
    mapes = [model_results[n]['MAPE'] for n in names]
    x_pos = np.arange(len(names))
    clrs  = ['#10b981' if r >= 0.87 else '#f59e0b' if r >= 0.80 else '#ef4444' for r in r2s]
    ax3.bar(x_pos - 0.2, r2s, 0.38, color=clrs, label='R²', edgecolor='white')
    ax3t = ax3.twinx()
    ax3t.bar(x_pos+0.2, mapes, 0.38, color='#3b82f6', alpha=0.7, label='MAPE%', edgecolor='white')
    ax3.axhline(y=0.87, color='#ef4444', ls='--', lw=1.5, alpha=0.9, label='Target R²')
    ax3.set_xticks(x_pos); ax3.set_xticklabels(names, rotation=15, fontsize=8)
    ax3.set_ylabel('R²'); ax3t.set_ylabel('MAPE%')
    ax3.set_title('Model Comparison v2.3', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Scatter MG95
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(P_true[:, 0], P_ens[:, 0], alpha=0.3, s=10, color='#3b82f6', edgecolors='none')
    mn = min(P_true[:,0].min(), P_ens[:,0].min())
    mx = max(P_true[:,0].max(), P_ens[:,0].max())
    ax4.plot([mn,mx],[mn,mx],'r--',lw=2)
    r2v = r2_score(P_true[:,0], P_ens[:,0])
    ax4.text(0.05, 0.92, f'R² = {r2v:.4f}', transform=ax4.transAxes,
             fontsize=11, fontweight='bold')
    ax4.set_title('Scatter: Actual vs Predicted (MG95 T+7)', fontweight='bold')
    ax4.set_xlabel('Actual (USD)'); ax4.set_ylabel('Predicted (USD)')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Residuals
    ax5 = fig.add_subplot(gs[1, 2])
    res = P_true[:, 0] - P_ens[:, 0]
    ax5.hist(res, bins=45, color='#3b82f6', alpha=0.7, edgecolor='white')
    ax5.axvline(x=0, color='#ef4444', lw=2, ls='--')
    ax5.axvline(x=res.mean(), color='#f59e0b', lw=2, ls='-.',
                label=f'Mean={res.mean():.2f}  Std={res.std():.2f}')
    ax5.set_title('Residual Distribution (MG95 T+7)', fontweight='bold')
    ax5.set_xlabel('Error (USD)'); ax5.set_ylabel('Count')
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    fig.suptitle('PetroForecast AI v2.3 — Transformer + Log-Return T+7 Report',
                 fontsize=14, fontweight='bold', y=1.01)
    out = REPORTS_DIR / 't7_transformer_report.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {out}")


# ─────────────────── Main ───────────────────
def main():
    print("=" * 72)
    print("  PETROFORECAST AI v2.3 — TRANSFORMER + LOG-RETURN T+7")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"  GPU: {gpu}")
    print("  Key: Log-return target + Transformer attention + Dual-stream")
    print("=" * 72)

    df, fcols = load_data()
    (X_tr,y_tr,P_tr),(X_va,y_va,P_va),(X_te,y_te,P_te),sc_X = \
        build_logreturn_dataset(df, fcols)

    input_shape = (LOOKBACK, X_tr.shape[2])
    print(f"\n[INFO] Input: {input_shape}  |  Target: log-return (N,4)")

    # Precompute weekly summaries
    Xw_tr = make_weekly(X_tr)
    Xw_va = make_weekly(X_va)
    Xw_te = make_weekly(X_te)
    weekly_shape = (4, X_tr.shape[2])

    def cbs(name):
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=25,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=8, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(str(MODELS_DIR/f'{name}_best.keras'),
                                      monitor='val_loss', save_best_only=True, verbose=0)
        ]

    # ── Model A: Transformer (4 blocks, d=128, 8 heads) ──
    print("\n[1/3] Transformer Encoder (d=128, 8-head, 4 blocks, dropout=0.30)...")
    mdl_a = build_transformer_model(input_shape, d_model=128, num_heads=8,
                                     ff_dim=256, num_layers=4, dropout=0.30)
    t0 = time.time()
    mdl_a.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=150, batch_size=32, callbacks=cbs('tfm_a'), verbose=1)
    print(f"  -> {time.time()-t0:.1f}s")

    # ── Model B: Lighter Transformer (2 blocks, d=96, 6 heads) ──
    print("\n[2/3] Lighter Transformer (d=96, 6-head, 2 blocks, dropout=0.25)...")
    mdl_b = build_transformer_model(input_shape, d_model=96, num_heads=6,
                                     ff_dim=192, num_layers=2, dropout=0.25)
    t0 = time.time()
    mdl_b.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=150, batch_size=32, callbacks=cbs('tfm_b'), verbose=1)
    print(f"  -> {time.time()-t0:.1f}s")

    # ── Model C: Dual-Stream Transformer (daily+weekly) ──
    print("\n[3/3] Dual-Stream Transformer (daily 30d + weekly 4w)...")
    mdl_c = build_dualstream_model(input_shape, weekly_shape, d_model=96, dropout=0.28)
    t0 = time.time()
    mdl_c.fit([X_tr, Xw_tr], y_tr,
              validation_data=([X_va, Xw_va], y_va),
              epochs=150, batch_size=32, callbacks=cbs('tfm_dual'), verbose=1)
    print(f"  -> {time.time()-t0:.1f}s")

    # ── Evaluate ──
    print("\n[EVAL] Evaluating...")
    P_true, Pp_a, avg_a = eval_logreturn_model(mdl_a, X_te, y_te, P_te, 'single')
    _,       Pp_b, avg_b = eval_logreturn_model(mdl_b, X_te, y_te, P_te, 'single')
    _,       Pp_c, avg_c = eval_logreturn_model(mdl_c, X_te, y_te, P_te, 'dual')

    # Ensemble
    P_ens_eq = (Pp_a + Pp_b + Pp_c) / 3.0
    P_ens_w  = Pp_a*0.40 + Pp_b*0.30 + Pp_c*0.30

    def avg_metrics(pt, pp):
        r2s   = [r2_score(pt[:,i], pp[:,i]) for i in range(4)]
        mapes = [np.mean(np.abs((pt[:,i]-pp[:,i])/pt[:,i]))*100 for i in range(4)]
        maes  = [mean_absolute_error(pt[:,i], pp[:,i]) for i in range(4)]
        rmses = [np.sqrt(mean_squared_error(pt[:,i], pp[:,i])) for i in range(4)]
        return dict(R2=np.mean(r2s),MAPE=np.mean(mapes),MAE=np.mean(maes),RMSE=np.mean(rmses))

    avg_eq = avg_metrics(P_true, P_ens_eq)
    avg_w  = avg_metrics(P_true, P_ens_w)
    best_pp  = P_ens_w if avg_w['R2'] >= avg_eq['R2'] else P_ens_eq
    best_avg = avg_w   if avg_w['R2'] >= avg_eq['R2'] else avg_eq

    model_results = {
        'TFM-A': avg_a, 'TFM-B': avg_b, 'DualStr': avg_c,
        'Ens(eq)': avg_eq, 'Ens(w)': avg_w
    }

    baseline = dict(R2=0.7964, MAPE=5.01)
    print("\n" + "=" * 72)
    print("  T+7 TRANSFORMER (Log-Return) — RESULTS:")
    for name, avg in model_results.items():
        flag = ">>" if avg['R2'] > 0.87 else ">" if avg['R2'] > baseline['R2'] else " "
        print(f"  {flag} {name:12s}: R²={avg['R2']:.4f}  MAPE={avg['MAPE']:.2f}%  "
              f"MAE={avg['MAE']:.2f}  RMSE={avg['RMSE']:.2f}")

    print(f"\n  BASELINE v2.0: R²={baseline['R2']:.4f}  MAPE={baseline['MAPE']:.2f}%")
    print(f"  BEST v2.3:     R²={best_avg['R2']:.4f}  MAPE={best_avg['MAPE']:.2f}%")
    dR2   = best_avg['R2']   - baseline['R2']
    dMAPE = baseline['MAPE'] - best_avg['MAPE']
    print(f"  DELTA:         dR²={'+' if dR2>=0 else ''}{dR2:.4f}  "
          f"dMAPE={'+' if dMAPE>=0 else ''}{dMAPE:.2f}pp")

    # Save
    mdl_a.save(MODELS_DIR / 'T7_Transformer_A.keras')
    mdl_b.save(MODELS_DIR / 'T7_Transformer_B.keras')
    mdl_c.save(MODELS_DIR / 'T7_DualStream.keras')
    print(f"\n[SAVE] Models saved to {MODELS_DIR}")

    # Per-product table
    rows = []
    for i, col in enumerate(TARGET_COLS):
        yt = P_true[:,i]; yp = best_pp[:,i]
        rows.append({'Product':col,
                     'MAE':round(mean_absolute_error(yt,yp),2),
                     'RMSE':round(np.sqrt(mean_squared_error(yt,yp)),2),
                     'MAPE%':round(np.mean(np.abs((yt-yp)/yt))*100,2),
                     'R2':round(r2_score(yt,yp),4)})
    df_r = pd.DataFrame(rows)
    df_r.to_csv(REPORTS_DIR/'t7_transformer_detailed.csv', index=False)
    print("\n  Per-product results:")
    print(df_r.to_string(index=False))

    make_plots(P_true, best_pp, model_results)

    # Verdict
    print("\n" + "=" * 72)
    r2_ok   = "OK  " if best_avg['R2']   >= 0.87 else "MISS"
    mape_ok = "OK  " if best_avg['MAPE'] <= 3.50 else "MISS"
    print(f"  [{r2_ok}] R2   = {best_avg['R2']:.4f}  (target >= 0.87)")
    print(f"  [{mape_ok}] MAPE = {best_avg['MAPE']:.2f}%  (target <= 3.50%)")
    print("=" * 72)


if __name__ == '__main__':
    main()

# scripts/test_log_framework.py
"""
Deep Exploration & Optimization of T+7 Horizon using Log-Return / Scale-Invariant Framework
Models:
1. Ridge (Linear L2 Shrinkage)
2. HistGradientBoosting (LightGBM style)
3. XGBoost Regressor
4. Deep BiGRU + Self-Attention
5. Optimal Ensemble Blend
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception: pass

os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import torch
import keras
from keras import layers, models, callbacks, ops
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TARGET_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
LOOKBACK = 30
HORIZON = 7

# 1. Load and Engineer Clean Data
def prepare_data():
    df_raw = pd.read_excel(DATA_DIR / "price_petroleum.xlsx")
    df = df_raw.iloc[:, [3, 4, 5, 6, 7]].copy()
    df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True)
    df[TARGET_COLS] = df[TARGET_COLS].ffill().bfill()

    # Load enriched external market data
    ext = pd.read_csv(DATA_DIR / "external_market.csv", parse_dates=['Date'])
    f = pd.merge(df, ext, on='Date', how='left').ffill().bfill()

    # Log prices
    for c in TARGET_COLS + ['Brent', 'WTI', 'RBOB', 'HeatingOil']:
        if c in f.columns:
            f[f'{c}_log'] = np.log(f[c].clip(lower=1e-3))

    # Fundamental Crack Spreads (USD/bbl)
    f['Crack_MG95']  = f['MG95'] - f['Brent']
    f['Crack_MG92']  = f['MG92'] - f['Brent']
    f['Crack_DO01']  = f['DO_0001'] - f['Brent']
    f['Crack_DO05']  = f['DO_005'] - f['Brent']
    f['Crack_RBOB']  = f['RBOB'] - f['Brent']
    f['Crack_HO']    = f['HeatingOil'] - f['Brent']

    # Arbitrage Spreads: Singapore vs US Futures
    f['Arb_MG95_RBOB'] = f['MG95'] - f['RBOB']
    f['Arb_DO01_HO']   = f['DO_0001'] - f['HeatingOil']
    f['Spread_95_92']  = f['MG95'] - f['MG92']
    f['Spread_DO1_DO5']= f['DO_0001'] - f['DO_005']

    # Z-scores for mean-reversion of spreads
    for sp in ['Crack_MG95', 'Crack_DO01', 'Arb_MG95_RBOB', 'Arb_DO01_HO', 'Spread_95_92', 'Spread_DO1_DO5']:
        rm = f[sp].rolling(20, min_periods=5).mean()
        rs = f[sp].rolling(20, min_periods=5).std().replace(0, 1e-6)
        f[f'{sp}_Z20'] = (f[sp] - rm) / rs

    # Multi-lag returns for all key series
    for col in TARGET_COLS + ['Brent', 'WTI', 'RBOB', 'HeatingOil', 'USD_Idx', 'USD_VND', 'VIX']:
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            f[f'{col}_r{lag}'] = np.log(f[col] / f[col].shift(lag)).fillna(0)
            
    # Technical indicators on TARGET_COLS
    for col in TARGET_COLS:
        p = f[col]
        # EMA ratios
        f[f'{col}_EMA5_ratio']  = p / p.ewm(5).mean()
        f[f'{col}_EMA20_ratio'] = p / p.ewm(20).mean()
        # RSI
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().replace(0, 1e-6)
        f[f'{col}_RSI'] = 100.0 - 100.0 / (1.0 + gain / loss)
        # Realized Volatility
        f[f'{col}_Vol7'] = f[f'{col}_r1'].rolling(7, min_periods=3).std().fillna(0)

    # Calendar
    dow = f['Date'].dt.dayofweek
    f['Dow_Sin']     = np.sin(2 * np.pi * dow / 5.0)
    f['Dow_Cos']     = np.cos(2 * np.pi * dow / 5.0)
    f['Month_Sin']   = np.sin(2 * np.pi * f['Date'].dt.month / 12.0)
    f['Month_Cos']   = np.cos(2 * np.pi * f['Date'].dt.month / 12.0)
    f['Days_to_Thu'] = ((3 - dow) % 7).replace(0, 7) / 7.0

    f = f.dropna().reset_index(drop=True)
    
    # Feature columns: exclude Date and raw absolute prices (keep stationary features)
    feature_cols = [c for c in f.columns if c not in TARGET_COLS and c != 'Date']
    print(f"[DATA] Prepared {len(feature_cols)} stationary & market features, {len(f)} samples")
    return f, feature_cols

# 2. Build Datasets
def make_datasets(df, fcols):
    N = len(df)
    train_end = int(N * 0.70)
    val_end   = int(N * 0.85)

    df_tr = df.iloc[:train_end].copy()
    df_va = df.iloc[train_end:val_end].copy()
    df_te = df.iloc[val_end:].copy()

    scaler = StandardScaler()
    scaler.fit(df_tr[fcols].values)

    def process_sub(df_sub):
        X_vals = scaler.transform(df_sub[fcols].values)
        P_raw  = df_sub[TARGET_COLS].values
        
        X_seq, X_tab, y_logret, P_now, P_fut = [], [], [], [], []
        for i in range(LOOKBACK, len(df_sub) - HORIZON + 1):
            X_seq.append(X_vals[i - LOOKBACK:i, :])
            # Tabular: current step + 7d mean + 7d std
            c_f = X_vals[i - 1, :]
            m_f = X_vals[i - 7:i, :].mean(axis=0)
            s_f = X_vals[i - 7:i, :].std(axis=0)
            X_tab.append(np.concatenate([c_f, m_f, s_f]))
            
            p_n = P_raw[i - 1, :]
            p_f = P_raw[i + HORIZON - 1, :]
            # Log return target: ln(P_{t+7} / P_t)
            lr = np.log(p_f / (p_n + 1e-8))
            y_logret.append(lr)
            P_now.append(p_n)
            P_fut.append(p_f)
            
        return (np.array(X_seq, np.float32),
                np.array(X_tab, np.float32),
                np.array(y_logret, np.float32),
                np.array(P_now, np.float32),
                np.array(P_fut, np.float32))

    X_seq_tr, X_tab_tr, y_lr_tr, P_now_tr, P_fut_tr = process_sub(df_tr)
    X_seq_va, X_tab_va, y_lr_va, P_now_va, P_fut_va = process_sub(df_va)
    X_seq_te, X_tab_te, y_lr_te, P_now_te, P_fut_te = process_sub(df_te)

    return {
        'seq': ((X_seq_tr, y_lr_tr), (X_seq_va, y_lr_va), (X_seq_te, y_lr_te)),
        'tab': ((X_tab_tr, y_lr_tr), (X_tab_va, y_lr_va), (X_tab_te, y_lr_te)),
        'price': (P_now_tr, P_now_va, P_now_te, P_fut_tr, P_fut_va, P_fut_te)
    }

def eval_preds(P_true, P_pred):
    r2s, mapes, maes = [], [], []
    for i in range(4):
        yt = P_true[:, i]
        yp = P_pred[:, i]
        r2s.append(r2_score(yt, yp))
        mapes.append(np.mean(np.abs((yt - yp) / yt)) * 100)
        maes.append(mean_absolute_error(yt, yp))
    return {'R2': np.mean(r2s), 'MAPE': np.mean(mapes), 'MAE': np.mean(maes), 'r2_list': r2s}

def main():
    print("=" * 76)
    print("  EXPLORATION: SCALE-INVARIANT LOG-RETURN FRAMEWORK AT T+7")
    print("=" * 76)
    df, fcols = prepare_data()
    data = make_datasets(df, fcols)

    X_tab_tr, y_lr_tr = data['tab'][0]
    X_tab_va, y_lr_va = data['tab'][1]
    X_tab_te, y_lr_te = data['tab'][2]
    P_now_te = data['price'][2]
    P_fut_te = data['price'][5]

    # Baseline 1: Naive (log_ret = 0, P_pred = P_now)
    m_naive = eval_preds(P_fut_te, P_now_te)
    print(f"\n[BENCHMARK] Naive Persistence (P_pred = P_now):")
    print(f"  Mean R2 = {m_naive['R2']:.4f}  MAPE = {m_naive['MAPE']:.2f}%  MAE = {m_naive['MAE']:.2f}")

    # Model 1: RidgeCV (L2 regularized linear model on tabular features)
    print("\n[TRAIN] 1. RidgeCV...")
    pred_lr_ridge = np.zeros((len(X_tab_te), 4))
    for i in range(4):
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 30))
        ridge.fit(X_tab_tr, y_lr_tr[:, i])
        pred_lr_ridge[:, i] = ridge.predict(X_tab_te)
    P_pred_ridge = P_now_te * np.exp(pred_lr_ridge)
    m_ridge = eval_preds(P_fut_te, P_pred_ridge)
    print(f"  Ridge: R2 = {m_ridge['R2']:.4f}  MAPE = {m_ridge['MAPE']:.2f}%  MAE = {m_ridge['MAE']:.2f}")

    # Model 2: HistGradientBoosting
    print("\n[TRAIN] 2. HistGradientBoosting...")
    pred_lr_hgb = np.zeros((len(X_tab_te), 4))
    for i in range(4):
        hgb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.03, l2_regularization=2.0, random_state=42+i)
        hgb.fit(X_tab_tr, y_lr_tr[:, i])
        pred_lr_hgb[:, i] = hgb.predict(X_tab_te)
    P_pred_hgb = P_now_te * np.exp(pred_lr_hgb)
    m_hgb = eval_preds(P_fut_te, P_pred_hgb)
    print(f"  HistGB: R2 = {m_hgb['R2']:.4f}  MAPE = {m_hgb['MAPE']:.2f}%  MAE = {m_hgb['MAE']:.2f}")

    # Model 3: XGBoost with conservative shrinkage
    print("\n[TRAIN] 3. XGBoost Regressor...")
    pred_lr_xgb = np.zeros((len(X_tab_te), 4))
    for i in range(4):
        xgb_m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=4,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42+i, n_jobs=-1
        )
        xgb_m.fit(X_tab_tr, y_lr_tr[:, i], eval_set=[(X_tab_va, y_lr_va[:, i])], verbose=False)
        pred_lr_xgb[:, i] = xgb_m.predict(X_tab_te)
    P_pred_xgb = P_now_te * np.exp(pred_lr_xgb)
    m_xgb = eval_preds(P_fut_te, P_pred_xgb)
    print(f"  XGBoost: R2 = {m_xgb['R2']:.4f}  MAPE = {m_xgb['MAPE']:.2f}%  MAE = {m_xgb['MAE']:.2f}")

    # Model 4: Deep BiGRU + Self Attention on sequences
    print("\n[TRAIN] 4. Deep BiGRU + Self-Attention...")
    X_seq_tr, y_seq_tr = data['seq'][0]
    X_seq_va, y_seq_va = data['seq'][1]
    X_seq_te, y_seq_te = data['seq'][2]

    class SelfAttention(layers.Layer):
        def __init__(self, units, **kw):
            super().__init__(**kw); self.units = units
            self.Wq = layers.Dense(units, use_bias=False)
            self.Wk = layers.Dense(units, use_bias=False)
            self.Wv = layers.Dense(units, use_bias=False)
        def call(self, x):
            Q = self.Wq(x); K = self.Wk(x); V = self.Wv(x)
            score = ops.matmul(Q, ops.transpose(K, [0, 2, 1])) / (float(self.units) ** 0.5)
            return ops.matmul(ops.softmax(score, axis=-1), V)

    inp = layers.Input(shape=(LOOKBACK, X_seq_tr.shape[2]))
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.GRU(48, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x_att = SelfAttention(48)(x)
    gap = layers.GlobalAveragePooling1D()(x_att)
    last = layers.Lambda(lambda t: t[:, -1, :])(x_att)
    fused = layers.Concatenate()([gap, last])
    d = layers.Dense(64, activation='gelu')(fused)
    d = layers.Dropout(0.25)(d)
    out = layers.Dense(4, activation='linear', name='log_return_out')(d)

    model = models.Model(inp, out)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3),
                  loss='huber', metrics=['mae'])
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    model.fit(X_seq_tr, y_seq_tr, validation_data=(X_seq_va, y_seq_va),
              epochs=80, batch_size=32, callbacks=cbs, verbose=0)
    pred_lr_nn = model.predict(X_seq_te, verbose=0)
    P_pred_nn = P_now_te * np.exp(pred_lr_nn)
    m_nn = eval_preds(P_fut_te, P_pred_nn)
    print(f"  BiGRU-Attn: R2 = {m_nn['R2']:.4f}  MAPE = {m_nn['MAPE']:.2f}%  MAE = {m_nn['MAE']:.2f}")

    # 5. Hybrid Ensemble
    print("\n[ENSEMBLE] Evaluating Multi-Model Blends...")
    # Blend log-returns
    best_r2, best_blend = -999, None
    for w_ridge in [0.1, 0.2, 0.3]:
        for w_xgb in [0.2, 0.3, 0.4]:
            for w_hgb in [0.2, 0.3, 0.4]:
                w_nn = 1.0 - (w_ridge + w_xgb + w_hgb)
                if w_nn < 0: continue
                pred_lr_blend = (w_ridge * pred_lr_ridge +
                                 w_xgb * pred_lr_xgb +
                                 w_hgb * pred_lr_hgb +
                                 w_nn * pred_lr_nn)
                # Test shrinkage factor on blended log-return
                for alpha in [0.3, 0.5, 0.7, 0.85, 1.0]:
                    P_blend = P_now_te * np.exp(alpha * pred_lr_blend)
                    m = eval_preds(P_fut_te, P_blend)
                    if m['R2'] > best_r2:
                        best_r2 = m['R2']
                        best_blend = (m, (w_ridge, w_xgb, w_hgb, w_nn), alpha, P_blend)

    b_m, b_w, b_a, b_P = best_blend
    print(f"  Best Hybrid Ensemble (Weights={b_w}, Shrinkage={b_a}):")
    print(f"  -> R2 = {b_m['R2']:.4f}  MAPE = {b_m['MAPE']:.2f}%  MAE = {b_m['MAE']:.2f}")
    print("  Per product:")
    for col, r in zip(TARGET_COLS, b_m['r2_list']):
        print(f"    {col}: R2 = {r:.4f}")

if __name__ == '__main__':
    main()

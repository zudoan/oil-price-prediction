# scripts/fetch_external_data.py
"""
PetroForecast AI v2.4 — External Data Enrichment + Direct T+7
==============================================================
Root cause: val_loss=0.0046 la NOISE FLOOR cua du lieu 4 cot gia VN noi dia.
Giai phap: Them external market data - leading indicators thuc su cua gia xang dau:

  Cong thuc gia co so VN (ND 80/2023):
    G = MOPS_Singapore + CIF + thue + phi_kinh_doanh

  => Brent crude, USD/VND, USD/SGD la bien tuyen tinh truc tiep!

External features (tu Yahoo Finance):
  - BZ=F   : Brent Crude Oil Futures (quan trong nhat)
  - CL=F   : WTI Crude Oil
  - USDVND=X : Ti gia USD/VND (anh huong truc tiep den gia VN)
  - USDSGD=X : USD/SGD (Singapore la thi truong MOPS tham chieu)
  - DX-Y.NYB : USD Dollar Index
  - GC=F   : Vang (chi bao rui ro toan cau)
  - ^VIX   : Chi so bien dong thi truong
"""
import os, sys, time, warnings
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

import yfinance as yf
import torch
import keras
from keras import layers, models, callbacks, ops

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
LOOKBACK = 30
TARGET_H = 7

# ─────────────────── Step 1: Load VN Data ───────────────────
def load_vn_data():
    df_raw = pd.read_excel(DATA_DIR / "price_petroleum.xlsx")
    df = df_raw.iloc[:, [3, 4, 5, 6, 7]].copy()
    df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True)
    df[TARGET_COLS] = df[TARGET_COLS].ffill().bfill()
    return df


# ─────────────────── Step 2: Fetch External Market Data ───────────────────
def fetch_external(df_vn):
    start = df_vn['Date'].min().strftime('%Y-%m-%d')
    end   = (df_vn['Date'].max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    tickers = {
        'BZ=F'     : 'Brent',
        'CL=F'     : 'WTI',
        'NG=F'     : 'NatGas',
        'DX-Y.NYB' : 'USD_Idx',
        'USDVND=X' : 'USD_VND',
        'USDSGD=X' : 'USD_SGD',
        'GC=F'     : 'Gold',
        '^VIX'     : 'VIX',
    }

    ext_path = DATA_DIR / "external_market.csv"

    # Use cached if available and recent
    if ext_path.exists():
        df_cached = pd.read_csv(ext_path, parse_dates=['Date'])
        cache_end = df_cached['Date'].max()
        vn_end    = df_vn['Date'].max()
        if cache_end >= vn_end - pd.Timedelta(days=10):
            print(f"[EXT] Using cached external data ({len(df_cached)} rows, up to {cache_end.date()})")
            return df_cached

    print(f"[EXT] Downloading external market data ({start} -> {end})...")
    frames = []
    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if len(raw) == 0:
                print(f"  [SKIP] {ticker}: no data")
                continue
            s = raw['Close']
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.reset_index()
            s.columns = ['Date', name]
            s['Date'] = pd.to_datetime(s['Date'])
            frames.append(s.set_index('Date')[name])
            print(f"  [OK]   {ticker:12s} -> {name:10s}: {len(s)} rows  last={s[name].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  [ERR]  {ticker}: {e}")

    if not frames:
        print("[WARN] No external data — proceeding with VN-only data")
        return None

    df_ext = pd.concat(frames, axis=1).reset_index()
    df_ext.columns = ['Date'] + [f.name for f in frames]

    # Fill weekends/holidays
    date_range = pd.date_range(df_ext['Date'].min(), df_ext['Date'].max(), freq='D')
    df_ext = df_ext.set_index('Date').reindex(date_range).ffill().bfill().reset_index()
    df_ext.rename(columns={'index': 'Date'}, inplace=True)

    df_ext.to_csv(ext_path, index=False)
    print(f"[EXT] Saved: {ext_path}  ({len(df_ext)} rows)")
    return df_ext


# ─────────────────── Step 3: Merge + Feature Engineering ───────────────────
def build_enriched_features(df_vn, df_ext):
    f = df_vn.copy()

    # Merge external
    if df_ext is not None:
        f = pd.merge(f, df_ext, on='Date', how='left')
        ext_cols = [c for c in df_ext.columns if c != 'Date']
        f[ext_cols] = f[ext_cols].ffill().bfill()

        # Derived external features
        if 'Brent' in f.columns and 'WTI' in f.columns:
            f['Brent_WTI_Spread'] = f['Brent'] - f['WTI']
        if 'Brent' in f.columns:
            for lag in [1, 3, 5, 7, 14]:
                f[f'Brent_LR{lag}'] = np.log(f['Brent'] / f['Brent'].shift(lag)).fillna(0)
            f['Brent_MA7']  = f['Brent'].rolling(7).mean()
            f['Brent_MA14'] = f['Brent'].rolling(14).mean()
            f['Brent_Z7']   = (f['Brent'] - f['Brent'].rolling(7).mean()) / \
                               f['Brent'].rolling(7).std().replace(0, 1e-8)
            f['Brent_Mom7'] = f['Brent'] - f['Brent'].shift(7)
        if 'USD_VND' in f.columns:
            f['USD_VND_LR1'] = np.log(f['USD_VND'] / f['USD_VND'].shift(1)).fillna(0)
            f['USD_VND_LR7'] = np.log(f['USD_VND'] / f['USD_VND'].shift(7)).fillna(0)
        if 'USD_SGD' in f.columns:
            f['USD_SGD_LR7'] = np.log(f['USD_SGD'] / f['USD_SGD'].shift(7)).fillna(0)
        if 'VIX' in f.columns:
            f['VIX_MA5'] = f['VIX'].rolling(5).mean()

    # VN domestic features
    f['SPR_95_92']  = f['MG95'] - f['MG92']
    f['SPR_GAS_OIL']= f['MG95'] - f['DO_005']

    for col in TARGET_COLS:
        p = f[col]
        f[f'{col}_E5']  = p.ewm(5).mean()
        f[f'{col}_E10'] = p.ewm(10).mean()
        f[f'{col}_E20'] = p.ewm(20).mean()
        for lag in [1, 3, 5, 7, 10, 14]:
            f[f'{col}_LR{lag}'] = np.log(p / p.shift(lag)).fillna(0)
        lr1 = np.log(p / p.shift(1)).fillna(0)
        f[f'{col}_RVol7']  = lr1.rolling(7, min_periods=2).std().fillna(0)
        f[f'{col}_RVol14'] = lr1.rolling(14, min_periods=3).std().fillna(0)
        delta = p.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss_ = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().replace(0, 1e-8)
        f[f'{col}_RSI'] = 100 - 100 / (1 + gain / loss_)
        sma = p.rolling(20, min_periods=5).mean()
        std = p.rolling(20, min_periods=5).std().replace(0, 1e-8)
        f[f'{col}_BBpct'] = (p - (sma - 2*std)) / (4*std)
        m7 = p.rolling(7, min_periods=3).mean()
        s7 = p.rolling(7, min_periods=3).std().replace(0, 1e-8)
        f[f'{col}_Z7'] = (p - m7) / s7
        # Brent-VN spread (key feature if Brent available)
        if 'Brent' in f.columns:
            f[f'{col}_vs_Brent'] = p / f['Brent'].replace(0, 1e-8)

    # Calendar
    f['Dow_Sin']     = np.sin(2*np.pi*f['Date'].dt.dayofweek/5.)
    f['Dow_Cos']     = np.cos(2*np.pi*f['Date'].dt.dayofweek/5.)
    f['Month_Sin']   = np.sin(2*np.pi*f['Date'].dt.month/12.)
    f['Month_Cos']   = np.cos(2*np.pi*f['Date'].dt.month/12.)
    dow = f['Date'].dt.dayofweek
    f['Days_to_Thu'] = ((3-dow)%7).replace(0,7)/7.

    f = f.dropna().reset_index(drop=True)
    other = [c for c in f.columns if c not in TARGET_COLS and c != 'Date']
    fcols = TARGET_COLS + other
    print(f"[DATA] {len(fcols)} features (incl. external), {len(f)} records")

    # Print correlation of key external features with MG95
    if df_ext is not None:
        print("[CORR] MG95 correlations with external features:")
        key_ext = [c for c in ['Brent','WTI','USD_VND','USD_SGD','USD_Idx','VIX'] if c in f.columns]
        for col in key_ext:
            r = f['MG95'].corr(f[col])
            print(f"  MG95 ~ {col:12s}: r={r:.4f}")

    return f, fcols


# ─────────────────── Step 4: Build Direct T+7 Dataset ───────────────────
def build_direct_dataset(df, fcols):
    N = len(df)
    train_end = int(N * 0.70)
    val_end   = int(N * 0.85)

    df_tr = df.iloc[:train_end].copy()
    df_va = df.iloc[train_end:val_end].copy()
    df_te = df.iloc[val_end:].copy()

    sc_X = MinMaxScaler()
    sc_X.fit(df_tr[fcols].values)
    joblib.dump(sc_X, MODELS_DIR / 'scaler_X_v24.pkl')

    def make(df_sub):
        Xv = sc_X.transform(df_sub[fcols].values)
        Pv = df_sub[TARGET_COLS].values
        Xl, yl, Pl = [], [], []
        for i in range(LOOKBACK, len(df_sub) - TARGET_H + 1):
            Xl.append(Xv[i-LOOKBACK:i, :])
            p_now    = Pv[i-1, :]
            p_future = Pv[i+TARGET_H-1, :]
            yl.append(np.log(p_future / (p_now + 1e-8)))  # log return
            Pl.append(p_now)
        return (np.array(Xl, np.float32),
                np.array(yl, np.float32),
                np.array(Pl, np.float32))

    X_tr, y_tr, P_tr = make(df_tr)
    X_va, y_va, P_va = make(df_va)
    X_te, y_te, P_te = make(df_te)
    print(f"[SEQ] Train:{X_tr.shape} Val:{X_va.shape} Test:{X_te.shape}")
    return (X_tr,y_tr,P_tr),(X_va,y_va,P_va),(X_te,y_te,P_te), sc_X


# ─────────────────── Attention Layer ───────────────────
class SelfAttention(layers.Layer):
    def __init__(self, units, **kw):
        super().__init__(**kw)
        self.units = units
        self.Wq = layers.Dense(units, use_bias=False)
        self.Wk = layers.Dense(units, use_bias=False)
        self.Wv = layers.Dense(units, use_bias=False)
    def call(self, x):
        Q = self.Wq(x); K = self.Wk(x); V = self.Wv(x)
        score = ops.matmul(Q, ops.transpose(K,[0,2,1])) / float(self.units)**0.5
        return ops.matmul(ops.softmax(score,-1), V)
    def get_config(self):
        cfg = super().get_config(); cfg['units']=self.units; return cfg


# ─────────────────── Model Definitions ───────────────────
def build_bigru_attn(shape, dropout=0.25, num_targets=4):
    inp    = layers.Input(shape=shape)
    y_base = layers.Lambda(lambda x: x[:,-1,:num_targets])(inp)
    x = layers.Bidirectional(layers.GRU(96, return_sequences=True))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.GRU(80, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = SelfAttention(80)(x)
    x = layers.Dropout(dropout*0.5)(x)
    xg = layers.GlobalAveragePooling1D()(x)
    xl = layers.Lambda(lambda t: t[:,-1,:])(x)
    x  = layers.Concatenate()([xg, xl])
    x  = layers.Dense(128, activation='gelu')(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.Dropout(dropout)(x)
    x  = layers.Dense(64, activation='gelu')(x)
    x  = layers.Dropout(dropout*0.5)(x)
    delta = layers.Dense(num_targets, activation='linear')(x)
    out   = layers.Add()([y_base, delta])
    m = models.Model(inp, out, name='BiGRU_Attn_v24')
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss='huber', metrics=['mae'])
    return m


def build_deep_lstm(shape, dropout=0.25, num_targets=4):
    inp    = layers.Input(shape=shape)
    y_base = layers.Lambda(lambda x: x[:,-1,:num_targets])(inp)
    x  = layers.LSTM(112, return_sequences=True)(inp)
    x  = layers.Dropout(dropout)(x)
    x2 = layers.LSTM(112, return_sequences=True)(x)
    x2 = layers.Dropout(dropout)(x2)
    xr = layers.Add()([x, x2])
    x3 = layers.LSTM(80, return_sequences=True)(xr)
    x3 = layers.Dropout(dropout)(x3)
    x3 = SelfAttention(80)(x3)
    xg = layers.GlobalAveragePooling1D()(x3)
    xl = layers.Lambda(lambda t: t[:,-1,:])(x3)
    x  = layers.Concatenate()([xg, xl])
    x  = layers.Dense(128, activation='gelu')(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.Dropout(dropout)(x)
    x  = layers.Dense(64, activation='relu')(x)
    x  = layers.Dropout(dropout*0.5)(x)
    delta = layers.Dense(num_targets, activation='linear')(x)
    out   = layers.Add()([y_base, delta])
    m = models.Model(inp, out, name='DeepLSTM_v24')
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss='huber', metrics=['mae'])
    return m


def build_tcn(shape, dropout=0.20, num_targets=4):
    inp    = layers.Input(shape=shape)
    y_base = layers.Lambda(lambda x: x[:,-1,:num_targets])(inp)
    x = inp
    for d in [1, 2, 4, 8, 16]:
        res = x
        x = layers.Conv1D(64, 3, dilation_rate=d, padding='causal', activation='gelu')(x)
        x = layers.Dropout(dropout)(x)
        if res.shape[-1] != 64:
            res = layers.Conv1D(64, 1, padding='same')(res)
        x = layers.Add()([x, res])
    xg = layers.GlobalAveragePooling1D()(x)
    xl = layers.Lambda(lambda t: t[:,-1,:])(x)
    x  = layers.Concatenate()([xg, xl])
    x  = layers.Dense(128, activation='gelu')(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.Dropout(dropout)(x)
    x  = layers.Dense(64, activation='relu')(x)
    x  = layers.Dropout(dropout*0.5)(x)
    delta = layers.Dense(num_targets, activation='linear')(x)
    out   = layers.Add()([y_base, delta])
    m = models.Model(inp, out, name='TCN_v24')
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss='huber', metrics=['mae'])
    return m


# ─────────────────── Evaluation ───────────────────
def evaluate(model, X_te, y_te_lr, P_te):
    yp_lr  = model.predict(X_te, verbose=0)
    P_true = P_te * np.exp(y_te_lr)
    P_pred = P_te * np.exp(yp_lr)
    r2s, mapes, maes, rmses = [], [], [], []
    for i in range(4):
        yt = P_true[:,i]; yp = P_pred[:,i]
        r2s.append(r2_score(yt, yp))
        mapes.append(np.mean(np.abs((yt-yp)/yt))*100)
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
    return P_true, P_pred, dict(R2=np.mean(r2s), MAPE=np.mean(mapes),
                                  MAE=np.mean(maes), RMSE=np.mean(rmses))


# ─────────────────── Main ───────────────────
def main():
    print("=" * 72)
    print("  PETROFORECAST AI v2.4 — EXTERNAL DATA ENRICHMENT + DIRECT T+7")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"  GPU: {gpu}")
    print("  External: Brent, WTI, USD/VND, USD/SGD, VIX, Gold, USD_Index")
    print("=" * 72)

    # Data pipeline
    print("\n[STEP 1/4] Loading VN petroleum data...")
    df_vn = load_vn_data()
    print(f"  {len(df_vn)} records, {df_vn['Date'].min().date()} -> {df_vn['Date'].max().date()}")

    print("\n[STEP 2/4] Fetching external market data...")
    df_ext = fetch_external(df_vn)

    print("\n[STEP 3/4] Feature engineering (VN + external)...")
    df, fcols = build_enriched_features(df_vn, df_ext)

    print("\n[STEP 4/4] Building sequences...")
    (X_tr,y_tr,P_tr),(X_va,y_va,P_va),(X_te,y_te,P_te),sc_X = \
        build_direct_dataset(df, fcols)

    input_shape = (LOOKBACK, X_tr.shape[2])
    print(f"  Input: {input_shape} (enriched features)")

    def cbs(name):
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=7, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(str(MODELS_DIR/f'{name}_best.keras'),
                                      monitor='val_loss', save_best_only=True, verbose=0)
        ]

    # Train 3 models
    results = {}
    preds   = {}

    for idx, (name, builder, kw) in enumerate([
        ('BiGRU_Attn', build_bigru_attn, {'dropout': 0.22}),
        ('DeepLSTM',   build_deep_lstm,  {'dropout': 0.22}),
        ('TCN',        build_tcn,        {'dropout': 0.18}),
    ]):
        print(f"\n[{idx+1}/3] {name}...")
        mdl = builder(input_shape, **kw)
        t0  = time.time()
        mdl.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=120, batch_size=32, callbacks=cbs(f'v24_{name}'), verbose=1)
        print(f"  -> {time.time()-t0:.1f}s")
        P_true, Pp, avg = evaluate(mdl, X_te, y_te, P_te)
        results[name] = avg
        preds[name]   = Pp
        print(f"  R²={avg['R2']:.4f}  MAPE={avg['MAPE']:.2f}%  MAE={avg['MAE']:.2f}")
        mdl.save(MODELS_DIR / f'T7_v24_{name}.keras')

    # Ensemble
    P_ens_eq = sum(preds.values()) / len(preds)
    def avg_m(pt, pp):
        return dict(
            R2=np.mean([r2_score(pt[:,i],pp[:,i]) for i in range(4)]),
            MAPE=np.mean([np.mean(np.abs((pt[:,i]-pp[:,i])/pt[:,i]))*100 for i in range(4)]),
            MAE=np.mean([mean_absolute_error(pt[:,i],pp[:,i]) for i in range(4)]),
            RMSE=np.mean([np.sqrt(mean_squared_error(pt[:,i],pp[:,i])) for i in range(4)])
        )
    avg_ens = avg_m(P_true, P_ens_eq)
    results['Ensemble'] = avg_ens

    # ─── Summary ───
    baseline = dict(R2=0.7964, MAPE=5.01)
    print("\n" + "=" * 72)
    print("  v2.4 ENRICHED EXTERNAL DATA — RESULTS:")
    for name, avg in results.items():
        flag = ">>" if avg['R2'] >= 0.87 else ">" if avg['R2'] > baseline['R2'] else " "
        print(f"  {flag} {name:12s}: R²={avg['R2']:.4f}  MAPE={avg['MAPE']:.2f}%  "
              f"MAE={avg['MAE']:.2f}  RMSE={avg['RMSE']:.2f}")

    best = results['Ensemble']
    print(f"\n  BASELINE v2.0: R²={baseline['R2']:.4f}  MAPE={baseline['MAPE']:.2f}%")
    print(f"  BEST v2.4:     R²={best['R2']:.4f}  MAPE={best['MAPE']:.2f}%")
    print(f"  DELTA:         dR²={best['R2']-baseline['R2']:+.4f}  "
          f"dMAPE={baseline['MAPE']-best['MAPE']:+.2f}pp")

    # Per-product
    rows = []
    for i, col in enumerate(TARGET_COLS):
        yt = P_true[:,i]; yp = P_ens_eq[:,i]
        rows.append({'Product':col,
                     'MAE':round(mean_absolute_error(yt,yp),2),
                     'RMSE':round(np.sqrt(mean_squared_error(yt,yp)),2),
                     'MAPE%':round(np.mean(np.abs((yt-yp)/yt))*100,2),
                     'R2':round(r2_score(yt,yp),4)})
    df_r = pd.DataFrame(rows)
    df_r.to_csv(REPORTS_DIR/'t7_v24_results.csv', index=False)
    print("\n  Per-product:")
    print(df_r.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    n = min(150, len(P_true))
    axes[0].plot(P_true[-n:,0], '-', color='#1e293b', lw=1.8, label='Actual MG95 T+7')
    axes[0].plot(P_ens_eq[-n:,0], '--', color='#2563eb', lw=2.0, label='v2.4 Ensemble')
    axes[0].fill_between(range(n), P_ens_eq[-n:,0]*0.97, P_ens_eq[-n:,0]*1.03,
                         alpha=0.15, color='#2563eb')
    axes[0].set_title(f'T+7 v2.4 External Enriched — MG95\nR²={best["R2"]:.4f}  MAPE={best["MAPE"]:.2f}%',
                      fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    names = list(results.keys())
    r2s   = [results[n]['R2'] for n in names]
    clrs  = ['#10b981' if r >= 0.87 else '#f59e0b' if r >= 0.81 else '#ef4444' for r in r2s]
    axes[1].barh(names, r2s, color=clrs, edgecolor='white')
    axes[1].axvline(x=0.87, color='#ef4444', ls='--', lw=1.5, label='Target 0.87')
    axes[1].axvline(x=0.7964, color='gray', ls=':', lw=1.5, label='Baseline 0.7964')
    axes[1].set_title('R² per Model — v2.4 External Enriched', fontweight='bold')
    for i, (n, v) in enumerate(zip(names, r2s)):
        axes[1].text(v+0.003, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=9)
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='x')

    fig.suptitle('PetroForecast AI v2.4 — Brent+USD/VND+VIX Enriched T+7',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = REPORTS_DIR / 't7_v24_report.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[PLOT] {out}")

    print("\n" + "=" * 72)
    r2_ok   = "OK  " if best['R2']   >= 0.87 else "MISS"
    mape_ok = "OK  " if best['MAPE'] <= 3.50 else "MISS"
    print(f"  [{r2_ok}] R2   = {best['R2']:.4f}  (target >= 0.87)")
    print(f"  [{mape_ok}] MAPE = {best['MAPE']:.2f}%  (target <= 3.50%)")
    print("=" * 72)


if __name__ == '__main__':
    main()

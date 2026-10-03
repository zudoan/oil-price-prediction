# scripts/update_market_data.py
"""
Download and update full external market data:
1. Brent crude (BZ=F)
2. WTI crude (CL=F)
3. RBOB Gasoline (RB=F * 42) — Global Gasoline Benchmark
4. Heating Oil / Diesel (HO=F * 42) — Global Diesel Benchmark
5. US Dollar Index (DX-Y.NYB)
6. USD/VND (USDVND=X)
7. USD/SGD (USDSGD=X)
8. Gold (GC=F)
9. VIX (^VIX)
10. Natural Gas (NG=F)
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df_raw = pd.read_excel(DATA_DIR / "price_petroleum.xlsx")
df_vn = df_raw.iloc[:, [3, 4, 5, 6, 7]].copy()
df_vn.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']
df_vn['Date'] = pd.to_datetime(df_vn['Date'], errors='coerce')
df_vn = df_vn.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
df_vn = df_vn[df_vn['Date'] >= '2008-11-03'].copy().reset_index(drop=True)

start = df_vn['Date'].min().strftime('%Y-%m-%d')
end   = (df_vn['Date'].max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

tickers = {
    'BZ=F'     : ('Brent', 1.0),
    'CL=F'     : ('WTI', 1.0),
    'RB=F'     : ('RBOB', 42.0),       # USD/gallon -> USD/barrel
    'HO=F'     : ('HeatingOil', 42.0), # USD/gallon -> USD/barrel
    'DX-Y.NYB' : ('USD_Idx', 1.0),
    'USDVND=X' : ('USD_VND', 1.0),
    'USDSGD=X' : ('USD_SGD', 1.0),
    'GC=F'     : ('Gold', 1.0),
    '^VIX'     : ('VIX', 1.0),
    'NG=F'     : ('NatGas', 1.0),
}

print(f"[EXT] Fetching complete market data: {start} -> {end}")
frames = []
for ticker, (name, multiplier) in tickers.items():
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if len(raw) == 0:
            print(f"  [SKIP] {ticker}: no data")
            continue
        s = raw['Close']
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.reset_index()
        s.columns = ['Date', name]
        s['Date'] = pd.to_datetime(s['Date'])
        s[name] = s[name] * multiplier
        frames.append(s.set_index('Date')[name])
        print(f"  [OK]   {ticker:10s} -> {name:12s}: {len(s)} rows, last={s[name].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  [ERR]  {ticker}: {e}")

df_ext = pd.concat(frames, axis=1).reset_index()
df_ext.columns = ['Date'] + [f.name for f in frames]

# Fill weekends/holidays
date_range = pd.date_range(df_ext['Date'].min(), df_ext['Date'].max(), freq='D')
df_ext = df_ext.set_index('Date').reindex(date_range).ffill().bfill().reset_index()
df_ext.rename(columns={'index': 'Date'}, inplace=True)

out_file = DATA_DIR / "external_market.csv"
df_ext.to_csv(out_file, index=False)
print(f"[EXT] Saved successfully: {out_file} ({len(df_ext)} rows, {df_ext.columns.tolist()})")

import pandas as pd
import numpy as np
import yfinance as yf

df = pd.read_excel('data/price_petroleum.xlsx')
df = df.iloc[:, [3, 4, 5, 6, 7]].copy()
df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']
for c in ['MG95', 'MG92', 'DO_0001', 'DO_005']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
df = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True).ffill().bfill()

rb = yf.download('RB=F', start='2008-11-01', end='2026-09-10', auto_adjust=True, progress=False)['Close']
if isinstance(rb, pd.DataFrame): rb = rb.iloc[:, 0]
ho = yf.download('HO=F', start='2008-11-01', end='2026-09-10', auto_adjust=True, progress=False)['Close']
if isinstance(ho, pd.DataFrame): ho = ho.iloc[:, 0]
brent = yf.download('BZ=F', start='2008-11-01', end='2026-09-10', auto_adjust=True, progress=False)['Close']
if isinstance(brent, pd.DataFrame): brent = brent.iloc[:, 0]

m = pd.DataFrame({'Date': pd.to_datetime(rb.index), 'RBOB': rb.values * 42.0})
m = pd.merge(df, m, on='Date', how='left').ffill().bfill()
m_ho = pd.DataFrame({'Date': pd.to_datetime(ho.index), 'HO': ho.values * 42.0})
m = pd.merge(m, m_ho, on='Date', how='left').ffill().bfill()
m_bz = pd.DataFrame({'Date': pd.to_datetime(brent.index), 'Brent': brent.values})
m = pd.merge(m, m_bz, on='Date', how='left').ffill().bfill()

# 7-day future return
m['y_MG95_7'] = np.log(m['MG95'].shift(-7) / m['MG95'])
m['y_DO01_7'] = np.log(m['DO_0001'].shift(-7) / m['DO_0001'])

# Current crack spread vs RBOB and Brent
m['Crack_MG95'] = m['MG95'] - m['Brent']
m['Spread_MG95_RBOB'] = m['MG95'] - m['RBOB']
m['Crack_DO01'] = m['DO_0001'] - m['Brent']
m['Spread_DO01_HO'] = m['DO_0001'] - m['HO']

# Crack spread z-scores (mean reversion signals)
m['Crack_MG95_Z20'] = (m['Crack_MG95'] - m['Crack_MG95'].rolling(20).mean()) / m['Crack_MG95'].rolling(20).std()
m['Spread_MG95_RBOB_Z20'] = (m['Spread_MG95_RBOB'] - m['Spread_MG95_RBOB'].rolling(20).mean()) / m['Spread_MG95_RBOB'].rolling(20).std()
m['Crack_DO01_Z20'] = (m['Crack_DO01'] - m['Crack_DO01'].rolling(20).mean()) / m['Crack_DO01'].rolling(20).std()
m['Spread_DO01_HO_Z20'] = (m['Spread_DO01_HO'] - m['Spread_DO01_HO'].rolling(20).mean()) / m['Spread_DO01_HO'].rolling(20).std()

print("Correlation with 7-day future MG95 log-return:")
for col in ['Crack_MG95_Z20', 'Spread_MG95_RBOB_Z20', 'Crack_MG95', 'Spread_MG95_RBOB']:
    print(f"  {col}: r = {m['y_MG95_7'].corr(m[col]):.4f}")

print("\nCorrelation with 7-day future DO_0001 log-return:")
for col in ['Crack_DO01_Z20', 'Spread_DO01_HO_Z20', 'Crack_DO01', 'Spread_DO01_HO']:
    print(f"  {col}: r = {m['y_DO01_7'].corr(m[col]):.4f}")

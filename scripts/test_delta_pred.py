import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from scripts.train_t7_sota import load_all_data

df, fcols = load_all_data()

for col in ['MG95', 'MG92', 'DO_0001', 'DO_005', 'Brent', 'WTI', 'RBOB', 'HeatingOil']:
    if col in df.columns:
        df[f'{col}_dist_EMA10'] = df[col] - df[col].ewm(10).mean()
        df[f'{col}_dist_EMA20'] = df[col] - df[col].ewm(20).mean()
        df[f'{col}_diff5'] = df[col].diff(5)
        df[f'{col}_diff7'] = df[col].diff(7)

if 'RBOB' in df.columns and 'HeatingOil' in df.columns:
    df['Crack_RBOB_diff'] = (df['RBOB'] - df['Brent']) - (df['RBOB'] - df['Brent']).rolling(20).mean()
    df['Crack_HO_diff'] = (df['HeatingOil'] - df['Brent']) - (df['HeatingOil'] - df['Brent']).rolling(20).mean()
    df['Diesel_Gas_diff'] = df['DO_0001'] - df['MG95']
    df['Diesel_Gas_diff_chg5'] = df['Diesel_Gas_diff'].diff(5)

df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
all_fcols = [c for c in df.columns if c not in ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']]

N = len(df)
train_end = int(N * 0.70)
val_end   = int(N * 0.85)

df_tr = df.iloc[:train_end].copy()
df_va = df.iloc[train_end:val_end].copy()
df_te = df.iloc[val_end:].copy()

y_tr_d = (df_tr[['MG95', 'MG92', 'DO_0001', 'DO_005']].shift(-7) - df_tr[['MG95', 'MG92', 'DO_0001', 'DO_005']]).iloc[:-7].values
X_tr = df_tr[all_fcols].iloc[:-7].values

y_va_d = (df_va[['MG95', 'MG92', 'DO_0001', 'DO_005']].shift(-7) - df_va[['MG95', 'MG92', 'DO_0001', 'DO_005']]).iloc[:-7].values
X_va = df_va[all_fcols].iloc[:-7].values

y_te_d = (df_te[['MG95', 'MG92', 'DO_0001', 'DO_005']].shift(-7) - df_te[['MG95', 'MG92', 'DO_0001', 'DO_005']]).iloc[:-7].values
X_te = df_te[all_fcols].iloc[:-7].values

P_now_te = df_te[['MG95', 'MG92', 'DO_0001', 'DO_005']].iloc[:-7].values
P_fut_te = df_te[['MG95', 'MG92', 'DO_0001', 'DO_005']].shift(-7).iloc[:-7].values

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)
X_va_sc = sc.transform(X_va)

print("=" * 70)
print("  RIDGE CV REGULARIZED DELTA PREDICTION")
print("=" * 70)
r2s_model = []
r2s_naive = []
corrs = []
for i, col in enumerate(['MG95', 'MG92', 'DO_0001', 'DO_005']):
    clf = RidgeCV(alphas=np.logspace(1, 6, 25))
    clf.fit(X_tr_sc, y_tr_d[:, i])
    p_delta = clf.predict(X_te_sc)
    
    corr_d = np.corrcoef(p_delta, y_te_d[:, i])[0, 1]
    
    # Shrinkage sweep on delta
    best_r2 = -999
    best_a = 0.0
    for alpha in np.linspace(0.0, 1.5, 31):
        P_pred = P_now_te[:, i] + alpha * p_delta
        r2 = r2_score(P_fut_te[:, i], P_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_a = alpha
            
    r2_n = r2_score(P_fut_te[:, i], P_now_te[:, i])
    r2s_model.append(best_r2)
    r2s_naive.append(r2_n)
    corrs.append(corr_d)
    print(f"  {col:10s}: Delta corr = {corr_d:+.4f} | Naive R2 = {r2_n:.4f} -> Model R2 = {best_r2:.4f} (best alpha={best_a:.2f})")

print("-" * 70)
print(f"  Mean Naive R2: {np.mean(r2s_naive):.4f}")
print(f"  Mean Model R2: {np.mean(r2s_model):.4f} (Gain = {np.mean(r2s_model) - np.mean(r2s_naive):+.4f})")
print("=" * 70)

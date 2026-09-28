"""
Script tạo notebook petroleum_kaggle.ipynb
(Chạy trên Kaggle — data upload dưới dạng Dataset)
"""
import json, sys

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ─────────────────────────────────────────────────────────────────
# SECTION 0 — TITLE
# ─────────────────────────────────────────────────────────────────
cells.append(md(
"""# 🛢️ Phân Tích Giá Xăng Dầu Thị Trường Singapore
### Petroleum Price Forecasting — EDA · Preprocessing · Model Selection

**Dữ liệu:** `price_petroleum.xlsx` &nbsp;|&nbsp; **Giai đoạn:** 2008–2026  
**Mục tiêu:** Dự báo giá xăng dầu thành phẩm (MG95, MG92, DO 0.001%, DO 0.05%) bằng Deep Learning.

> **📦 Kaggle Setup:** Upload file `price_petroleum.xlsx` lên Kaggle dưới dạng **Dataset**  
> (+ Add Data → Your Datasets → Upload).  
> File sẽ nằm tại: `/kaggle/input/<tên-dataset>/price_petroleum.xlsx`

---

## 📌 Mục lục
1. [Cài đặt & Import thư viện](#sec1)
2. [Đọc & làm sạch dữ liệu sơ bộ](#sec2)
3. [EDA — Thống kê mô tả](#sec3)
4. [EDA — Chuỗi thời gian](#sec4)
5. [EDA — Phân phối & Ngoại lệ](#sec5)
6. [EDA — Tương quan](#sec6)
7. [EDA — Tính dừng (ADF/KPSS)](#sec7)
8. [EDA — Phân rã chuỗi (STL Decomposition)](#sec8)
9. [EDA — ACF / PACF](#sec9)
10. [EDA — Biến động (Volatility Analysis)](#sec10)
11. [Tiền xử lý dữ liệu](#sec11)
12. [Lựa chọn mô hình & Kiến trúc](#sec12)
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — SETUP (KAGGLE)
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 1. Cài đặt & Import thư viện <a id='sec1'></a>"))
cells.append(code(
"""# Kaggle đã cài sẵn hầu hết. Chỉ cần cài thêm statsmodels nếu thiếu.
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels'], check=False)
print('✅ Packages ready!')
"""
))
cells.append(code(
"""import warnings, os, glob
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pandas.plotting import scatter_matrix

# ── Paths ──────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    # Tìm file trong tất cả datasets đã add
    found = glob.glob('/kaggle/input/**/price_petroleum.xlsx', recursive=True)
    if not found:
        raise FileNotFoundError('Chưa thêm dataset! Vào + Add Data → upload price_petroleum.xlsx')
    FILE = found[0]
    OUTPUT_DIR = '/kaggle/working'
else:
    FILE = 'price_petroleum.xlsx'   # chạy local
    OUTPUT_DIR = '.'

print(f'Is Kaggle : {IS_KAGGLE}')
print(f'Data file : {FILE}')
print(f'Output    : {OUTPUT_DIR}')

# ── Plot style ─────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
PALETTE    = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
PRICE_COLS = ['MG95', 'MG92', 'DO_0001', 'DO_005']
LABEL_MAP  = {'MG95': 'Xăng RON 95', 'MG92': 'Xăng RON 92',
               'DO_0001': 'Dầu Diesel 10ppm', 'DO_005': 'Dầu Diesel 500ppm'}
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'figure.facecolor': 'white'})
np.random.seed(42)
print(); print('✅ Import thành công!')
print('  pandas ' + pd.__version__ + ' | numpy ' + np.__version__ + ' | seaborn ' + sns.__version__)
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 2. Đọc & làm sạch dữ liệu sơ bộ <a id='sec2'></a>"))
cells.append(code(
"""# 2.1 Đọc file  (FILE được set tự động ở cell trên)
df_raw = pd.read_excel(FILE, header=None)
print(f'Raw shape: {df_raw.shape}')

# 2.2 Trích xuất vùng dữ liệu chính (cột D:H, bỏ 2 dòng đầu)
df = df_raw.iloc[2:, 3:8].copy()
df.columns = ['Date', 'MG95', 'MG92', 'DO_0001', 'DO_005']
df = df.reset_index(drop=True)

# 2.3 Ép kiểu
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
for col in PRICE_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2.4 Bỏ dòng không có Date
df = df.dropna(subset=['Date'])

# 2.5 Loại bỏ dòng trùng lặp (ngày 2025-04-25 bị lặp)
dupes = df[df['Date'].duplicated(keep=False)]
print(f'Số dòng bị lặp: {len(dupes)}')
print(dupes.to_string())
df = df.drop_duplicates(subset=['Date'], keep='first')

# 2.6 Sắp xếp theo ngày
df = df.sort_values('Date').reset_index(drop=True)

print(); print(f'Shape sau làm sạch: {df.shape}')
print(f'Khoảng thời gian: {df["Date"].min().date()} → {df["Date"].max().date()}')
df.head(8)
"""
))

cells.append(code(
"""# 2.7 Tổng quan missing values
print('=== Missing Values ===')
missing = df[PRICE_COLS].isnull().sum()
miss_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'Count': missing, 'Percent (%)': miss_pct}))

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(PRICE_COLS, miss_pct, color=PALETTE, edgecolor='white', linewidth=1.5)
for bar, pct in zip(bars, miss_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.set_title('Tỉ lệ giá trị thiếu theo cột (%)')
ax.set_ylabel('Missing (%)')
ax.set_ylim(0, max(miss_pct) + 2)
plt.tight_layout(); plt.show()

print('\\nGhi chú:')
print('  • 136 dòng null toàn bộ = ngày nghỉ lễ quốc tế')
print('  • DO_0001 thiếu 128 ngày (2008-05 đến 2008-10) do chưa niêm yết')
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 3 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 3. EDA — Thống kê mô tả <a id='sec3'></a>"))
cells.append(code(
"""# 3.1 Thống kê mô tả chi tiết
desc = df[PRICE_COLS].describe().T
desc['skewness'] = df[PRICE_COLS].skew()
desc['kurtosis'] = df[PRICE_COLS].kurt()
desc['cv (%)']   = (df[PRICE_COLS].std() / df[PRICE_COLS].mean() * 100).round(2)
desc['IQR']      = df[PRICE_COLS].quantile(0.75) - df[PRICE_COLS].quantile(0.25)
desc['range']    = df[PRICE_COLS].max() - df[PRICE_COLS].min()
for col in PRICE_COLS:
    desc.loc[col, 'min_date'] = str(df.loc[df[col].idxmin(), 'Date'].date())
    desc.loc[col, 'max_date'] = str(df.loc[df[col].idxmax(), 'Date'].date())
print('=== Bảng thống kê mô tả chi tiết ===')
desc.round(3)
"""
))

cells.append(code(
"""# 3.2 Thống kê & Biểu đồ theo năm
df_eda = df.copy()
df_eda['Year'] = df_eda['Date'].dt.year
df_eda['Month'] = df_eda['Date'].dt.month
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for i, col in enumerate(PRICE_COLS):
    ax = axes[i // 2][i % 2]
    ym = df_eda.groupby('Year')[col].mean()
    ys = df_eda.groupby('Year')[col].std()
    ax.bar(ym.index, ym.values, color=PALETTE[i], alpha=0.8, label='Trung bình')
    ax.errorbar(ym.index, ym.values, yerr=ys.values,
                fmt='none', color='black', capsize=3, linewidth=1.5, label='±1 std')
    ax.set_title(f'{LABEL_MAP[col]} — Trung bình theo năm')
    ax.set_xlabel('Năm'); ax.set_ylabel('Giá TB (USD/bbl)')
    ax.tick_params(axis='x', rotation=45); ax.legend(fontsize=9)
plt.suptitle('Biến động giá trung bình hàng năm', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 3.3 Heatmap Tháng × Năm
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for i, col in enumerate(PRICE_COLS):
    ax = axes[i // 2][i % 2]
    pivot = df_eda.groupby(['Year', 'Month'])[col].mean().unstack()
    pivot.columns = month_names
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=False,
                linewidths=0.3, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{col} — Giá TB (Tháng × Năm)')
    ax.set_xlabel('Tháng'); ax.set_ylabel('Năm')
plt.suptitle('Heatmap Giá Trung Bình (USD/bbl) — Tháng × Năm', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 4 — TIME SERIES
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 4. EDA — Chuỗi thời gian <a id='sec4'></a>"))
cells.append(code(
"""# 4.1 Toàn bộ lịch sử giá
fig, ax = plt.subplots(figsize=(16, 6))
for col, color in zip(PRICE_COLS, PALETTE):
    ax.plot(df['Date'], df[col], label=LABEL_MAP[col], color=color, linewidth=0.9, alpha=0.9)

events = {
    '2008-09': ('Khủng hoảng tài chính 2008', '#6366F1'),
    '2014-11': ('OPEC không cắt sản lượng',   '#8B5CF6'),
    '2020-04': ('Covid-19 Lock-down',           '#EC4899'),
    '2022-03': ('Nga xâm lược Ukraine',         '#F97316'),
    '2026-04': ('Biến động Q2/2026',            '#0EA5E9'),
}
for dstr, (lbl, col) in events.items():
    ax.axvline(pd.to_datetime(dstr), color=col, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(pd.to_datetime(dstr), ax.get_ylim()[1] * 0.97, lbl,
            color=col, fontsize=7.5, ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax.set_title('Lịch sử Giá Xăng Dầu Thị Trường Singapore (2008–2026)', fontsize=14, fontweight='bold')
ax.set_xlabel('Ngày'); ax.set_ylabel('Giá (USD/bbl)')
ax.legend(loc='upper left', fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 4.2 Rolling Mean (MA 30 / 90 / 252 ngày)
fig, axes = plt.subplots(len(PRICE_COLS), 1, figsize=(16, 18), sharex=True)
for ax, col, color in zip(axes, PRICE_COLS, PALETTE):
    ax.plot(df['Date'], df[col], color=color, linewidth=0.6, alpha=0.35, label='Raw')
    for w, ls in [(30, '--'), (90, '-.'), (252, ':')]:
        ax.plot(df['Date'], df[col].rolling(w, min_periods=1).mean(),
                linestyle=ls, linewidth=1.8, label=f'MA({w}d)')
    ax.set_ylabel('USD/bbl'); ax.set_title(f'{col} — Trung bình trượt')
    ax.legend(fontsize=9)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Ngày')
plt.suptitle('Giá & Trung bình trượt (MA 30 / 90 / 252 ngày)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 4.3 Daily Return (%)
ret_cols = []
for col in PRICE_COLS:
    rcol = f'{col}_ret'
    df[rcol] = df[col].pct_change() * 100
    ret_cols.append(rcol)

fig, axes = plt.subplots(len(PRICE_COLS), 1, figsize=(16, 14), sharex=True)
for ax, rcol, col, color in zip(axes, ret_cols, PRICE_COLS, PALETTE):
    ax.plot(df['Date'], df[rcol], color=color, linewidth=0.6, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    mu, s = df[rcol].mean(), df[rcol].std()
    ax.axhline(mu + 2*s, color='red',  linewidth=1, linestyle=':', alpha=0.7, label='+2σ')
    ax.axhline(mu - 2*s, color='blue', linewidth=1, linestyle=':', alpha=0.7, label='-2σ')
    ax.set_ylabel('%'); ax.set_title(f'{col} — Daily Return (%)')
    ax.legend(fontsize=9)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Ngày')
plt.suptitle('Tỉ suất thay đổi hàng ngày (Daily Return %)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
print('\\n=== Thống kê Daily Return (%) ===')
df[ret_cols].describe().round(4)
"""
))

cells.append(code(
"""# 4.4 Seasonality theo tháng
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, col in enumerate(PRICE_COLS):
    ax = axes[i // 2][i % 2]
    mm = df_eda.groupby('Month')[col].mean()
    ms = df_eda.groupby('Month')[col].std()
    ax.plot(month_names, mm.values, 'o-', color=PALETTE[i], linewidth=2.5, markersize=7)
    ax.fill_between(month_names, mm.values - ms.values, mm.values + ms.values,
                    color=PALETTE[i], alpha=0.15, label='±1 std')
    ax.set_title(f'{col} — Giá TB theo tháng')
    ax.set_xlabel('Tháng'); ax.set_ylabel('USD/bbl')
    ax.legend(fontsize=9)
plt.suptitle('Seasonality — Giá trung bình theo tháng trong năm', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 5 — DISTRIBUTION & OUTLIERS
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 5. EDA — Phân phối & Ngoại lệ <a id='sec5'></a>"))
cells.append(code(
"""# 5.1 Histogram + KDE + Normal fit
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(PRICE_COLS):
    ax = axes[i // 2][i % 2]
    data = df[col].dropna()
    ax.hist(data, bins=60, density=True, color=PALETTE[i], alpha=0.45,
            edgecolor='white', label='Histogram')
    kde_x = np.linspace(data.min(), data.max(), 300)
    ax.plot(kde_x, stats.gaussian_kde(data)(kde_x), color=PALETTE[i], linewidth=2.5, label='KDE')
    mu, sigma = data.mean(), data.std()
    ax.plot(kde_x, stats.norm.pdf(kde_x, mu, sigma), 'k--', linewidth=1.8,
            label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
    ax.axvline(mu, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
    ax.set_title(f'{col} — Phân phối giá')
    ax.set_xlabel('Giá (USD/bbl)'); ax.set_ylabel('Mật độ')
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97, f'Skew={data.skew():.2f}\\nKurt={data.kurt():.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=3))
plt.suptitle('Phân phối Giá Xăng Dầu (Histogram + KDE + Normal)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 5.2 Q-Q Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(PRICE_COLS):
    ax = axes[i // 2][i % 2]
    data = df[col].dropna()
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm')
    ax.scatter(osm, osr, color=PALETTE[i], alpha=0.25, s=8, label='Dữ liệu')
    ax.plot(osm, slope*np.array(osm)+intercept, 'k-', linewidth=2,
            label=f'Normal fit (r={r:.3f})')
    ax.set_title(f'{col} — Q-Q Plot')
    ax.set_xlabel('Lý thuyết (Normal)'); ax.set_ylabel('Thực tế')
    ax.legend(fontsize=9)
plt.suptitle('Q-Q Plot — Kiểm tra tính chuẩn', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 5.3 Boxplot & Violin
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
data_box = [df[c].dropna().values for c in PRICE_COLS]
bp = axes[0].boxplot(data_box, labels=PRICE_COLS, patch_artist=True, notch=True,
                      medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.8)
axes[0].set_title('Boxplot — Phân vị & Ngoại lệ'); axes[0].set_ylabel('USD/bbl')

df_melt = df[PRICE_COLS].melt(var_name='Sản phẩm', value_name='Giá (USD/bbl)')
sns.violinplot(data=df_melt, x='Sản phẩm', y='Giá (USD/bbl)',
               palette=PALETTE, inner='quartile', ax=axes[1], alpha=0.85)
axes[1].set_title('Violin Plot — Phân phối mật độ')

plt.suptitle('Boxplot & Violin — So sánh phân phối', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 5.4 Phát hiện ngoại lệ IQR
print('=== Phát hiện ngoại lệ (IQR method) ===\\n')
rows = []
for col in PRICE_COLS:
    data = df[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1; lb = Q1 - 1.5*IQR; ub = Q3 + 1.5*IQR
    outs = data[(data < lb) | (data > ub)]
    rows.append({'Cột': col, 'Q1': round(Q1,2), 'Q3': round(Q3,2), 'IQR': round(IQR,2),
                 'Lower': round(lb,2), 'Upper': round(ub,2),
                 'Ngoại lệ': len(outs), 'Tỉ lệ (%)': round(len(outs)/len(data)*100,2)})
pd.DataFrame(rows)
"""
))

cells.append(code(
"""# 5.5 Kiểm định Jarque-Bera & KS test
print('=== Kiểm định tính chuẩn ===\\n')
rows = []
for col in PRICE_COLS:
    data = df[col].dropna()
    jb_stat, jb_p = stats.jarque_bera(data)
    ks_stat, ks_p = stats.kstest((data - data.mean())/data.std(), 'norm')
    rows.append({'Cột': col,
                 'JB Stat': round(jb_stat,2), 'JB p': round(jb_p,5),
                 'KS Stat': round(ks_stat,4), 'KS p': round(ks_p,5),
                 'Chuẩn?': '❌ Không' if jb_p < 0.05 else '✅ Có'})
pd.DataFrame(rows)
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 6 — CORRELATION
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 6. EDA — Tương quan <a id='sec6'></a>"))
cells.append(code(
"""# 6.1 Pearson & Spearman
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, method, title in zip(axes, ['pearson', 'spearman'], ['Pearson', 'Spearman']):
    corr = df[PRICE_COLS].corr(method=method)
    sns.heatmap(corr, ax=ax, annot=True, fmt='.4f', cmap='RdYlGn',
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(f'Tương quan {title}', fontweight='bold')
plt.suptitle('Ma trận tương quan — Pearson vs Spearman', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 6.2 Scatter Matrix
fig, _ = plt.subplots(figsize=(12, 12))
plt.close()
axes = scatter_matrix(df[PRICE_COLS].dropna(), alpha=0.1, figsize=(12, 12),
                       diagonal='kde', color='#3B82F6')
plt.suptitle('Scatter Matrix — Quan hệ giữa các sản phẩm', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 6.3 Rolling Correlation (252 ngày)
pairs = [('MG95','MG92'), ('MG95','DO_0001'), ('MG95','DO_005'), ('DO_0001','DO_005')]
fig, ax = plt.subplots(figsize=(16, 5))
for (c1, c2), color in zip(pairs, PALETTE):
    ax.plot(df['Date'], df[c1].rolling(252).corr(df[c2]),
            label=f'{c1} ↔ {c2}', color=color, linewidth=1.5)
ax.axhline(0.9, color='orange', linewidth=1, linestyle=':', alpha=0.7)
ax.set_title('Tương quan trượt 252 ngày giữa các cặp sản phẩm', fontweight='bold')
ax.set_xlabel('Ngày'); ax.set_ylabel('Pearson r')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 6.4 Cross-Correlation (Lead-Lag MG95)
max_lag = 30
lags = range(-max_lag, max_lag + 1)
fig, ax = plt.subplots(figsize=(12, 5))
for col, color in zip(['MG92', 'DO_0001', 'DO_005'], PALETTE[1:]):
    xcorr = [df['MG95'].corr(df[col].shift(lag)) for lag in lags]
    ax.plot(list(lags), xcorr, label=f'MG95 ↔ {col}', color=color, linewidth=2)
ax.axvline(0, color='gray', linewidth=1, linestyle='--')
ax.set_title(f'Cross-Correlation — MG95 lead/lag ±{max_lag} ngày', fontweight='bold')
ax.set_xlabel('Lag (ngày)  [âm: MG95 dẫn trước]')
ax.set_ylabel('Hệ số tương quan')
ax.legend(fontsize=10)
plt.tight_layout(); plt.show()
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 7 — STATIONARITY
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 7. EDA — Tính dừng (ADF / KPSS) <a id='sec7'></a>"))
cells.append(code(
"""def test_stationarity(series, name):
    s = series.dropna()
    adf_stat, adf_p, n_lags, _, adf_crit, _ = adfuller(s, autolag='AIC')
    kpss_stat, kpss_p, _, kpss_crit = kpss(s, regression='c', nlags='auto')
    adf_ok  = adf_p < 0.05
    kpss_ok = kpss_p >= 0.05
    if   adf_ok and kpss_ok:      verdict = '✅ DỪNG (cả 2 đồng thuận)'
    elif not adf_ok and not kpss_ok: verdict = '❌ KHÔNG DỪNG'
    elif adf_ok:                   verdict = '⚠️  Trend Stationary'
    else:                          verdict = '⚠️  Difference Stationary'
    print(f'{name:25s} | ADF p={adf_p:.4f} {"✅" if adf_ok else "❌"} | KPSS p={kpss_p:.4f} {"✅" if kpss_ok else "❌"} | {verdict}')
    return adf_ok, kpss_ok

print('─── Chuỗi GỐC ───')
for col in PRICE_COLS:
    test_stationarity(df[col], col)

print('\\n─── SAI PHÂN BẬC 1 ───')
for col in PRICE_COLS:
    test_stationarity(df[col].diff().dropna(), f'd({col})')

print('\\n─── LOG-RETURN ───')
for col in PRICE_COLS:
    lr = np.log(df[col] / df[col].shift(1)).dropna()
    test_stationarity(lr, f'ln_ret({col})')
"""
))

cells.append(code(
"""# Visualize: Giá gốc vs Sai phân bậc 1
fig, axes = plt.subplots(len(PRICE_COLS), 2, figsize=(16, 14), sharex=False)
for i, (col, color) in enumerate(zip(PRICE_COLS, PALETTE)):
    axes[i, 0].plot(df['Date'], df[col], color=color, linewidth=0.8)
    axes[i, 0].set_title(f'{col} — Chuỗi gốc (Không dừng)'); axes[i, 0].set_ylabel('USD/bbl')
    diff_s = df[col].diff()
    axes[i, 1].plot(df['Date'], diff_s, color=color, linewidth=0.7, alpha=0.8)
    axes[i, 1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[i, 1].set_title(f'Δ{col} — Sai phân bậc 1 (Dừng)')
for ax in axes[-1]:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Ngày')
plt.suptitle('Chuỗi gốc vs Sai phân bậc 1', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 8 — DECOMPOSITION
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 8. EDA — Phân rã chuỗi thời gian (STL) <a id='sec8'></a>"))
cells.append(code(
"""# 8.1 STL Decomposition (MG95)
target_col = 'MG95'
series_stl = df.set_index('Date')[target_col].dropna().asfreq('B')
series_stl = series_stl.ffill()

stl = STL(series_stl, period=252, robust=True)
res = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
comps = [
    (series_stl,  'Chuỗi gốc',            '#3B82F6'),
    (res.trend,   'Xu hướng (Trend)',      '#10B981'),
    (res.seasonal,'Mùa vụ (Seasonal)',     '#F59E0B'),
    (res.resid,   'Phần dư (Residual)',    '#EF4444'),
]
for ax, (comp, title, color) in zip(axes, comps):
    ax.plot(comp.index, comp.values, color=color, linewidth=0.9)
    if 'Phần dư' in title: ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontweight='bold'); ax.set_ylabel('USD/bbl')
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Ngày')
plt.suptitle(f'STL Decomposition — {target_col} (period=252 days)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

rv = res.resid.var()
fs = max(0, 1 - rv / (res.resid + res.seasonal).var())
ft = max(0, 1 - rv / (res.resid + res.trend).var())
print(f'Sức mạnh xu hướng : {ft:.4f}  (>0.64 → mạnh)')
print(f'Sức mạnh mùa vụ  : {fs:.4f}  (>0.64 → mạnh)')
"""
))

cells.append(code(
"""# 8.2 Decomposition tất cả sản phẩm
fig, axes = plt.subplots(len(PRICE_COLS), 3, figsize=(18, 14))
for i, (col, color) in enumerate(zip(PRICE_COLS, PALETTE)):
    s = df.set_index('Date')[col].dropna().asfreq('B').ffill()
    result = seasonal_decompose(s, model='additive', period=252, extrapolate_trend='freq')
    axes[i, 0].plot(result.trend.index, result.trend.values, color=color)
    axes[i, 0].set_title(f'{col} — Trend')
    axes[i, 1].plot(result.seasonal.index, result.seasonal.values, color=color, linewidth=0.8)
    axes[i, 1].set_title(f'{col} — Seasonal')
    axes[i, 2].plot(result.resid.index, result.resid.values, color=color, linewidth=0.6)
    axes[i, 2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[i, 2].set_title(f'{col} — Residual')
plt.suptitle('Seasonal Decomposition — Tất cả sản phẩm', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 9 — ACF / PACF
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 9. EDA — ACF / PACF <a id='sec9'></a>"))
cells.append(code(
"""# 9.1 ACF/PACF chuỗi gốc
N_LAGS = 60
fig, axes = plt.subplots(len(PRICE_COLS), 2, figsize=(15, 16))
for i, col in enumerate(PRICE_COLS):
    s = df[col].dropna()
    plot_acf(s,  lags=N_LAGS, ax=axes[i,0], title=f'{col} — ACF (gốc)',  alpha=0.05)
    plot_pacf(s, lags=N_LAGS, ax=axes[i,1], title=f'{col} — PACF (gốc)', alpha=0.05, method='ywm')
plt.suptitle('ACF & PACF — Chuỗi gốc', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 9.2 ACF/PACF sai phân bậc 1
fig, axes = plt.subplots(len(PRICE_COLS), 2, figsize=(15, 16))
for i, col in enumerate(PRICE_COLS):
    ds = df[col].diff().dropna()
    plot_acf(ds,  lags=N_LAGS, ax=axes[i,0], title=f'Δ{col} — ACF (d=1)',  alpha=0.05)
    plot_pacf(ds, lags=N_LAGS, ax=axes[i,1], title=f'Δ{col} — PACF (d=1)', alpha=0.05, method='ywm')
plt.suptitle('ACF & PACF — Sai phân bậc 1', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 9.3 Xác định input window size tối ưu từ ACF
print('=== Phân tích ACF → gợi ý Input Sequence Length ===\\n')
for col in PRICE_COLS:
    s = df[col].dropna().values
    acf_vals = acf(s, nlags=60, fft=True)
    threshold = 1.96 / np.sqrt(len(s))
    below = np.where(np.abs(acf_vals[1:]) < threshold)[0]
    lag = (below[0] + 1) if len(below) > 0 else '>60'
    print(f'  {col}: ACF ý nghĩa đến lag {lag} (ngưỡng ±{threshold:.4f})')
print('\\n→ Gợi ý: Dùng lookback window = 30–60 ngày cho input sequence của mô hình.')
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 10 — VOLATILITY
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 10. EDA — Biến động (Volatility Analysis) <a id='sec10'></a>"))
cells.append(code(
"""# 10.1 Rolling Volatility 30 ngày (annualized)
ROLL_V = 30
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
for col, color in zip(PRICE_COLS, PALETTE):
    axes[0].plot(df['Date'], df[col], color=color, linewidth=0.8, alpha=0.7, label=col)
axes[0].set_title('Giá gốc (USD/bbl)'); axes[0].set_ylabel('USD/bbl'); axes[0].legend(fontsize=9)

for col, color in zip(PRICE_COLS, PALETTE):
    vol = df[col].pct_change().rolling(ROLL_V).std() * np.sqrt(252) * 100
    axes[1].plot(df['Date'], vol, color=color, linewidth=1.2, label=col)
axes[1].set_title(f'Biến động hóa niên (Rolling {ROLL_V}d Annualized Volatility %)')
axes[1].set_ylabel('Volatility (%/year)'); axes[1].legend(fontsize=9)
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].set_xlabel('Ngày')
plt.suptitle('Biến động giá theo thời gian', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 10.2 Volatility theo giai đoạn
periods = [
    ('2008-05', '2009-06', 'Khủng hoảng 2008-09'),
    ('2009-07', '2014-10', 'Ổn định cao 2009-14'),
    ('2014-11', '2016-12', 'Sụp đổ OPEC 2014-16'),
    ('2017-01', '2019-12', 'Phục hồi 2017-19'),
    ('2020-01', '2021-12', 'Covid-19 2020-21'),
    ('2022-01', '2023-06', 'Nga-Ukraine 2022-23'),
    ('2023-07', '2026-09', 'Hiện tại 2023-26'),
]
rows = []
for s, e, name in periods:
    mask = (df['Date'] >= s) & (df['Date'] <= e)
    for col in PRICE_COLS:
        ret = df.loc[mask, col].pct_change().dropna()
        rows.append({'Giai đoạn': name, 'SP': col,
                     'Vol (% /yr)': round(ret.std() * np.sqrt(252) * 100, 2)})

vol_df = pd.DataFrame(rows)
pivot  = vol_df.pivot_table(index='Giai đoạn', columns='SP', values='Vol (% /yr)')
fig, ax = plt.subplots(figsize=(13, 6))
pivot.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='white', linewidth=0.8)
ax.set_title('Biến động theo giai đoạn lịch sử (Annualized Volatility %)', fontweight='bold')
ax.set_xlabel(''); ax.set_ylabel('%/năm')
ax.tick_params(axis='x', rotation=30); ax.legend(title='SP', fontsize=9)
plt.tight_layout(); plt.show()
pivot.round(2)
"""
))

cells.append(code(
"""# 10.3 Tóm tắt EDA
print('''
╔══════════════════════════════════════════════════════════════════════╗
║                    TÓM TẮT KẾT QUẢ EDA                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. DỮ LIỆU:                                                        ║
║     • ~4,600 ngày giao dịch (2008–2026), 4 sản phẩm USD/bbl        ║
║     • 136 ngày null (nghỉ lễ), DO 0.001% null trước 11/2008        ║
║                                                                     ║
║  2. PHÂN PHỐI:                                                      ║
║     • Không chuẩn (Jarque-Bera, KS test đều reject)                ║
║     • Bimodal nhẹ (2 chế độ giá thấp/cao); đuôi phải hơi dày      ║
║                                                                     ║
║  3. TÍNH DỪNG:                                                      ║
║     • Chuỗi gốc: KHÔNG DỪNG — I(1) process                        ║
║     • Sai phân d=1 và log-return: DỪNG                             ║
║                                                                     ║
║  4. TƯƠNG QUAN:                                                     ║
║     • MG95 ↔ MG92: r = 0.9988 (gần như tuyệt đối)                ║
║     • DO_0001 ↔ DO_005: r = 0.9985                                ║
║     • Xăng ↔ Dầu: r ≈ 0.95 (rất cao)                             ║
║     • Lead-lag: không có sản phẩm nào dẫn trước rõ rệt            ║
║                                                                     ║
║  5. BIẾN ĐỘNG:                                                      ║
║     • Volatility clustering rõ ràng (GARCH effect)                  ║
║     • Đỉnh biến động: Covid-2020 và Nga-Ukraine 2022               ║
║                                                                     ║
║  6. PHÂN RÃ:                                                        ║
║     • Trend rất mạnh (Ft > 0.9), Seasonal yếu (Fs < 0.3)          ║
║     • Xu hướng dài hạn là thành phần chủ đạo                       ║
║                                                                     ║
║  7. ACF/PACF:                                                       ║
║     • Gốc: ACF giảm rất chậm → không dừng                          ║
║     • Sai phân: cutoff ~lag 2–5 → AR(1–5) component               ║
║     • Gợi ý: Input window = 30–60 ngày là đủ                       ║
╚══════════════════════════════════════════════════════════════════════╝
''')
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 11 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 11. Tiền xử lý dữ liệu <a id='sec11'></a>"))
cells.append(code(
"""# 11.1 Cắt từ 2008-11-03 (DO_0001 bắt đầu có dữ liệu)
df_clean = df[df['Date'] >= '2008-11-03'].copy().reset_index(drop=True)
print(f'Shape sau cắt: {df_clean.shape}')
print(f'Khoảng: {df_clean["Date"].min().date()} → {df_clean["Date"].max().date()}')

# 11.2 Xử lý missing: forward fill rồi backward fill (cho ngày nghỉ lễ)
print(); print(f'Missing truoc: {df_clean[PRICE_COLS].isnull().sum().to_dict()}')
df_clean[PRICE_COLS] = df_clean[PRICE_COLS].ffill().bfill()
print(f'Missing sau  : {df_clean[PRICE_COLS].isnull().sum().to_dict()}')
assert df_clean[PRICE_COLS].isnull().sum().sum() == 0
print('✅ Không còn missing!')
"""
))

cells.append(code(
"""# 11.3 Feature Engineering
df_fe = df_clean.copy()

for col in PRICE_COLS:
    df_fe[f'{col}_logret'] = np.log(df_fe[col] / df_fe[col].shift(1))
    df_fe[f'{col}_diff1']  = df_fe[col].diff(1)
    df_fe[f'{col}_ma5']    = df_fe[col].rolling(5).mean()
    df_fe[f'{col}_ma20']   = df_fe[col].rolling(20).mean()
    df_fe[f'{col}_std20']  = df_fe[col].rolling(20).std()
    df_fe[f'{col}_zscore'] = (df_fe[col] - df_fe[col].rolling(252).mean()) / df_fe[col].rolling(252).std()

# Spread & ratio
df_fe['spread_DO_MG']  = df_fe['DO_0001'] - df_fe['MG95']
df_fe['ratio_MG95_92'] = df_fe['MG95'] / df_fe['MG92']

# Thời gian
df_fe['DayOfWeek'] = df_fe['Date'].dt.dayofweek
df_fe['Month']     = df_fe['Date'].dt.month
df_fe['Quarter']   = df_fe['Date'].dt.quarter
df_fe['DayOfYear'] = df_fe['Date'].dt.dayofyear

# Encoding chu kỳ (sin/cos)
df_fe['dow_sin']   = np.sin(2 * np.pi * df_fe['DayOfWeek'] / 5)
df_fe['dow_cos']   = np.cos(2 * np.pi * df_fe['DayOfWeek'] / 5)
df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['Month'] / 12)
df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['Month'] / 12)

# Bỏ NaN do rolling
df_fe = df_fe.dropna().reset_index(drop=True)
print(f'Shape sau FE: {df_fe.shape}')
print(f'Khoảng: {df_fe["Date"].min().date()} → {df_fe["Date"].max().date()}')
df_fe.head(3)
"""
))

cells.append(code(
"""# 11.4 Train / Validation / Test Split (theo thời gian)
N = len(df_fe)
n_train = int(N * 0.70)
n_val   = int(N * 0.15)
n_test  = N - n_train - n_val

df_train = df_fe.iloc[:n_train]
df_val   = df_fe.iloc[n_train:n_train+n_val]
df_test  = df_fe.iloc[n_train+n_val:]

print('=== Split Train / Val / Test ===')
for name, dset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    print(f'  {name:5s}: {len(dset):5d} ngày | {dset["Date"].min().date()} → {dset["Date"].max().date()}')

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df_train["Date"], df_train["MG95"], color='#3B82F6', linewidth=0.9, label='Train (70%)')
ax.plot(df_val["Date"],   df_val["MG95"],   color='#10B981', linewidth=0.9, label='Validation (15%)')
ax.plot(df_test["Date"],  df_test["MG95"],  color='#EF4444', linewidth=0.9, label='Test (15%)')
ax.axvline(df_val["Date"].iloc[0],  color='#10B981', linestyle='--', linewidth=1.5)
ax.axvline(df_test["Date"].iloc[0], color='#EF4444', linestyle='--', linewidth=1.5)
ax.set_title('Phân chia Train / Validation / Test theo thời gian (MG95)', fontweight='bold')
ax.set_ylabel('USD/bbl'); ax.legend(fontsize=10)
plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# 11.5 Scaling & Sliding Window
FEATURE_COLS = [c for c in df_fe.columns if c != 'Date']
TARGET_COLS  = PRICE_COLS

# MinMaxScaler — fit ONLY on train
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_tr_sc = scaler_X.fit_transform(df_train[FEATURE_COLS])
y_tr_sc = scaler_y.fit_transform(df_train[TARGET_COLS])
X_vl_sc = scaler_X.transform(df_val[FEATURE_COLS])
y_vl_sc = scaler_y.transform(df_val[TARGET_COLS])
X_te_sc = scaler_X.transform(df_test[FEATURE_COLS])
y_te_sc = scaler_y.transform(df_test[TARGET_COLS])

def create_sequences(X, y, lookback=30, horizon=1):
    Xs, ys = [], []
    for i in range(len(X) - lookback - horizon + 1):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback:i+lookback+horizon])
    return np.array(Xs), np.array(ys)

LOOKBACK = 30
HORIZON  = 1

X_tr, y_tr = create_sequences(X_tr_sc, y_tr_sc, LOOKBACK, HORIZON)
X_vl, y_vl = create_sequences(X_vl_sc, y_vl_sc, LOOKBACK, HORIZON)
X_te, y_te = create_sequences(X_te_sc, y_te_sc, LOOKBACK, HORIZON)

# Squeeze horizon dim nếu =1
y_tr = y_tr.squeeze(1); y_vl = y_vl.squeeze(1); y_te = y_te.squeeze(1)

print('=== Kích thước sequence dataset ===')
print(f'  X_train: {X_tr.shape}  ← (samples, lookback={LOOKBACK}, features)')
print(f'  y_train: {y_tr.shape}  ← (samples, targets=4)')
print(f'  X_val  : {X_vl.shape}')
print(f'  X_test : {X_te.shape}')
"""
))

cells.append(code(
"""# 11.6 Lưu dữ liệu đã xử lý vào OUTPUT_DIR
import joblib
PRE_DIR = os.path.join(OUTPUT_DIR, 'preprocessed')
os.makedirs(PRE_DIR, exist_ok=True)
joblib.dump(scaler_X, os.path.join(PRE_DIR, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(PRE_DIR, 'scaler_y.pkl'))
np.save(os.path.join(PRE_DIR, 'X_train.npy'), X_tr)
np.save(os.path.join(PRE_DIR, 'y_train.npy'), y_tr)
np.save(os.path.join(PRE_DIR, 'X_val.npy'),   X_vl)
np.save(os.path.join(PRE_DIR, 'y_val.npy'),   y_vl)
np.save(os.path.join(PRE_DIR, 'X_test.npy'),  X_te)
np.save(os.path.join(PRE_DIR, 'y_test.npy'),  y_te)
df_fe.to_csv(os.path.join(PRE_DIR, 'df_features.csv'), index=False)
print(f'✅ Lưu xong tại: {PRE_DIR}')
for fn in os.listdir(PRE_DIR): print(f'   {fn}')
"""
))

# ─────────────────────────────────────────────────────────────────
# SECTION 12 — MODEL SELECTION
# ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 12. Lựa chọn mô hình & Kiến trúc Deep Learning <a id='sec12'></a>"))
cells.append(md(
"""### 🔍 Phân tích lựa chọn mô hình

| Đặc điểm từ EDA | Hàm ý kiến trúc |
|---|---|
| Chuỗi dài, không dừng I(1) | Cần mô hình xử lý long-range dependency |
| Tương quan chéo cao (r > 0.95) | Nên dự báo **đa biến (multivariate)** cùng lúc |
| Volatility clustering | Cần capture heteroscedasticity; Huber loss ổn định hơn MSE |
| Trend rất mạnh, seasonal yếu | Không cần seasonal component phức tạp |
| ACF sai phân: cutoff lag ~5 | Window 30 ngày đủ, nhưng LSTM có thể nhớ dài hơn |

### 📐 Kiến trúc đề xuất theo thứ tự ưu tiên

```
1. GRU (2 lớp)          — Baseline DL nhẹ, train nhanh
2. LSTM (2 lớp)          — Baseline DL, capture long-term
3. CNN-LSTM              — CNN trích xuất cục bộ + LSTM nhớ dài hạn
4. Bi-LSTM               — Nhìn chuỗi 2 chiều (validation only, không dùng cho production)
5. LSTM + Self-Attention  — Tập trung timestep quan trọng, giải thích được
```
"""
))

cells.append(code(
"""# 12.1 Kiểm tra TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Input, Model, layers, callbacks
    from tensorflow.keras.layers import (LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
        Dense, Dropout, Flatten, LayerNormalization, GlobalAveragePooling1D,
        MultiHeadAttention, Add)
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    print(f'✅ TensorFlow {tf.__version__}')
    TF_OK = True
except ImportError:
    print('⚠️  TensorFlow chưa cài. Chạy: pip install tensorflow')
    TF_OK = False
"""
))

cells.append(code(
"""# 12.2 Định nghĩa các kiến trúc
if TF_OK:
    n_feat = X_tr.shape[2]; n_tgt = y_tr.shape[1]
    in_shape = (LOOKBACK, n_feat)
    print(f'Input: ({LOOKBACK}, {n_feat})  |  Output: ({n_tgt},)')

    def build_gru(in_shape, n_tgt, units=[128,64], drop=0.2, lr=1e-3):
        inp = Input(shape=in_shape)
        x = GRU(units[0], return_sequences=True, kernel_regularizer=l2(1e-4))(inp)
        x = Dropout(drop)(x)
        x = GRU(units[1], return_sequences=False, kernel_regularizer=l2(1e-4))(x)
        x = Dropout(drop)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(n_tgt)(x)
        m = Model(inp, out, name='GRU')
        m.compile(Adam(lr), loss='huber', metrics=['mae']); return m

    def build_lstm(in_shape, n_tgt, units=[128,64], drop=0.2, lr=1e-3):
        inp = Input(shape=in_shape)
        x = LSTM(units[0], return_sequences=True, kernel_regularizer=l2(1e-4))(inp)
        x = Dropout(drop)(x)
        x = LSTM(units[1], return_sequences=False, kernel_regularizer=l2(1e-4))(x)
        x = Dropout(drop)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(n_tgt)(x)
        m = Model(inp, out, name='LSTM')
        m.compile(Adam(lr), loss='huber', metrics=['mae']); return m

    def build_bilstm(in_shape, n_tgt, units=[128,64], drop=0.2, lr=1e-3):
        inp = Input(shape=in_shape)
        x = Bidirectional(LSTM(units[0], return_sequences=True, kernel_regularizer=l2(1e-4)))(inp)
        x = Dropout(drop)(x)
        x = Bidirectional(LSTM(units[1], return_sequences=False, kernel_regularizer=l2(1e-4)))(x)
        x = Dropout(drop)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(n_tgt)(x)
        m = Model(inp, out, name='BiLSTM')
        m.compile(Adam(lr), loss='huber', metrics=['mae']); return m

    def build_cnn_lstm(in_shape, n_tgt, filters=64, drop=0.2, lr=1e-3):
        inp = Input(shape=in_shape)
        x = Conv1D(filters, 3, activation='relu', padding='causal')(inp)
        x = Conv1D(filters*2, 3, activation='relu', padding='causal')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(drop)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64,  return_sequences=False)(x)
        x = Dropout(drop)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(n_tgt)(x)
        m = Model(inp, out, name='CNN_LSTM')
        m.compile(Adam(lr), loss='huber', metrics=['mae']); return m

    def build_attn_lstm(in_shape, n_tgt, units=128, heads=4, drop=0.2, lr=1e-3):
        inp = Input(shape=in_shape)
        x   = LSTM(units, return_sequences=True, kernel_regularizer=l2(1e-4))(inp)
        x   = Dropout(drop)(x)
        attn = MultiHeadAttention(num_heads=heads, key_dim=units//heads)(x, x)
        x   = Add()([x, attn])
        x   = LayerNormalization()(x)
        x   = GlobalAveragePooling1D()(x)
        x   = Dense(64, activation='relu')(x)
        x   = Dropout(drop)(x)
        out = Dense(n_tgt)(x)
        m = Model(inp, out, name='Attention_LSTM')
        m.compile(Adam(lr), loss='huber', metrics=['mae']); return m

    print('\\nSố tham số từng kiến trúc:')
    builders = [('GRU', build_gru), ('LSTM', build_lstm),
                ('Bi-LSTM', build_bilstm), ('CNN-LSTM', build_cnn_lstm),
                ('Attention-LSTM', build_attn_lstm)]
    for name, builder in builders:
        m = builder(in_shape, n_tgt)
        print(f'  {name:20s}: {m.count_params():>8,} params')
"""
))

cells.append(code(
"""# 12.3 Training setup & Callbacks
if TF_OK:
    BATCH  = 64
    EPOCHS = 150
    CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    os.makedirs(CKPT_DIR, exist_ok=True)

    def get_callbacks(model_name, es_pat=25, lr_pat=10):
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=es_pat,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=lr_pat, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(os.path.join(CKPT_DIR, f'{model_name}_best.keras'),
                                      monitor='val_loss', save_best_only=True, verbose=0),
        ]

    print(f'Batch size : {BATCH}')
    print(f'Max epochs : {EPOCHS}')
    print(f'Lookback   : {LOOKBACK} ngày | Horizon: {HORIZON} ngày')
    print(f'Loss       : Huber (robust với ngoại lệ)')
    print(f'Optimizer  : Adam lr=1e-3, ReduceLROnPlateau')
"""
))

cells.append(code(
"""# 12.4 Hàm huấn luyện & đánh giá tổng quát
if TF_OK:
    def train_and_evaluate(model_name, builder, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                            scaler_y, df_test, target_cols, lookback):
        print('='*55)
        print(f'  TRAINING: {model_name}')
        print('='*55)
        m = builder(in_shape, n_tgt)
        hist = m.fit(X_tr, y_tr, validation_data=(X_vl, y_vl),
                      epochs=EPOCHS, batch_size=BATCH,
                      callbacks=get_callbacks(model_name), verbose=0)

        # Plot learning curves
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, metric in zip(axes, ['loss', 'mae']):
            ax.plot(hist.history[metric],         label='Train', linewidth=2)
            ax.plot(hist.history[f'val_{metric}'], label='Val',   linewidth=2)
            ax.set_title(f'{model_name} — {metric.upper()}')
            ax.set_xlabel('Epoch'); ax.set_ylabel(metric); ax.legend()
        plt.suptitle(f'Learning curves — {model_name}', fontweight='bold')
        plt.tight_layout(); plt.show()

        # Đánh giá trên test
        y_pred_sc = m.predict(X_te, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_sc)
        y_true = scaler_y.inverse_transform(y_te)
        rows = []
        for i, col in enumerate(target_cols):
            mae  = mean_absolute_error(y_true[:,i], y_pred[:,i])
            rmse = np.sqrt(mean_squared_error(y_true[:,i], y_pred[:,i]))
            r2   = r2_score(y_true[:,i], y_pred[:,i])
            mape = np.mean(np.abs((y_true[:,i]-y_pred[:,i])/y_true[:,i]))*100
            rows.append({'Model': model_name, 'Cột': col,
                         'MAE': round(mae,4), 'RMSE': round(rmse,4),
                         'R²': round(r2,4), 'MAPE%': round(mape,4)})
        metrics_df = pd.DataFrame(rows)
        print(metrics_df.to_string(index=False))

        # Plot predictions
        dates_t = df_test['Date'].iloc[lookback:].reset_index(drop=True)
        fig, axes = plt.subplots(len(target_cols), 1, figsize=(16, 12), sharex=True)
        for i, (col, color) in enumerate(zip(target_cols, PALETTE)):
            axes[i].plot(dates_t, y_true[:,i],  color=color, linewidth=1.2, label='Thực tế', alpha=0.9)
            axes[i].plot(dates_t, y_pred[:,i], color='black', linewidth=1, linestyle='--',
                         label='Dự báo', alpha=0.8)
            axes[i].set_title(f'{col} — Thực tế vs Dự báo ({model_name})')
            axes[i].set_ylabel('USD/bbl'); axes[i].legend(fontsize=9)
        axes[-1].set_xlabel('Ngày')
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.suptitle(f'{model_name} — Kết quả dự báo Test Set', fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.show()

        return m, hist, metrics_df

    print('✅ Hàm huấn luyện sẵn sàng!')
"""
))

cells.append(code(
"""# 12.5 Huấn luyện lần lượt các mô hình
# (Bỏ comment dòng bên dưới để chạy từng mô hình)
if TF_OK:
    all_metrics = []

    # --- GRU ---
    model_gru, hist_gru, met_gru = train_and_evaluate(
        'GRU', build_gru, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
        scaler_y, df_test, TARGET_COLS, LOOKBACK)
    all_metrics.append(met_gru)

    # --- LSTM ---
    model_lstm, hist_lstm, met_lstm = train_and_evaluate(
        'LSTM', build_lstm, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
        scaler_y, df_test, TARGET_COLS, LOOKBACK)
    all_metrics.append(met_lstm)

    # --- CNN-LSTM ---
    model_cnn, hist_cnn, met_cnn = train_and_evaluate(
        'CNN_LSTM', build_cnn_lstm, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
        scaler_y, df_test, TARGET_COLS, LOOKBACK)
    all_metrics.append(met_cnn)

    # --- Attention-LSTM ---
    model_attn, hist_attn, met_attn = train_and_evaluate(
        'Attention_LSTM', build_attn_lstm, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
        scaler_y, df_test, TARGET_COLS, LOOKBACK)
    all_metrics.append(met_attn)

    print('\\n✅ Huấn luyện xong tất cả mô hình!')
else:
    print('Cài TensorFlow để chạy: pip install tensorflow')
"""
))

cells.append(code(
"""# 12.6 Bảng so sánh tất cả mô hình
if TF_OK and 'all_metrics' in dir() and all_metrics:
    comparison = pd.concat(all_metrics, ignore_index=True)
    print('=== Bảng so sánh tổng hợp ===')
    pivot_compare = comparison.pivot_table(index='Model', values=['MAE','RMSE','R²','MAPE%'],
                                            aggfunc='mean').round(4)
    print(pivot_compare.sort_values('RMSE'))

    # Radar / Bar comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_plot = ['MAE', 'RMSE', 'R²', 'MAPE%']
    colors_m = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6']
    for i, metric in enumerate(metrics_plot):
        ax = axes[i//2][i%2]
        vals = pivot_compare[metric].sort_values(ascending=(metric != 'R²'))
        bars = ax.bar(vals.index, vals.values, color=colors_m[:len(vals)], edgecolor='white')
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(f'So sánh {metric} (trung bình 4 SP)', fontweight='bold')
        ax.set_ylabel(metric); ax.tick_params(axis='x', rotation=20)
    plt.suptitle('So sánh hiệu năng các mô hình', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()
else:
    print('Chạy cell 12.5 trước để có kết quả so sánh.')
"""
))

cells.append(code(
"""# 12.7 Lưu mô hình tốt nhất vào OUTPUT_DIR
if TF_OK and 'all_metrics' in dir() and all_metrics:
    comparison = pd.concat(all_metrics, ignore_index=True)
    best_model_name = comparison.groupby('Model')['RMSE'].mean().idxmin()
    print(f'🏆 Mô hình tốt nhất: {best_model_name}')
    models_dict = {'GRU': model_gru, 'LSTM': model_lstm,
                   'CNN_LSTM': model_cnn, 'Attention_LSTM': model_attn}
    best_model = models_dict[best_model_name]
    save_path = os.path.join(OUTPUT_DIR, f'{best_model_name}_final.keras')
    best_model.save(save_path)
    print(f'✅ Lưu tại: {save_path}')
    if IS_KAGGLE:
        print(f'📤 Download tại Kaggle: Output tab → {best_model_name}_final.keras')
"""
))

cells.append(code(
"""# 12.8 Lộ trình tiếp theo
print('''
╔══════════════════════════════════════════════════════════════════╗
║          LỘ TRÌNH TIẾP THEO — NÂNG CAO HIỆU NĂNG              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  BƯỚC 1: Hyperparameter Tuning (keras-tuner / Optuna)          ║
║    • Lookback : [10, 20, 30, 60]                               ║
║    • Units    : [64, 128, 256]                                  ║
║    • Dropout  : [0.1, 0.2, 0.3]                                ║
║    • LR       : [1e-4, 5e-4, 1e-3]                            ║
║                                                                 ║
║  BƯỚC 2: Multi-step forecasting (Horizon > 1)                  ║
║    • Dự báo 5, 10, 20 ngày tới                                 ║
║    • Thử Direct Multi-Output vs Recursive                       ║
║                                                                 ║
║  BƯỚC 3: Walk-Forward Cross-Validation                         ║
║    • Đánh giá chính xác hơn trên nhiều giai đoạn               ║
║                                                                 ║
║  BƯỚC 4: Ensemble                                               ║
║    • Trung bình dự báo từ 3 mô hình tốt nhất                   ║
║                                                                 ║
║  BƯỚC 5: Thêm biến ngoại sinh (Exogenous features)             ║
║    • Giá dầu thô: Brent, WTI, Dubai                            ║
║    • Chỉ số USD (DXY), Lãi suất Fed                            ║
║                                                                 ║
╚══════════════════════════════════════════════════════════════════╝
''')
"""
))

# ─────────────────────────────────────────────────────────────────
# WRITE NOTEBOOK
# ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        }
    },
    "cells": cells
}

out_path = 'petroleum_kaggle.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'✅ Đã tạo notebook: {out_path}')
print(f'   Tổng số cells  : {len(cells)}')
print(f'   Upload lên Kaggle: New Notebook → Upload → chọn {out_path}')
print(f'   Nhớ Add Dataset: price_petroleum.xlsx trước khi Run All!')

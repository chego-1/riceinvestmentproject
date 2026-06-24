import pandas as pd
import numpy as np
import statsmodels.api as sm
import openassetpricing as oap

# load Ken French data
print("Loading Ken French data...")
ff = pd.read_csv('data/raw/F-F_Research_Data_Factors.csv', 
                 skiprows=3,
                 names=['date', 'Mkt_RF', 'SMB', 'HML', 'RF'],
                 index_col=False)

# drop annual rows at bottom (they have 4 digit dates)
ff = ff[ff['date'].astype(str).str.len() == 6]
ff['date'] = ff['date'].astype(int)
ff['Mkt_RF'] = pd.to_numeric(ff['Mkt_RF'], errors='coerce')
ff['RF'] = pd.to_numeric(ff['RF'], errors='coerce')
ff = ff.dropna()

print(f"French data loaded: {len(ff)} months")

# load our backtest portfolio returns
print("Downloading portfolio returns...")
openap = oap.OpenAP()

signals = [
    'BM', 'Mom12m', 'GP', 'AssetGrowth',
    'TrendFactor', 'ChTax', 'EarningsStreak',
    'MS', 'NOA', 'ResidualMomentum', 'roaq',
    'DivSeason', 'AbnormalAccruals', 'CompositeDebtIssuance',
    'IntMom', 'MomVol', 'OScore', 'Accruals', 'IdioVol3F',
    'FirmAgeMom', 'dNoa', 'DelCOA', 'EntMult',
    'ShareIss1Y', 'NetDebtFinance', 'InvGrowth',
    'hire', 'BMdec', 'PS', 'Mom6mJunk'
]

dfs = []
loaded = []
for signal in signals:
    try:
        port = openap.dl_port('op', 'polars', [signal])
        port_pd = port.to_pandas()
        ls = port_pd[port_pd['port'] == 'LS'][['date', 'ret']].copy()
        ls['date'] = pd.to_datetime(ls['date']).dt.strftime('%Y%m').astype(int)
        ls = ls.rename(columns={'ret': signal})
        dfs.append(ls)
        loaded.append(signal)
        print(f"  {signal}: OK")
    except Exception as e:
        print(f"  {signal}: skipped ({e})")

print(f"\nLoaded {len(loaded)}/{len(signals)} signals")

combined = dfs[0]
for df in dfs[1:]:
    combined = combined.merge(df, on='date', how='inner')

combined['strategy_ret'] = combined[loaded].mean(axis=1)

# merge with Ken French data
merged = combined.merge(ff[['date', 'Mkt_RF', 'RF']], on='date')

# filter to 2000-2020
merged = merged[(merged['date'] >= 200001) & (merged['date'] <= 202012)]

# compute excess return = strategy return - risk free rate
merged['excess_ret'] = merged['strategy_ret'] - merged['RF']

print(f"\nRunning CAPM regression on {len(merged)} months ({len(loaded)} signals)...")

# CAPM regression: excess_ret = alpha + beta * Mkt_RF + error
X = sm.add_constant(merged['Mkt_RF'])
y = merged['excess_ret']

model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(model.summary())

alpha = model.params['const']
alpha_annual = alpha * 12
t_stat = model.tvalues['const']
p_value = model.pvalues['const']
beta = model.params['Mkt_RF']

print(f"\n=== CAPM ALPHA RESULTS ===")
print(f"Monthly Alpha: {alpha:.4f}%")
print(f"Annualized Alpha: {alpha_annual:.2f}%")
print(f"T-stat: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Beta: {beta:.4f}")
print(f"R-squared: {model.rsquared:.4f}")

if p_value < 0.05:
    print(f"\n✅ Alpha is statistically significant at 5% level")
    print(f"Strategy generates {alpha_annual:.2f}% annual excess return beyond market exposure")
else:
    print(f"\n❌ Alpha is NOT statistically significant")
    print(f"Cannot conclude strategy generates true excess returns")
import openassetpricing as oap
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

print("Downloading portfolio returns...")
ls_frames = []
loaded = []

for signal in signals:
    try:
        port = openap.dl_port('op', 'polars', [signal])
        ls = (port
              .filter(pl.col('port') == 'LS')
              .select(['date', 'ret'])
              .rename({'ret': signal}))
        ls_frames.append(ls)
        loaded.append(signal)
        print(f"  {signal}: OK")
    except Exception as e:
        print(f"  {signal}: skipped ({e})")

print(f"\nLoaded {len(loaded)}/{len(signals)} signals")

combined = ls_frames[0]
for frame in ls_frames[1:]:
    combined = combined.join(frame, on='date', how='inner')

combined = combined.filter(
    (pl.col('date') >= pl.date(2000, 1, 1)) &
    (pl.col('date') <= pl.date(2020, 12, 31))
)

combined = combined.with_columns(
    (pl.sum_horizontal([pl.col(s) for s in loaded]) / len(loaded)).alias('combined_ret')
)

df = combined.to_pandas()
df = df.sort_values('date')

df['cumulative'] = (1 + df['combined_ret'] / 100).cumprod()

total_return = df['cumulative'].iloc[-1] - 1
avg_monthly = df['combined_ret'].mean()
sharpe = (df['combined_ret'].mean() / df['combined_ret'].std()) * np.sqrt(12)

print(f"\n=== BACKTEST RESULTS (2000-2020, {len(loaded)} signals) ===")
print(f"Total Return: {total_return:.1%}")
print(f"Avg Monthly Return: {avg_monthly:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['cumulative'], linewidth=2, color='steelblue')
plt.axhline(y=1, color='gray', linestyle=':')
plt.title(f'Backtest: Combined Signal Strategy (2000-2020, {len(loaded)} signals)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return ($1 invested)')
plt.tight_layout()
plt.savefig('reports/backtest_results.png')
print("Chart saved to reports/backtest_results.png")

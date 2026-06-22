import openassetpricing as oap
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

openap = oap.OpenAP()

print("Downloading portfolio returns...")
bm = openap.dl_port('op', 'polars', ['BM'])
mom = openap.dl_port('op', 'polars', ['Mom12m'])
gp = openap.dl_port('op', 'polars', ['GP'])
ag = openap.dl_port('op', 'polars', ['AssetGrowth'])

# keep only the long-short portfolio (port == 'LS')
def get_ls(df, name):
    return (df
        .filter(pl.col('port') == 'LS')
        .select(['date', 'ret'])
        .rename({'ret': name}))

bm_ls = get_ls(bm, 'BM')
mom_ls = get_ls(mom, 'Mom12m')
gp_ls = get_ls(gp, 'GP')
ag_ls = get_ls(ag, 'AssetGrowth')

# combine all signals
combined = bm_ls.join(mom_ls, on='date').join(gp_ls, on='date').join(ag_ls, on='date')

# filter to 2000-2020
combined = combined.filter(
    (pl.col('date') >= pl.date(2000, 1, 1)) &
    (pl.col('date') <= pl.date(2020, 12, 31))
)

# equal weight the signals into one combined strategy
combined = combined.with_columns(
    ((pl.col('BM') + pl.col('Mom12m') + pl.col('GP') + pl.col('AssetGrowth')) / 4).alias('combined_ret')
)

# convert to pandas for math
df = combined.to_pandas()
df = df.sort_values('date')

# cumulative returns
df['cumulative'] = (1 + df['combined_ret'] / 100).cumprod()

# stats
total_return = df['cumulative'].iloc[-1] - 1
avg_monthly = df['combined_ret'].mean()
sharpe = (df['combined_ret'].mean() / df['combined_ret'].std()) * np.sqrt(12)

print(f"\n=== BACKTEST RESULTS (2000-2020) ===")
print(f"Total Return: {total_return:.1%}")
print(f"Avg Monthly Return: {avg_monthly:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

# plot
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['cumulative'], linewidth=2, color='steelblue')
plt.axhline(y=1, color='gray', linestyle=':')
plt.title('Backtest: Combined Signal Strategy (2000-2020)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return ($1 invested)')
plt.tight_layout()
plt.savefig('reports/backtest_results.png')
print("Chart saved to reports/backtest_results.png")
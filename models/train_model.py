import openassetpricing as oap
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import os

print("Downloading signals...")
openap = oap.OpenAP()
data = openap.dl_signal('pandas', [
    'BM', 'Mom12m', 'GP', 'AssetGrowth',
    'TrendFactor', 'ChTax', 'EarningsStreak',
    'MS', 'NOA', 'ResidualMomentum', 'roaq',
    'DivSeason', 'AbnormalAccruals', 'CompositeDebtIssuance',
    'IntMom', 'MomVol', 'OScore', 'Accruals', 'IdioVol3F',
    'FirmAgeMom', 'dNoa', 'DelCOA', 'EntMult',
    'ShareIss1Y', 'NetDebtFinance', 'InvGrowth',
    'hire', 'BMdec', 'PS', 'Mom6mJunk'
])

# Factor-momentum features, from "Factor Momentum Everywhere" (Ehsani & Linnainmaa) in
# papers/summaries.json: prior 1-month factor return is the strongest predictor of
# near-term factor performance, with 12-month lookback also adding value. These are
# market-wide (same value for every stock in a given month), unlike the characteristics above.
print("Loading Fama-French factor data for factor-momentum features...")
ff = pd.read_csv(
    'data/raw/F-F_Research_Data_Factors.csv',
    skiprows=3,
    names=['yyyymm', 'Mkt_RF', 'SMB', 'HML', 'RF'],
    index_col=False,
)
ff = ff[ff['yyyymm'].astype(str).str.len() == 6]
ff['yyyymm'] = ff['yyyymm'].astype(int)
for col in ['Mkt_RF', 'SMB', 'HML']:
    ff[col] = pd.to_numeric(ff[col], errors='coerce')
ff = ff.dropna().sort_values('yyyymm').reset_index(drop=True)

factor_cols = ['Mkt_RF', 'SMB', 'HML']
for col in factor_cols:
    ff[f'{col}_1m'] = ff[col]
    ff[f'{col}_12m'] = ff[col].rolling(12).sum()

ff_features = [f'{c}_1m' for c in factor_cols] + [f'{c}_12m' for c in factor_cols]
ff_merge = ff[['yyyymm'] + ff_features]

print("Building target variable...")
data = data.sort_values(['permno', 'yyyymm'])
data = data[data['yyyymm'] <= 199912]
data['target'] = data.groupby('permno')['Mom12m'].shift(-1)

print("Merging factor-momentum features onto signal panel (by yyyymm)...")
data = data.merge(ff_merge, on='yyyymm', how='left')
data = data.dropna(subset=['target'] + ff_features)

features = [
    'BM', 'Mom12m', 'GP', 'AssetGrowth',
    'TrendFactor', 'ChTax', 'EarningsStreak',
    'MS', 'NOA', 'ResidualMomentum', 'roaq',
    'DivSeason', 'AbnormalAccruals', 'CompositeDebtIssuance',
    'IntMom', 'MomVol', 'OScore', 'Accruals', 'IdioVol3F',
    'FirmAgeMom', 'dNoa', 'DelCOA', 'EntMult',
    'ShareIss1Y', 'NetDebtFinance', 'InvGrowth',
    'hire', 'BMdec', 'PS', 'Mom6mJunk'
] + ff_features

print(f"Clean dataset shape: {data.shape}")

X = data[features]
y = data['target']

# Time-based split (fixed session 3). The old split did `data.iloc[:split]` after
# sorting by ['permno', 'yyyymm'] — since permno is the primary sort key, that put
# entire early-permno stocks' full histories in train and entire later-permno stocks'
# full histories in test. That's a cross-sectional (leave-some-stocks-out) split
# disguised as a time split, not the out-of-time validation Freyberger et al. and
# Rossi (2018) recommend. Fixed to split strictly on calendar month instead.
months = sorted(data['yyyymm'].unique())
cutoff_month = months[int(len(months) * 0.8)]
print(f"Time-based split: train yyyymm <= {cutoff_month}, test yyyymm > {cutoff_month}")

train_mask = data['yyyymm'] <= cutoff_month
test_mask = data['yyyymm'] > cutoff_month
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

print("Training LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=127,
    min_child_samples=200,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# OOS R^2 (Gu, Kelly & Xiu 2020 use this as their primary validation metric, not RMSE alone)
ss_res = np.sum((y_test - preds) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
oos_r2 = 1 - ss_res / ss_tot

# Directional accuracy (Rossi 2018 evaluates BRT on this alongside MSE)
directional_accuracy = (np.sign(preds) == np.sign(y_test)).mean()

print(f"\nRMSE: {rmse:.4f}")
print(f"OOS R^2: {oos_r2:.4f}")
print(f"Directional accuracy: {directional_accuracy:.1%}")

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

os.makedirs('models', exist_ok=True)
with open('models/lgbm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to models/lgbm_model.pkl")
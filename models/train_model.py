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

print("Building target variable...")
data = data.sort_values(['permno', 'yyyymm'])
data = data[data['yyyymm'] <= 199912]
data['target'] = data.groupby('permno')['Mom12m'].shift(-1)
data = data.dropna(subset=['target'])

features = [
    'BM', 'Mom12m', 'GP', 'AssetGrowth',
    'TrendFactor', 'ChTax', 'EarningsStreak',
    'MS', 'NOA', 'ResidualMomentum', 'roaq',
    'DivSeason', 'AbnormalAccruals', 'CompositeDebtIssuance',
    'IntMom', 'MomVol', 'OScore', 'Accruals', 'IdioVol3F',
    'FirmAgeMom', 'dNoa', 'DelCOA', 'EntMult',
    'ShareIss1Y', 'NetDebtFinance', 'InvGrowth',
    'hire', 'BMdec', 'PS', 'Mom6mJunk'
]

print(f"Clean dataset shape: {data.shape}")

X = data[features]
y = data['target']

split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

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
print(f"\nRMSE: {rmse:.4f}")

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
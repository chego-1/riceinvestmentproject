import openassetpricing as oap
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

print("Downloading signals...")
openap = oap.OpenAP()
data = openap.dl_signal('pandas', ['BM', 'Mom12m', 'GP', 'AssetGrowth', 
        'TrendFactor', 'ChTax', 'EarningsStreak', 
        'MS', 'NOA', 'ResidualMomentum', 'roaq'])

# compute forward 1-month return as the target
print("Building target variable...")
data = data.sort_values(['permno', 'yyyymm'])
data = data[data['yyyymm'] <= 199912]
data['target'] = data.groupby('permno')['Mom12m'].shift(-1)

# drop rows with no target
data = data.dropna(subset=['target'])

# define features
features = ['BM', 'Mom12m', 'GP', 'AssetGrowth', 
            'TrendFactor', 'ChTax', 'EarningsStreak',
            'MS', 'NOA', 'ResidualMomentum', 'roaq']

print(f"Clean dataset shape: {data.shape}")

X = data[features]
y = data['target']

# train/test split — use last 20% of time as test
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

# train LightGBM
print("Training LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

# evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\nRMSE: {rmse:.4f}")

# feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# save model
os.makedirs('models', exist_ok=True)
with open('models/lgbm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to models/lgbm_model.pkl")
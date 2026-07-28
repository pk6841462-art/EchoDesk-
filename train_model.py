"""
Train ML Models and Export focus_model.pkl
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

data_path = os.path.join('Dataset', 'study_dataset.csv')
if not os.path.exists(data_path):
    data_path = os.path.join('..', 'Dataset', 'study_dataset.csv')

df = pd.read_csv(data_path)

posture_map = {'Good': 3, 'Slight Lean': 2, 'Moderate Lean': 1, 'Poor': 0}
df['posture_score'] = df['posture'].map(posture_map).fillna(3).astype(int)
df['owner_mode_bin'] = df['owner_mode'].astype(int)
df['light_temperature_ratio'] = df['light'] / (df['temperature'] + 1.0)
df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1.0)
df['stress_flag'] = (
    (df['temperature'] > 32.0) | 
    (df['light'] < 180.0) | 
    (df['study_duration'] > 90.0) | 
    (df['posture_score'] == 0)
).astype(int)

feature_cols = [
    'temperature', 'humidity', 'light', 'study_duration', 
    'posture_score', 'user_presence', 'owner_mode_bin', 
    'light_temperature_ratio', 'temp_humidity_ratio', 'stress_flag'
]

X = df[feature_cols]
y_focus = df['focus_score']

X_train, X_test, y_train, y_test = train_test_split(X, y_focus, test_size=0.20, random_state=42)

models = {
    'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
    'Decision Tree': make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=42)),
    'Random Forest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
    'Gradient Boosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42)),
    'Support Vector Regressor (SVR)': make_pipeline(StandardScaler(), SVR(C=10.0))
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({'Model': name, 'R2 Score': r2, 'MAE': mae, 'RMSE': rmse})
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
print("=== MODEL COMPARISON LEADERBOARD ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R2 Score']
best_model = fitted_models[best_model_name]

print(f"\nBEST MODEL SELECTED: {best_model_name} (R2 Score: {best_r2:.4f})")

os.makedirs('ML', exist_ok=True)
joblib.dump(best_model, os.path.join('ML', 'focus_model.pkl'))
joblib.dump(best_model, 'focus_model.pkl')
print(f"Model saved cleanly to ML/focus_model.pkl")

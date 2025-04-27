import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load synthetic data
data = pd.read_csv('synthetic_data.csv')

# Compute delta_SOC
data['delta_SOC'] = data['SOC'].diff().fillna(0)

# Define features and target
features = ['SOC', 'delta_SOC', 'Cell_Temperature_Average', 'Battery_Current']
X = data[features]
y = data['degradation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"RandomForest Test MSE: {mse:.6f}")

# Save model
joblib.dump(model, 'degradation_model.pkl')
print("Degradation model trained and saved to 'degradation_model.pkl'.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load data and model
data = pd.read_csv('synthetic_data.csv')
degradation_model = joblib.load('degradation_model.pkl')
results = pd.read_csv('strategy_results.csv')

# Ensure timestamp_utc is in datetime format
data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'])

# Compute delta_SOC for model predictions
data['delta_SOC'] = data['SOC'].diff().fillna(0)

# Plot 1: Synthetic Data
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(data['timestamp_utc'], data['SOC'], label='SOC (%)')
plt.title('Synthetic SOC over Time')
plt.xlabel('Time')
plt.ylabel('SOC (%)')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(data['timestamp_utc'], data['Battery_Current'], label='Battery Current (A)')
plt.title('Synthetic Battery Current over Time')
plt.xlabel('Time')
plt.ylabel('Current (A)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(data['timestamp_utc'], data['RTM_price'], label='RTM Price ($/MWh)')
plt.title('Synthetic RTM Prices over Time')
plt.xlabel('Time')
plt.ylabel('Price ($/MWh)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: RandomForest Model Performance
features = ['SOC', 'delta_SOC', 'Cell_Temperature_Average', 'Battery_Current']
X = data[features]
y = data['degradation']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = degradation_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Degradation')
plt.ylabel('Predicted Degradation')
plt.title(f'RandomForest Model: Actual vs Predicted Degradation (MSE: {mse:.6f})')
plt.show()

# Plot 3: Performance Comparison
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
scenarios = results['Scenario'].unique()
index = np.arange(len(scenarios))

for i, strategy in enumerate(['AI', 'Charge', 'Discharge', 'Simple']):
    strategy_rewards = [results[(results['Scenario'] == s) & (results['Strategy'] == strategy)]['Reward'].values[0] for s in scenarios]
    plt.bar(index + i * bar_width, strategy_rewards, bar_width, label=strategy)

plt.xlabel('Scenario')
plt.ylabel('Total Reward')
plt.title('Performance Comparison Across Scenarios')
plt.xticks(index + 1.5 * bar_width, scenarios)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

# Parameters
start_time = datetime(2025, 4, 3, 0, 0)
end_time = start_time + timedelta(days=7)
time_step = timedelta(minutes=1)
num_points = int((end_time - start_time) / time_step)  # 10,080

# Generate timestamps
timestamps = [start_time + i * time_step for i in range(num_points)]

# Generate DAM prices (hourly)
num_hours = 7 * 24  # 168
dam_prices_hourly = 50 + 20 * np.sin(2 * np.pi * np.arange(num_hours) / 24) + np.random.normal(0, 5, num_hours)
dam_prices = np.repeat(dam_prices_hourly, 60)  # Repeat for each minute

# Generate RTM prices (every 5 minutes)
num_5min = 7 * 24 * 12  # 2,016
rtm_prices_5min = dam_prices_hourly.repeat(12) + np.random.normal(0, 10, num_5min)
rtm_prices = np.repeat(rtm_prices_5min, 5)  # Repeat for each minute

# Battery parameters
C = 360.0  # Ah
V_min = 1200.0  # V
V_max = 1400.0  # V
N_cells = 414
I_max = 74.4  # A
seg_name = "Battery_String_01_4B6BAT2"
seg_id = "5cada56c-8ad1-11ef-94ce-42010afa015a"

# Initialize lists
soc = [50.0]  # Initial SOC
current = [0.0]
voltage = [V_min + (V_max - V_min) * (soc[0] / 100)]
cell_voltage_avg = [voltage[0] / N_cells]
max_cell_voltage = [cell_voltage_avg[0] + 0.003]
min_cell_voltage = [cell_voltage_avg[0] - 0.002]
base_temperature = 20 + 5 * np.sin(2 * np.pi * np.arange(num_points) / 1440) + np.random.normal(0, 1, num_points)
cell_temp_avg = base_temperature
cell_temp_min = cell_temp_avg - 1
cell_temp_max = cell_temp_avg + 1
max_cell_soc = [soc[0] + 2]
min_cell_soc = [soc[0] - 1]
available_charge_capacity = [((100 - soc[0]) / 100) * C]
available_discharge_capacity = [(soc[0] / 100) * C]

# Simulate battery operation
for i in range(1, num_points):
    action = np.random.choice([0, 1, 2])  # 0: idle, 1: charge, 2: discharge
    curr = 0.0 if action == 0 else (I_max if action == 1 else -I_max)
    delta_charge = curr * (1 / 60)  # Ah per minute
    new_soc = max(0.0, min(100.0, soc[-1] + (delta_charge / C) * 100))
    new_voltage = V_min + (V_max - V_min) * (new_soc / 100)
    new_cell_voltage_avg = new_voltage / N_cells
    new_max_cell_voltage = new_cell_voltage_avg + 0.003
    new_min_cell_voltage = new_cell_voltage_avg - 0.002
    new_available_charge_capacity = ((100 - new_soc) / 100) * C
    new_available_discharge_capacity = (new_soc / 100) * C
    soc.append(new_soc)
    current.append(curr)
    voltage.append(new_voltage)
    cell_voltage_avg.append(new_cell_voltage_avg)
    max_cell_voltage.append(new_max_cell_voltage)
    min_cell_voltage.append(new_min_cell_voltage)
    available_charge_capacity.append(new_available_charge_capacity)
    available_discharge_capacity.append(new_available_discharge_capacity)
    max_cell_soc.append(new_soc + 2)
    min_cell_soc.append(new_soc - 1)

# Compute delta_SOC
delta_soc = [0.0] + [soc[i] - soc[i-1] for i in range(1, len(soc))]

# Compute synthetic degradation
a, b, c, d = 0.0001, 0.001, 0.0001, 0.0001
degradation = [a * (s - 50)**2 + b * abs(ds) + c * (t - 20)**2 + d * abs(curr)
               for s, ds, t, curr in zip(soc, delta_soc, cell_temp_avg, current)]

# Create DataFrame with all columns from sample
data = pd.DataFrame({
    'timestamp_utc': timestamps,
    'seg_name': [seg_name] * num_points,
    'seg_id': [seg_id] * num_points,
    'Battery_Current': current,
    'Battery_Voltage': voltage,
    'Cell_Voltage_Average': cell_voltage_avg,
    'Max_1_Cell_Voltage_Value': max_cell_voltage,
    'Min_1_Cell_Voltage_Value': min_cell_voltage,
    'Cell_Temperature_Average': cell_temp_avg,
    'Cell_Temperature_1_Min': cell_temp_min,
    'Cell_Temperature_1_Max': cell_temp_max,
    'SOC': soc,
    'Max_Cell_SOC': max_cell_soc,
    'Min_Cell_SOC': min_cell_soc,
    'Available_Charge_Capacity': available_charge_capacity,
    'Available_Discharge_Capacity': available_discharge_capacity,
    '__index_level_0__': range(num_points),
    'DAM_price': dam_prices,
    'RTM_price': rtm_prices,
    'degradation': degradation
})

# Save to CSV
data.to_csv('synthetic_data.csv', index=False)
print("Synthetic data generated and saved to 'synthetic_data.csv'.")
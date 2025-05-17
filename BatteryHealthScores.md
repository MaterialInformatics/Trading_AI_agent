# Battery Health Score Calculation from Parquet Files

This document provides a detailed description of how various battery health scores are calculated from parquet files containing battery performance data. These scores assess different facets of battery health, such as state of charge (SOC) consistency, voltage behavior, temperature behavior, capacity integrity, balancing response, and the ability to reach maximum SOC. Additionally, a cumulative score is computed as a weighted average of these individual scores to provide an overall health assessment. The calculations are performed per battery segment, identified by `seg_name`, using data typically representing a day's worth of measurements.

The parquet files include the following key columns:
- **`timestamp_utc`**: Timestamp of the data point (in UTC).
- **`SOC`**: State of Charge of the battery (in %).
- **`Battery_Current`**: Current flowing through the battery (in Amperes, A).
- **`Battery_Voltage`**: Voltage across the battery (in Volts, V).
- **`seg_name`**: Unique identifier for each battery segment.
- **`Cell_Temperature_Average`**: Average temperature of the battery cells (in °C).
- **`Available_Charge_Capacity`**: Available capacity for charging (in Ampere-hours, Ah).
- **`Available_Discharge_Capacity`**: Available capacity for discharging (in Ah).

Below, each health score is described in detail, including the mathematical formulas, intermediate steps, and the rationale behind the calculations. The final section explains the cumulative score and how scores are aggregated across multiple files.

---

## 1. SOC Consistency Score

The **SOC Consistency Score** measures how closely the actual SOC aligns with the expected SOC, which is derived by integrating the battery current over time and adjusting for balancing currents. A high score indicates that the SOC behaves as expected based on charge flow.

### Calculation Steps

1. **Filter and Sort Data:**
   - Select all rows for a specific `seg_name` from the parquet file.
   - Sort the data by `timestamp_utc` in ascending order to ensure chronological processing.

2. **Compute Time Differences:**
   - Calculate the time difference between consecutive timestamps in hours:
     ```math
     \Delta t_i = \frac{t_{i+1} - t_i}{3600}
     ```
     where \( t_i \) is the timestamp at index \( i \) (in seconds), and \( \Delta t_i \) is in hours.

3. **Extract and Compute Capacities:**
   - Extract:
     - \( I_i \): `Battery_Current` at timestamp \( t_i \) (in A).
     - \( C_{\text{charge}, i} \): `Available_Charge_Capacity` at \( t_i \) (in Ah).
     - \( C_{\text{discharge}, i} \): `Available_Discharge_Capacity` at \( t_i \) (in Ah).
   - Compute the total capacity at each timestamp:
     ```math
     C_{\text{total}, i} = C_{\text{charge}, i} + C_{\text{discharge}, i}
     ```

4. **Smooth Current:**
   - Apply a rolling mean to \( I_i \) with a window size of 50 to reduce noise:
     ```math
     I_{\text{smoothed}, i} = \text{rolling mean of } I_i \text{ over 50 previous points}
     ```
     - If fewer than 50 points are available, use the available points.

5. **Compute Smoothed Incremental Charge:**
   - Use the trapezoidal rule to estimate the charge added or removed between consecutive timestamps:
     ```math
     \Delta Q_{\text{smoothed}, i} = \left( \frac{I_{\text{smoothed}, i} + I_{\text{smoothed}, i+1}}{2} \right) \Delta t_i
     ```
     where \( \Delta Q_{\text{smoothed}, i} \) is in Ah.

6. **Compute SOC Changes:**
   - **SOC change due to charge/discharge:**
     ```math
     \Delta \text{SOC}_{\text{charge}, i} = -\frac{\Delta Q_{\text{smoothed}, i}}{C_{\text{total}, i}} \times 100\%
     ```
     - The negative sign accounts for discharge (positive current) reducing SOC and charge (negative current) increasing SOC.
   - **SOC change due to balancing current:**
     - Assume a balancing current \( I_{\text{bal}} \) (typically a small constant, e.g., derived from system specs or data).
     ```math
     \Delta \text{SOC}_{\text{bal}, i} = \frac{I_{\text{bal}} \Delta t_i}{C_{\text{total}, i}} \times 100\%
     ```

7. **Compute Cumulative SOC Changes:**
   - Accumulate the SOC changes over time:
     ```math
     \Delta \text{SOC}_{\text{cum}, i} = \sum_{k=1}^{i} \Delta \text{SOC}_{\text{charge}, k}
     ```
     ```math
     \Delta \text{SOC}_{\text{bal cum}, i} = \sum_{k=1}^{i} \Delta \text{SOC}_{\text{bal}, k}
     ```

8. **Compute Expected SOC:**
   - Start with the initial SOC (\( \text{SOC}_0 \)) and adjust based on cumulative changes:
     ```math
     \text{SOC}_{\text{expected}, i} = \text{SOC}_0 + \Delta \text{SOC}_{\text{cum}, i} - \Delta \text{SOC}_{\text{bal cum}, i}
     ```
     where \( \text{SOC}_0 = \text{SOC}_1 \) (the first recorded SOC).

9. **Compute Deviations:**
   - Compare the actual SOC to the expected SOC:
     ```math
     d_i = \text{SOC}_{\text{actual}, i} - \text{SOC}_{\text{expected}, i}
     ```
     where \( \text{SOC}_{\text{actual}, i} \) is the `SOC` value at \( t_i \).

10. **Outlier Detection:**
    - Calculate the interquartile range (IQR) of deviations \( d_i \):
      - \( Q1 \): 25th percentile of \( d_i \).
      - \( Q3 \): 75th percentile of \( d_i \).
      - \( \text{IQR} = Q3 - Q1 \).
    - Identify outliers:
      - Mild outliers: \( |d_i| > 1.5 \times \text{IQR} \).
      - Severe outliers: \( |d_i| > 3 \times \text{IQR} \).

11. **Compute Adjustment Factor:**
    - Combine the proportion of outliers and the mean absolute deviation:
      ```math
      f_{\text{adjusted}} = \min\left( \left( \frac{\text{number of mild outliers}}{n} \right) \times 0.5 + \left( \frac{\text{number of severe outliers}}{n} \right) \times 1.0 + 0.01 \times \text{mean}|d_i|, 1.0 \right)
      ```
      where \( n \) is the total number of data points.

12. **Compute SOC Consistency Score:**
    - Convert the adjustment factor to a score between 0 and 100:
      ```math
      \text{score} = \max(0, 100 \times (1 - f_{\text{adjusted}}))
      ```

---

## 2. Voltage Behavior Score

The **Voltage Behavior Score** evaluates how a battery’s voltage, adjusted for internal resistance, compares to the average adjusted voltage across all batteries at the same timestamp. It detects anomalies in voltage behavior.

### Calculation Steps

1. **Filter and Sort Data:**
   - Select data for a specific `seg_name` and sort by `timestamp_utc`.

2. **Compute Voltage and Current Differences:**
   - Calculate differences between consecutive measurements:
     ```math
     \Delta V_i = V_{i} - V_{i-1}
     ```
     ```math
     \Delta I_i = I_{i} - I_{i-1}
     ```
     where \( V_i \) is `Battery_Voltage` and \( I_i \) is `Battery_Current` at \( t_i \).

3. **Estimate Internal Resistance:**
   - Compute the average resistance where current changes occur:
     ```math
     R_i = \text{mean of } \left( \frac{\Delta V_j}{\Delta I_j} \mid \Delta I_j \neq 0 \right)
     ```
     - If no \( \Delta I_j \neq 0 \), default \( R_i = 0.01 \) Ω.

4. **Compute Adjusted Voltage:**
   - Adjust the voltage to account for the effect of current on internal resistance:
     ```math
     V_{\text{adjusted}, i} = V_i + I_i \times R_i
     ```

5. **Compute Mean Adjusted Voltage Across All Batteries:**
   - For each timestamp \( t \), calculate the mean adjusted voltage across all segments:
     ```math
     \text{mean\_voltage}_t = \text{mean of } V_{\text{adjusted}, i} \text{ for all batteries at } t
     ```

6. **Compute Deviations:**
   - Measure how the battery’s adjusted voltage deviates from the mean:
     ```math
     d_i = V_{\text{adjusted}, i} - \text{mean\_voltage}_{t_i}
     ```

7. **Compute Z-Scores:**
   - Standardize the deviations:
     ```math
     z_i = \frac{d_i - \mu_d}{\sigma_d}
     ```
     where \( \mu_d \) and \( \sigma_d \) are the mean and standard deviation of \( d_i \).

8. **Identify Outliers:**
   - Mild outliers: \( |z_i| > 1.5 \).
   - Severe outliers: \( |z_i| > 2.5 \).

9. **Compute Adjustment Factor:**
   - Penalize based on outliers and deviation magnitude:
     ```math
     f_{\text{adjusted}} = \min\left( \left( \frac{\text{number of mild outliers} + 2 \times \text{number of severe outliers}}{n} \right) + 0.05 \times \text{mean}|d_i|, 1.0 \right)
     ```

10. **Compute Voltage Behavior Score:**
    - Calculate the score:
      ```math
      \text{score} = \max(0, 100 \times (1 - f_{\text{adjusted}}))
      ```

---

## 3. Temperature Behavior Score

The **Temperature Behavior Score** assesses how a battery’s temperature deviates from the average temperature across all batteries at the same timestamp, identifying abnormal thermal behavior.

### Calculation Steps

1. **Filter and Sort Data:**
   - Select data for a specific `seg_name` and sort by `timestamp_utc`.

2. **Compute Mean Temperature Across All Batteries:**
   - For each timestamp \( t \):
     ```math
     \text{mean\_temp}_t = \text{mean of } \text{Cell\_Temperature\_Average}_i \text{ for all batteries at } t
     ```

3. **Compute Deviations:**
   - Calculate the difference from the mean:
     ```math
     d_i = \text{Cell\_Temperature\_Average}_i - \text{mean\_temp}_{t_i}
     ```

4. **Compute Z-Scores:**
   - Standardize the deviations:
     ```math
     z_i = \frac{d_i - \mu_d}{\sigma_d}
     ```
     where \( \mu_d \) and \( \sigma_d \) are the mean and standard deviation of \( d_i \).

5. **Identify Outliers:**
   - Mild outliers: \( |z_i| > 1.5 \).
   - Severe outliers: \( |z_i| > 2.5 \).

6. **Compute Adjustment Factor:**
   - Combine outlier counts and deviation magnitude:
     ```math
     f_{\text{adjusted}} = \min\left( \left( \frac{\text{number of mild outliers} + 2 \times \text{number of severe outliers}}{n} \right) + 0.05 \times \text{mean}|d_i|, 1.0 \right)
     ```

7. **Compute Temperature Behavior Score:**
   - Calculate the score:
     ```math
     \text{score} = \max(0, 100 \times (1 - f_{\text{adjusted}}))
     ```

---

## 4. Capacity Integrity Score

The **Capacity Integrity Score** estimates the battery’s true capacity by analyzing the relationship between SOC deviations and cumulative charge, then compares it to the nominal capacity.

### Calculation Steps

1. **Extract Intermediate Data:**
   - Use outputs from the SOC Consistency Score calculation:
     - \( \Delta Q_{\text{smoothed}, i} \): Smoothed incremental charge.
     - \( d_i \): SOC deviations.
     - \( C_{\text{total}, i} \): Total capacity.

2. **Compute Cumulative Charge:**
   - Sum the incremental charges:
     ```math
     Q_{\text{cum}, i} = \sum_{k=1}^{i} \Delta Q_{\text{smoothed}, k}
     ```

3. **Compute Nominal Capacity:**
   - Average the total capacity over all timestamps:
     ```math
     C_{\text{nominal}} = \text{mean of } C_{\text{total}, i}
     ```

4. **Perform Linear Regression:**
   - Regress SOC deviations \( d_i \) against cumulative charge \( Q_{\text{cum}, i} \):
     ```math
     d_i = a + b \times Q_{\text{cum}, i}
     ```
     where \( b \) is the slope (in %/Ah).

5. **Estimate True Capacity:**
   - Use the slope to adjust the nominal capacity:
     ```math
     C_{\text{estimated}} = \frac{1}{\frac{b}{100} + \frac{1}{C_{\text{nominal}}}}
     ```
     - If \( \frac{b}{100} + \frac{1}{C_{\text{nominal}}} \leq 0 \), set \( C_{\text{estimated}} = C_{\text{nominal}} \).

6. **Compute Relative Deviation:**
   - Measure the deviation from nominal capacity:
     ```math
     \text{deviation} = \left| \frac{C_{\text{estimated}} - C_{\text{nominal}}}{C_{\text{nominal}}} \right|
     ```

7. **Compute Capacity Integrity Score:**
   - Calculate the score:
     ```math
     \text{score} = \max(0, 100 \times (1 - \min(\text{deviation}, 1.0)))
     ```

---

## 5. Balancing Response Score

The **Balancing Response Score** evaluates how effectively the battery responds to balancing currents when SOC exceeds 80%, comparing actual SOC changes to expected changes.

### Calculation Steps

1. **Filter and Sort Data:**
   - Select data for a specific `seg_name` and sort by `timestamp_utc`.

2. **Identify High Charge Periods:**
   - Flag timestamps where:
     ```math
     \text{high\_charge} = \text{SOC} > 80\%
     ```

3. **Compute Time Differences:**
   - Calculate time intervals:
     ```math
     \Delta t_i = \frac{t_{i+1} - t_i}{3600}
     ```

4. **Compute Expected SOC Drop Due to Balancing:**
   - Cumulative effect of balancing current:
     ```math
     \text{expected\_soc\_drop}_i = \frac{I_{\text{bal}} \times \sum_{k=1}^{i} \Delta t_k}{C_{\text{total}, 0}} \times 100\%
     ```
     where \( C_{\text{total}, 0} \) is the initial total capacity.

5. **Compute Actual SOC Differences:**
   - Calculate changes in SOC:
     ```math
     \text{soc\_diff}_i = \text{SOC}_{i} - \text{SOC}_{i-1}
     ```

6. **Compute Expected SOC Differences:**
   - Expected change due to balancing:
     ```math
     \text{expected\_diff}_i = -\left( \text{expected\_soc\_drop}_i - \text{expected\_soc\_drop}_{i-1} \right)
     ```

7. **Compute Deviations for High Charge Periods:**
   - For indices where `high_charge` is True:
     ```math
     d_i = \text{soc\_diff}_i - \text{expected\_diff}_i
     ```

8. **Outlier Detection:**
   - Compute z-scores:
     ```math
     z_i = \frac{d_i - \mu_d}{\sigma_d}
     ```
   - Mild outliers: \( |z_i| > 1.5 \).
   - Severe outliers: \( |z_i| > 2.5 \).

9. **Compute Adjustment Factor:**
   - Penalize based on outliers and deviations:
     ```math
     f_{\text{adjusted}} = \min\left( \left( \frac{\text{number of mild outliers} + 2 \times \text{number of severe outliers}}{n} \right) + 0.05 \times \text{mean}|d_i|, 1.0 \right)
     ```

10. **Compute Balancing Response Score:**
    - Calculate the score:
      ```math
      \text{score} = \max(0, 100 \times (1 - f_{\text{adjusted}}))
      ```

---

## 6. Max SOC Reachable Score

The **Max SOC Reachable Score** measures how close the battery’s maximum observed SOC comes to 100%, indicating its ability to fully charge.

### Calculation Steps

1. **Compute Maximum SOC:**
   - Find the highest SOC value:
     ```math
     \text{max\_soc} = \max(\text{SOC})
     ```

2. **Compute Deviation:**
   - Calculate the shortfall from 100%:
     ```math
     \text{deviation} = \frac{100 - \text{max\_soc}}{100}
     ```

3. **Compute Max SOC Reachable Score:**
   - Apply a penalty and compute the score:
     ```math
     \text{score} = \max(0, 100 \times (1 - \min(\text{deviation} + 0.05, 1.0)))
     ```
     - The 0.05 penalty ensures small deviations still impact the score.

---

## 7. Cumulative Score

The **Cumulative Score** provides an overall assessment of battery health by combining the individual scores with predefined weights.

### Calculation

- **Assign Weights:**
  - SOC Consistency Score: 0.25
  - Voltage Behavior Score: 0.15
  - Temperature Behavior Score: 0.25
  - Capacity Integrity Score: 0.15
  - Balancing Response Score: 0.1
  - Max SOC Reachable Score: 0.1

- **Compute Weighted Average:**
  ```math
  \text{Cumulative Score} = \sum_{\text{score}} \text{weight}_{\text{score}} \times \text{score}

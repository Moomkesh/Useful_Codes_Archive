import pandas as pd
import numpy as np
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures

# Sample time series data
data = {
    'Timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'RotSpeed': list(range(10))
}
df = pd.DataFrame(data)
df.set_index('Timestamp', inplace=True)

# Define lag periods
num_lags = 3  
num_lags_list = list(range(1, num_lags))

# Step 1: Apply Lag Features (Keep NaN for now)
lf = LagFeatures(
    variables=['RotSpeed'],
    periods=num_lags_list,  
    fill_value=None,  # Keep NaN values initially
    drop_na=False,  # Do not drop rows immediately
    drop_original=False,
    sort_index=True,
)

df_lagged = lf.fit_transform(df)

# Step 2: Apply Rolling Window Features (Ensure min_periods=3)
wf = WindowFeatures(
    variables=['RotSpeed'],
    window=3,  
    functions=['mean'],
    drop_original=False,
    missing_values='ignore'
).set_params(min_periods=3)  # Ensure first valid rolling mean at 2023-01-03

df_transformed = wf.fit_transform(df_lagged)

# Step 3: Add Exponential Weighted Moving Average (EWM)
df_transformed[f"RotSpeed_ewm_3"] = df_transformed["RotSpeed"].ewm(span=3, adjust=False).mean()

# Step 4: Prepare X and y
df_transformed["y"] = df_transformed["RotSpeed"].shift(-1)  

# Drop NaN selectively to ensure 2023-01-03 is included
df_final = df_transformed.dropna(subset=["y", "RotSpeed_window_3_mean", "RotSpeed_ewm_3"])

# Display the final transformed DataFrame
print(df_final)

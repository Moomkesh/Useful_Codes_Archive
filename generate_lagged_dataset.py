import pandas as pd
import numpy as np

# Function to generate lag features for multiple columns
def generate_lagged_dataset(df, feature_names, target_column, num_lags):
    if num_lags == 0:
        # Return original feature set (excluding Time) and target
        X = df[feature_names].copy()
        y = df[target_column].copy()
        return X, y

    df_lagged = df.copy()

    # Generate lag features for each feature
    lagged_features = []
    for feature in feature_names:
        for lag in range( num_lags + 1,0,-1):  # Start from 1 to num_lags
            lag_col = f"{feature}_Lag_{lag}"
            df_lagged[lag_col] = df_lagged[feature].shift(lag)
            lagged_features.append(lag_col)

    # Drop rows with NaN values caused by shifting
    df_lagged.dropna(inplace=True)

    # Define X (features) and y (target)
    X = df_lagged[lagged_features]
    y = df_lagged[target_column]

    return X, y

# Create sample data
np.random.seed(42)  # For reproducibility
num_rows = 20
time_values = np.arange(0, num_rows * 10, 10)  # Time in seconds (incrementing by 10s)
rot_speed_values = np.random.randint(100, 500, size=num_rows)
temperature_values = np.random.uniform(20, 100, size=num_rows)
pressure_values = np.random.uniform(900, 1100, size=num_rows)
A = np.random.uniform(1000, 2100, size=num_rows)

# Create a DataFrame
df_X = pd.DataFrame({
    "Time": time_values,
    "RotSpeed": rot_speed_values,
    "Temperature": temperature_values,
    "Pressure": pressure_values,
    'A': A,
})

# Example usage
num_lags = 1  # Change this to test with or without lags
feature_names = ["RotSpeed", "Temperature", "Pressure"]  # Features to create lags for
target_column = "A"  # Target variable

X, y = generate_lagged_dataset(df_X, feature_names, target_column, num_lags)

# Display results
print("Features (X):")
print(X.head())  # Show first few rows of X

print("\nTarget (y):")
print(y.head())  # Show first few rows of y

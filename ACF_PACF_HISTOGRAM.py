from statsmodels.tsa.stattools import pacf, acf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(__file__), 'generated_data',
                         'new_entries_combined.xlsx')

# Read the Excel file
df = pd.read_excel(file_path)

df_col = df.columns

nlags = 50

for col in df_col:

    # Example time series data
    time_series = df[col].to_numpy()

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(time_series, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of {col}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Scatter plot of y_t vs y_t-1
    yt = time_series[1:]     # Current values (y_t)
    yt_1 = time_series[:-1]  # Lagged values (y_t-1)

    plt.figure(figsize=(8, 5))
    plt.scatter(yt_1, yt, alpha=0.7, edgecolors="black")
    plt.title(f"Scatter Plot of y_t vs y_t-1 for {col}")
    plt.xlabel("y_t-1 (Lagged Value)")
    plt.ylabel("y_t (Current Value)")
    plt.grid()
    plt.show()

    # Calculate the ACF
    acf_values = acf(time_series, nlags=nlags)

    # Plot the ACF
    plt.figure(figsize=(8, 5))
    plt.stem(range(len(acf_values)), acf_values, basefmt=" ")
    plt.title(f"Autocorrelation Function (ACF) - {col}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid()
    plt.show()

    # Calculate the PACF
    pacf_values = pacf(time_series, nlags=nlags, method="yw")

    # Plot the PACF
    plt.figure(figsize=(8, 5))
    plt.stem(range(len(pacf_values)), pacf_values, basefmt=" ")
    plt.title(f"Partial Autocorrelation Function (PACF) - {col}")
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.grid()
    plt.show()

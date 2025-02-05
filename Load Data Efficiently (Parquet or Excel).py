import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# ======================================
# 1. Define Paths
# ======================================
file_path = r"D:_Wind8ms.xlsx"
cache_path = "fast_output.parquet"  # Use Parquet for large datasets (much faster than Excel)

# ======================================
# 2. Load Data Efficiently (Parquet or Excel)
# ======================================
if os.path.exists(cache_path):
    print("Loading data from Parquet cache...")
    df = pd.read_parquet(cache_path)  # Load from cache if available
else:
    print("Reading data from Excel (this may take time)...")
    
    # Read only the needed sheets
    df_sheets = pd.read_excel(file_path, sheet_name=['FAST_Output', 'Debug_Output'], engine="openpyxl")

    # Extract each sheet into a DataFrame
    df_fast = df_sheets['FAST_Output']
    df_debug = df_sheets['Debug_Output']

    # Ensure "Time" is properly formatted in both DataFrames
    df_fast['Time'] = pd.to_numeric(df_fast['Time'], errors='coerce')
    df_debug['Time'] = pd.to_numeric(df_debug['Time'], errors='coerce')

    # Sort both dataframes by time
    df_fast = df_fast.sort_values(by='Time')
    df_debug = df_debug.sort_values(by='Time')

    # Interpolate Debug_Output onto FAST_Output time grid
    df_debug_interp = df_debug.set_index('Time').interpolate(method='linear').reset_index()

    # Merge using nearest matching time values
    df = pd.merge_asof(df_fast, df_debug_interp, on='Time', direction='nearest')

    # Save as Parquet for fast future access
    df.to_parquet(cache_path, engine="pyarrow", compression="snappy")
    print("Excel data cached in Parquet format for faster access next time.")

# Print confirmation
print("Data successfully loaded!")
print(df.head())  # Display first few rows

# Quick Summary of Data
print("\nDataset Info:")
print(df.info())

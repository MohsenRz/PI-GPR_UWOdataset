""" 
making a simple XGBoost model for CSO data prediction for comparison
Model is purely exogenous (only uses time and precipitation)
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from pathlib import Path

## Load Data
BASE = Path(__file__).parent.parent.parent

data_path = BASE / "data" / "RAW_data" / "pickled_data"

CSO = pd.read_pickle(
    data_path / "overflow_to_CSO" / "sensor_bf_plsRKBA1101_rubbasin_ara_2021-01-01_to_2021-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2021_cleaned.pkl")

## Preprocess Data
CSO['timestamp'] = pd.to_datetime(CSO['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters
train_start_time = pd.to_datetime("2021-04-10 00:00:00")
train_days = 30  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

CSO_train = CSO[(CSO['timestamp'] >= train_start_time) & (CSO['timestamp'] <= train_end_time)]
precipitation_train = precipitation[(precipitation['timestamp'] >= train_start_time) & (precipitation['timestamp'] <= train_end_time)]

CSO_test = CSO[(CSO['timestamp'] > train_end_time) & (CSO['timestamp'] <= test_end_time)]
precipitation_test = precipitation[(precipitation['timestamp'] > train_end_time) & (precipitation['timestamp'] <= test_end_time)]

CSO_train.set_index('timestamp', inplace=True)
precipitation_train.set_index('timestamp', inplace=True)
CSO_test.set_index('timestamp', inplace=True)
precipitation_test.set_index('timestamp', inplace=True)

interval_string = f'{timeinterval}min' # resample interval
CSO_train_resampled = CSO_train.resample(interval_string).mean().fillna(0)
precipitation_train_resampled = precipitation_train.resample(interval_string).mean().fillna(0)
CSO_test_resampled = CSO_test.resample(interval_string).mean().fillna(0)
precipitation_test_resampled = precipitation_test.resample(interval_string).mean().fillna(0)

# Merge Data
merged_train = CSO_train_resampled.join(precipitation_train_resampled, 
                                         lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_test = CSO_test_resampled.join(precipitation_test_resampled, 
                                       lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_train.dropna(inplace=True)
merged_test.dropna(inplace=True)

print(f"Training period: {train_start_time} to {train_end_time} ({train_days} days)")
print(f"Test period: {train_end_time} to {test_end_time} ({test_hours} hours)")
print(f"Training samples: {len(merged_train)}")
print(f"Test samples: {len(merged_test)}")

def find_optimal_short_term_rain_lag(merged_data, interval_minutes, max_lag_hours=2, lag_step_minutes=15):
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    inflow = merged_data[inflow_col].values
    precip = merged_data[precip_col].values
    
    lag_range = range(0, max_lag_hours * 60 + 1, lag_step_minutes)
    correlations = {}
    
    for lag_minutes in lag_range:
        window_size = int(lag_minutes / interval_minutes)
        if window_size < 1:
            window_size = 1
        
        rain_accum = pd.Series(precip).rolling(window=window_size).sum().fillna(0).values
        
        valid_idx = ~(np.isnan(inflow) | np.isnan(rain_accum))
        if valid_idx.sum() > 0:
            corr = np.corrcoef(inflow[valid_idx], rain_accum[valid_idx])[0, 1]
            correlations[lag_minutes] = corr
    
    optimal_lag = max(correlations, key=correlations.get)
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    
    return optimal_lag, correlations

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, 
                   short_lag_minutes=120, long_lag_hours=24): 
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    rain_series = merged_data[precip_col]
    
    window_size_short = int(short_lag_minutes / interval_minutes) 
    if window_size_short < 1: window_size_short = 1
    rain_short = rain_series.rolling(window=window_size_short).sum().fillna(0).values.reshape(-1, 1)
    
    window_size_long = int(long_lag_hours * 60 / interval_minutes)
    if window_size_long < 1: window_size_long = 1
    rain_long = rain_series.rolling(window=window_size_long).sum().fillna(0).values.reshape(-1, 1)
    
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
        
    X_multi = np.hstack((rain_short, rain_long))
    Y = inflow
        
    return X_multi, Y, timestamps

def model_evaluation(Y_true, Y_pred):
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    max_true = np.max(Y_true.ravel())
    max_pred = np.max(Y_pred.ravel())
    
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
        "Max Actual (L/s)": max_true,
        "Max Predicted (L/s)": max_pred
    }

def plot_results(timestamps_train, Y_train, Y_pred_train,
                 timestamps_test, Y_test, Y_pred_test):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=15, 
               label='Training Data', alpha=0.5, zorder=3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='XGBoost Fit (Training)', linewidth=2, zorder=4)
    
    # Plot test data
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data (Actual)', alpha=0.7, zorder=3)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='XGBoost Prediction (Test)', linewidth=2, zorder=4)
    
    if len(timestamps_train) > 0:
        ax.axvline(x=timestamps_train[-1], color='black', linestyle='--', 
                   linewidth=1.5, label='Train/Test Split', zorder=5)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('CSO Inflow (L/s)', fontsize=12)
    ax.set_title(f'XGBoost: CSO Prediction (Train: {train_days} days, Test: {test_hours} hours)', 
                 fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    total_start = time.perf_counter()
    print("="*70)
    print("XGBoost Model for CSO Prediction (Exogenous)")
    print("="*70)
    
    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_short_term_rain_lag(merged_train, timeinterval)
    
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, 
                                                        short_lag_minutes=optimal_lag, long_lag_hours=24)
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    print("\nTraining XGBoost model...")
    # Initialize XGBRegressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                         random_state=42, objective='reg:squarederror')
    
    # Train the model
    model.fit(X_train_scaled, Y_train_scaled.ravel())
    
    # --- PREDICTION ON TRAIN ---
    Y_pred_train_sc = model.predict(X_train_scaled).reshape(-1, 1)
    Y_pred_train = scaler_Y.inverse_transform(Y_pred_train_sc)
    
    # --- PREDICTION ON TEST ---
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, 
                                                     short_lag_minutes=optimal_lag, long_lag_hours=24)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    Y_pred_test_sc = model.predict(X_test_scaled).reshape(-1, 1)
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_sc)
    
    # --- COMPARISON TABLE ---
    train_metrics = model_evaluation(Y_train, Y_pred_train)
    test_metrics = model_evaluation(Y_test, Y_pred_test)
    
    results_df = pd.DataFrame({
        'Training Set': train_metrics,
        'Test Set': test_metrics
    })
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(results_df.round(4))
    print("="*50)
    
    total_time = time.perf_counter() - total_start
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print("Prediction complete!")
    
    #--- PLOTTING ---
    plot_results(timestamps_train, Y_train, Y_pred_train,
                 timestamps_test, Y_test, Y_pred_test)

if __name__ == "__main__":
    main()

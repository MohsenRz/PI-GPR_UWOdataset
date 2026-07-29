""" 
making a simple LSTM model for WWTP data prediction for comparison with GPR
IT IS A REAL-TIME PREDICTIVE MODEL, NOT AN EXOGENEOUS MODEL 
Author: Mohsen 
created at: 29/07/2026
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from pathlib import Path

## Load Data
BASE = Path(__file__).parent.parent.parent

data_path = BASE / "data" / "RAW_data" / "pickled_data"

WWTP_inflow = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

## Preprocess Data
WWTP_inflow['timestamp'] = pd.to_datetime(WWTP_inflow['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters
train_start_time = pd.to_datetime("2019-09-30 00:00:00")
train_days = 30  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

WWTP_inflow_train = WWTP_inflow[(WWTP_inflow['timestamp'] >= train_start_time) & (WWTP_inflow['timestamp'] <= train_end_time)]
precipitation_train = precipitation[(precipitation['timestamp'] >= train_start_time) & (precipitation['timestamp'] <= train_end_time)]

WWTP_inflow_test = WWTP_inflow[(WWTP_inflow['timestamp'] > train_end_time) & (WWTP_inflow['timestamp'] <= test_end_time)]
precipitation_test = precipitation[(precipitation['timestamp'] > train_end_time) & (precipitation['timestamp'] <= test_end_time)]

WWTP_inflow_train.set_index('timestamp', inplace=True)
precipitation_train.set_index('timestamp', inplace=True)
WWTP_inflow_test.set_index('timestamp', inplace=True)
precipitation_test.set_index('timestamp', inplace=True)

interval_string = f'{timeinterval}min' # resample interval
WWTP_train_resampled = WWTP_inflow_train.resample(interval_string).mean()
precipitation_train_resampled = precipitation_train.resample(interval_string).sum()
WWTP_test_resampled = WWTP_inflow_test.resample(interval_string).mean()
precipitation_test_resampled = precipitation_test.resample(interval_string).sum()

# Merge Data
merged_train = WWTP_train_resampled.join(precipitation_train_resampled, 
                                         lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_test = WWTP_test_resampled.join(precipitation_test_resampled, 
                                       lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_train.dropna(inplace=True)
merged_test.dropna(inplace=True)

print(f"Training period: {train_start_time} to {train_end_time} ({train_days} days)")
print(f"Test period: {train_end_time} to {test_end_time} ({test_hours} hours)")
print(f"Training samples: {len(merged_train)}")
print(f"Test samples: {len(merged_test)}")

def find_optimal_rain_lag(merged_data, interval_minutes, max_lag_hours=2, lag_step_minutes=15):
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

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, lag_minutes=120,
                   full_rain_series=None): 
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    day_in_minutes = 24 * 60
    time_of_day = np.array([ts.hour * 60 + ts.minute for ts in timestamps])
    sin_time = np.sin(2 * np.pi * time_of_day / day_in_minutes).reshape(-1, 1)
    cos_time = np.cos(2 * np.pi * time_of_day / day_in_minutes).reshape(-1, 1)
    
    window_size = int(lag_minutes / interval_minutes) 
    if window_size < 1: window_size = 1
    
    if full_rain_series is None:
        rain_series = merged_data[precip_col]
        rain_accum = rain_series.rolling(window=window_size).sum().fillna(0)
    else:
        rain_accum_full = full_rain_series.rolling(window=window_size).sum().fillna(0)
        rain_accum = rain_accum_full.loc[timestamps]
    
    rain_accum = rain_accum.values.reshape(-1, 1)
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
        
    X_multi = np.hstack((sin_time, cos_time, rain_accum, inflow))
    Y = inflow
        
    return X_multi, Y, timestamps

def create_sequences(X, Y, timestamps, seq_length):
    """
    Convert the (N, features) input to (N-seq_length, seq_length, features)
    suitable for LSTM input.
    """
    X_seq, Y_seq, ts_seq = [], [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        Y_seq.append(Y[i+seq_length])
        ts_seq.append(timestamps[i+seq_length])
    return np.array(X_seq), np.array(Y_seq), np.array(ts_seq)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def model_evaluation(Y_true, Y_pred):
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
    }

def plot_results(timestamps_train, Y_train, Y_pred_train,
                 timestamps_test, Y_test, Y_pred_test):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=15, 
               label='Training Data', alpha=0.5, zorder=3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='LSTM Fit (Training)', linewidth=2, zorder=4)
    
    # Plot test data
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data (Actual)', alpha=0.7, zorder=3)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='LSTM Prediction (Test)', linewidth=2, zorder=4)
    
    if len(timestamps_train) > 0:
        ax.axvline(x=timestamps_train[-1], color='black', linestyle='--', 
                   linewidth=1.5, label='Train/Test Split', zorder=5)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WWTP Inflow', fontsize=12)
    ax.set_title(f'LSTM: WWTP Inflow Prediction (Train: {train_days} days, Test: {test_hours} hours)', 
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
    print("LSTM Model for WWTP Inflow Prediction")
    print("="*70)
    
    merged_all = pd.concat([merged_train, merged_test]).sort_index()
    merged_all = merged_all[~merged_all.index.duplicated(keep='first')]
    precip_col = [col for col in merged_all.columns if 'precipitation' in col.lower()][0]
    full_rain_series = merged_all[precip_col]

    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_rain_lag(merged_train, timeinterval)
    
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, 
                                                        lag_minutes=optimal_lag, full_rain_series=full_rain_series)
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    seq_length = 4 # 1 hours of historical data (15 min intervals)
    X_train_seq, Y_train_seq, ts_train_seq = create_sequences(X_train_scaled, Y_train_scaled, timestamps_train, seq_length)
    
    print("\nTraining LSTM model...")
    model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    
    # Train the model
    model.fit(X_train_seq, Y_train_seq, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
    
    # --- PREDICTION ON TRAIN ---
    Y_pred_train_sc = model.predict(X_train_seq)
    Y_pred_train = scaler_Y.inverse_transform(Y_pred_train_sc)
    Y_train_actual = scaler_Y.inverse_transform(Y_train_seq)
    
    # --- PREDICTION ON TEST ---
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, 
                                                     lag_minutes=optimal_lag, full_rain_series=full_rain_series)
    X_test_scaled = scaler_X.transform(X_test)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    X_test_seq, Y_test_seq, ts_test_seq = create_sequences(X_test_scaled, Y_test_scaled, timestamps_test, seq_length)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    Y_pred_test_sc = model.predict(X_test_seq)
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_sc)
    Y_test_actual = scaler_Y.inverse_transform(Y_test_seq)
    
    # --- COMPARISON TABLE ---
    train_metrics = model_evaluation(Y_train_actual, Y_pred_train)
    test_metrics = model_evaluation(Y_test_actual, Y_pred_test)
    
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
    plot_results(ts_train_seq, Y_train_actual, Y_pred_train,
                 ts_test_seq, Y_test_actual, Y_pred_test)

if __name__ == "__main__":
    main()

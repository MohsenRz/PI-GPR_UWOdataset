"""
making a Gaussian Process Regression model for CSO prediction
sensor data is used 
this is a naive model that only takes CSO data for predictions 
automatic lag time detection is used to find the optimal lag time for precipitation data in short term
a new attribute is added: long-term accumulated precipitation, to see the effect in filling the upstream tank
Author: Mohsen 
Date: 03/03/2026
updated: 09/04/2026 for getting results for the paper 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from scipy.stats import norm
from pathlib import Path
import tensorflow_probability as tfp
import pickle

# load data
BASE = Path(__file__).parent.parent.parent

data_path = BASE / "data" / "RAW_data" / "pickled_data"

CSO = pd.read_pickle(
    data_path / "overflow_to_CSO" / "sensor_bf_plsRKBA1101_rubbasin_ara_2021-01-01_to_2021-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2021_cleaned.pkl")

# preprocess data
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

CSO_test = CSO[(CSO['timestamp'] >= train_end_time) & (CSO['timestamp'] <= test_end_time)]
precipitation_test = precipitation[(precipitation['timestamp'] >= train_end_time) & (precipitation['timestamp'] <= test_end_time)]

CSO_train = CSO_train.set_index('timestamp')
CSO_test = CSO_test.set_index('timestamp')
precipitation_train = precipitation_train.set_index('timestamp')
precipitation_test = precipitation_test.set_index('timestamp')

interval_string = f'{timeinterval}min' # resample interval
CSO_train_resampled = CSO_train.resample(interval_string).mean().fillna(0)
CSO_test_resampled = CSO_test.resample(interval_string).mean().fillna(0)
precipitation_train_resampled = precipitation_train.resample(interval_string).mean().fillna(0)
precipitation_test_resampled = precipitation_test.resample(interval_string).mean().fillna(0)

# Merge data 
merged_train = CSO_train_resampled.join(precipitation_train_resampled, 
                                         lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_test = CSO_test_resampled.join(precipitation_test_resampled, 
                                       lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_train.dropna(inplace=True)
merged_test.dropna(inplace=True)
average_CSO = CSO['value'].mean()
average_precipitation = precipitation['value'].mean()

print(f"Training period: {train_start_time} to {train_end_time} ({train_days} days)")
print(f"Test period: {train_end_time} to {test_end_time} ({test_hours} hours)")
print(f"Training samples: {len(merged_train)}")
print(f"Test samples: {len(merged_test)}")
print(f"Data statistics: mean daily CSO = {average_CSO*3.6*24:.1f} m3,\
      mean precipitation = {merged_train.filter(like='precipitation').mean().values[0]*24/timeinterval:.2f} mm/day")

def find_optimal_short_term_rain_lag(merged_data, interval_minutes, max_lag_hours=2, lag_step_minutes=15):
    """
    Perform cross-correlation analysis to find optimal precipitation lag.
    
    Returns:
        optimal_lag_minutes: Best lag time in minutes
        correlations: Dictionary of {lag_minutes: correlation_value}
    """
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    inflow = merged_data[inflow_col].values
    precip = merged_data[precip_col].values
    
    # Test different lag windows
    lag_range = range(0, max_lag_hours * 60 + 1, lag_step_minutes)
    correlations = {}
    
    for lag_minutes in lag_range:
        window_size = int(lag_minutes / interval_minutes)
        if window_size < 1:
            window_size = 1
        
        # Rolling sum for this lag window
        rain_accum = pd.Series(precip).rolling(window=window_size).sum().fillna(0).values
        
        # Calculate correlation (excluding NaNs)
        valid_idx = ~(np.isnan(inflow) | np.isnan(rain_accum))
        if valid_idx.sum() > 0:
            corr = np.corrcoef(inflow[valid_idx], rain_accum[valid_idx])[0, 1]
            correlations[lag_minutes] = corr
    
    # Find optimal lag
    optimal_lag = max(correlations, key=correlations.get)
    
    """
    # Plot results
    plt.figure(figsize=(10, 5))
    lags = list(correlations.keys())
    corrs = list(correlations.values())
    plt.plot(lags, corrs, 'b-o', linewidth=2, markersize=4)
    plt.axvline(x=optimal_lag, color='r', linestyle='--', 
                label=f'Optimal lag: {optimal_lag} min')
    plt.xlabel('Precipitation Accumulation Window (minutes)', fontsize=12)
    plt.ylabel('Cross-correlation with CSO Inflow', fontsize=12)
    plt.title('Cross-Correlation Analysis: Rain Lag vs CSO Response', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    """
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    
    return optimal_lag, correlations

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, 
                   short_lag_minutes=120, long_lag_hours=24): 
    #creating multi dimensional input as timestamps and precipitation data
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)  # time in minutes
    
    rain_series = merged_data[precip_col]
    
    ## Feature ONE: Short-term accumulated precipitation
    # Calculate window size dynamically
    window_size_short = int(short_lag_minutes / interval_minutes) 
    if window_size_short < 1: window_size_short = 1
    rain_short = rain_series.rolling(window=window_size_short).sum().fillna(0).values.reshape(-1, 1)
    
    ## Feature TWO: Long-term accumulated precipitation
    window_size_long = int(long_lag_hours * 60 / interval_minutes)
    if window_size_long < 1: window_size_long = 1
    rain_long = rain_series.rolling(window=window_size_long).sum().fillna(0).values.reshape(-1, 1)
    
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
    
    X_multi = np.hstack((time_feat, rain_short, rain_long))
    Y = inflow
        
    return X_multi, Y, timestamps

def build_gpr_model(X_train, Y_train, time_std_dev):
    """
    Build and train GPR model
    X_train: Scaled training data 
    Y_train: Scaled target data
    time_std_dev: the scaling factor (std) of the Time Column from scalara_X.scale_[0]
    """ 
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev  # Adjust period based on scaling
    print(f"Scaled period for daily cycle: {scaled_period}")
    
    ### kernel design 
    time_kernel = gpflow.kernels.RBF(variance=1.0, active_dims=[0])
    bounded_transform_time = tfp.bijectors.Sigmoid(
        low=tf.constant(2.0 * scaled_period, dtype=tf.float64), 
        high=tf.constant(14.0 * scaled_period, dtype=tf.float64))
    time_kernel.lengthscales = gpflow.Parameter(7.0 * scaled_period, transform=bounded_transform_time)
    
    ### rain kernel 
    kernel_rain_short = gpflow.kernels.Matern12(variance=1, active_dims=[1])
    bounded_transform_rain_short = tfp.bijectors.Sigmoid(
        low=tf.constant(0.1, dtype=tf.float64), 
        high=tf.constant(1, dtype=tf.float64)) 
    kernel_rain_short.lengthscales = gpflow.Parameter(0.5, transform=bounded_transform_rain_short)
    
    kernel_rain_long = gpflow.kernels.Matern52(variance=1, active_dims=[2])
    bounded_transform_rain_long = tfp.bijectors.Sigmoid(
        low=tf.constant(0.5, dtype=tf.float64), 
        high=tf.constant(5, dtype=tf.float64)) 
    kernel_rain_long.lengthscales = gpflow.Parameter(2, transform=bounded_transform_rain_long)
    
    kernel = time_kernel + kernel_rain_short + kernel_rain_long 
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    model = gpflow.models.GPR(data=(X_train_tf, Y_train_tf), 
                              kernel=kernel, mean_function=None)
    
    # We force the model to maintain a minimum baseline noise so the bounds don't disappear
    #bounded_transform_noise = tfp.bijectors.Sigmoid(
    #    low=tf.constant(0.05, dtype=tf.float64),  # Lower bound for noise variance
    #    high=tf.constant(1.0, dtype=tf.float64))
    #model.likelihood.variance = gpflow.Parameter(0.1, transform=bounded_transform_noise)
    
    # optimisation 
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 method='L-BFGS-B')
    print("\nModel Summary:")
    gpflow.utilities.print_summary(model)
    
    return model

def model_evaluation(Y_true, Y_pred, std_pred):
    """
    Returns a dictionary of metrics for easy table formatting.
    """
    # Standard Metrics
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    # Coverage
    lower_bound = Y_pred.ravel() - 1.96 * std_pred.ravel()
    upper_bound = Y_pred.ravel() + 1.96 * std_pred.ravel()
    points_inside = np.sum((Y_true.ravel() >= lower_bound) & 
                           (Y_true.ravel() <= upper_bound))
    coverage = points_inside / len(Y_true)
    
    # Entropy (Nats)
    variance = std_pred.ravel() ** 2
    # Use log2 for Bits
    entropy_per_point = 0.5 * np.log2(2 * np.pi * np.e * variance)
    mean_entropy = np.mean(entropy_per_point)
    
    max_true = np.max(Y_true.ravel())
    max_pred = np.max(Y_pred.ravel())
    
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
        "Coverage (%)": coverage * 100,
        "Entropy (nats)": mean_entropy,
        "Max Actual (L/s)": max_true,
        "Max Predicted (L/s)": max_pred
    }
    
def plot_results(timestamps_train, Y_train, Y_pred_train, std_train,
                 timestamps_test, Y_test, Y_pred_test, std_test):
    """
    Plot training and test results with uncertainty.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # plotting training data 
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=15, 
               label='Training Data', alpha=0.5, zorder=3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='GPR Fit (Training)', linewidth=2, zorder=4)
    ax.fill_between(timestamps_train, 
                    Y_pred_train.ravel() - 1.96 * std_train.ravel(),
                    Y_pred_train.ravel() + 1.96 * std_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Training)', zorder=2)
    
    # plotting test data
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data (Actual)', alpha=0.7, zorder=3)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='GPR Prediction (Test)', linewidth=2, zorder=4)
    ax.fill_between(timestamps_test,
                    Y_pred_test.ravel() - 1.96 * std_test.ravel(),
                    Y_pred_test.ravel() + 1.96 * std_test.ravel(),
                    alpha=0.2, color='red', label='95% CI (Test)', zorder=2)
    
    # Add vertical line separating train/test
    ax.axvline(x=timestamps_train[-1], color='black', linestyle='--', 
               linewidth=1.5, label='Train/Test Split', zorder=5)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('CSO flow (L/s)', fontsize=12)
    ax.set_title(f'GPR: CSO Prediction (Train: {train_days} days, Test: {test_hours} hours)', 
                 fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def main():
    total_start = time.perf_counter()
    print("="*70 + "\nNaive CSO GPR Prediction\n" + "="*70)
    
    # Find optimal lag
    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_short_term_rain_lag(merged_train, timeinterval)

    # Prepare training data with optimal lag
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, 
                                                        short_lag_minutes=optimal_lag, long_lag_hours=24)
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    print("\nTraining GPR model...")
    # Pass Time Std Dev for Period Calculation
    model = build_gpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0])
    
    # --- PREDICTION ON TRAIN ---
    X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float64)
    mean_train_sc, var_train_y_sc = model.predict_y(X_train_tf)
    _, var_train_f_sc = model.predict_f(X_train_tf)
    
    Y_pred_train = scaler_Y.inverse_transform(mean_train_sc.numpy())
    std_train_y = np.sqrt(var_train_y_sc.numpy()) * scaler_Y.scale_
    std_train_f = np.sqrt(var_train_f_sc.numpy()) * scaler_Y.scale_
    
    # --- PREDICTION ON TEST ---
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, 
                                                     short_lag_minutes=optimal_lag, long_lag_hours=24)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float64)
    
    mean_test_sc, var_test_y_sc = model.predict_y(X_test_tf)
    _, var_test_f_sc = model.predict_f(X_test_tf)
    
    Y_pred_test = scaler_Y.inverse_transform(mean_test_sc.numpy())
    std_test_y = np.sqrt(var_test_y_sc.numpy()) * scaler_Y.scale_
    std_test_f = np.sqrt(var_test_f_sc.numpy()) * scaler_Y.scale_
    
    # --- COMPARISON TABLE ---
    train_metrics = model_evaluation(Y_train, Y_pred_train, std_train_y)
    test_metrics = model_evaluation(Y_test, Y_pred_test, std_test_y)
    
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
    plot_results(timestamps_train, Y_train, Y_pred_train, std_train_y,
                 timestamps_test, Y_test, Y_pred_test, std_test_y)


    # saving figures for a later use 
    results_data = {
        'train': {
            'time': timestamps_train,
            'actual': Y_train.ravel(),
            'pred': Y_pred_train.ravel(),
            'std': std_train_y.ravel()  
        },
        'test': {
            'time': timestamps_test,
            'actual': Y_test.ravel(),
            'pred': Y_pred_test.ravel(),
            'std': std_test_y.ravel()
        },
        'metadata': {
            'train_days': train_days,
            'test_hours': test_hours
        }
    }
    save_path = BASE / "results" / "CSO_outputs" / "2021_CSO_GPR_kernel.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Data successfully saved to: {save_path}")
    
if __name__ == "__main__":
    main()


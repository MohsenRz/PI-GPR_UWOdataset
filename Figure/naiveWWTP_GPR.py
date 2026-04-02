""" 
making a Gaussian Process Regression model for WWTP data prediction
naive model with no constraints on the kernel hyperparameters, only using time and precipitation as input features + kernel fixed model 
sensor data is used 
Author: Mohsen 
created at: 27/11/2025
updated at: 01/04/2026
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from scipy.stats import norm
from pathlib import Path
import tensorflow_probability as tfp

# Set font type globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

## Load Data
BASE = Path(__file__).parent.parent

data_path = BASE / "RAW_data" / "pickled_data"

WWTP_inflow = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2021-01-01_to_2021-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2021_cleaned.pkl")

## Preprocess Data
WWTP_inflow['timestamp'] = pd.to_datetime(WWTP_inflow['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters
train_start_time = pd.to_datetime("2021-04-10 00:00:00")
train_days = 30  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

WWTP_inflow_train = WWTP_inflow[(WWTP_inflow['timestamp'] >= train_start_time) & (WWTP_inflow['timestamp'] <= train_end_time)]
precipitation_train = precipitation[(precipitation['timestamp'] >= train_start_time) & (precipitation['timestamp'] <= train_end_time)]

WWTP_inflow_test = WWTP_inflow[(WWTP_inflow['timestamp'] >= train_end_time) & (WWTP_inflow['timestamp'] <= test_end_time)]
precipitation_test = precipitation[(precipitation['timestamp'] >= train_end_time) & (precipitation['timestamp'] <= test_end_time)]

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
print(f"Data statistics: mean inflow = {merged_train.filter(like='inflow').mean().values[0]:.2f} L/s,\
      mean precipitation = {merged_train.filter(like='precipitation').mean().values[0]*24/timeinterval:.2f} mm/day")

def find_optimal_rain_lag(merged_data, interval_minutes, max_lag_hours=2, lag_step_minutes=15):
    """
    Perform cross-correlation analysis to find optimal precipitation lag.
    for the WWTP inflow, maximum 1 hour lag can be expected 
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
    
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    
    return optimal_lag, correlations

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, lag_minutes=120): 
    #creating multi dimensional input as timestamps and precipitation data
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)  # time in minutes
    ## I sum up the previous 60 mins precipitation to consider lag effect
    # Calculate window size dynamically (Target 60 mins / Interval)
    window_size = int(lag_minutes / interval_minutes) 
    if window_size < 1: window_size = 1
    
    rain_series = merged_data[precip_col]
    # rolling sum, fill NaN at start with 0
    rain_accum = rain_series.rolling(window=window_size).sum().fillna(0).values.reshape(-1, 1)
    
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
    
    X_multi = np.hstack((time_feat, rain_accum))
    Y = inflow
        
    return X_multi, Y, timestamps

def build_gpr_model(X_train, Y_train, time_std_dev, kernel_type='naive'):
    """
    Build and train GPR model
    X_train: Scaled training data 
    Y_train: Scaled target data
    time_std_dev: the scaling factor (std) of the Time Column from scalara_X.scale_[0]
    kernel_type: 'naive' for simple SquaredExponential, 'designed' for complex kernel
    """ 
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev  # Adjust period based on scaling
    print(f"Scaled period for daily cycle: {scaled_period}")
    
    if kernel_type == 'naive':
        # Naive kernel: simple SquaredExponential
        kernel = gpflow.kernels.SquaredExponential()
        print(f"Using NAIVE kernel: SquaredExponential")
    else:
        # Designed kernel: Periodic + RBF + Matern12
        ### daily kernel 
        kernel_daily = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[0], variance=1), 
                                                period=scaled_period)   # daily periodicity
        bounded_transform_daily = tfp.bijectors.Sigmoid(
            low=tf.constant(scaled_period/(24*60), dtype=tf.float64), # at least one minute
            high=tf.constant(scaled_period/3, dtype=tf.float64))    # at most 8 hours 
        kernel_daily.base_kernel.lengthscales = gpflow.Parameter(0.01, transform=bounded_transform_daily)
        ### trend kernel 
        kernel_long_term = gpflow.kernels.RBF(variance=1.0, active_dims=[0])
        bounded_transform_long_term = tfp.bijectors.Sigmoid(
            low=tf.constant(4.0 * scaled_period, dtype=tf.float64),
            high=tf.constant(8.0 * scaled_period, dtype=tf.float64))
        kernel_long_term.lengthscales = gpflow.Parameter(7.0 * scaled_period, transform=bounded_transform_long_term)
        ### rain kernel 
        kernel_rain = gpflow.kernels.Matern12(variance=1, active_dims=[1])
        bounded_transform_rain = tfp.bijectors.Sigmoid(
            low=tf.constant(0.1, dtype=tf.float64), 
            high=tf.constant(1, dtype=tf.float64)) 
        kernel_rain.lengthscales = gpflow.Parameter(0.5, transform=bounded_transform_rain)
        
        kernel_daily.active_dims = [0]
        kernel_long_term.active_dims = [0]
        kernel_rain.active_dims = [1]   
        
        gpflow.set_trainable(kernel_daily.period, False)
        kernel = kernel_daily * kernel_long_term + kernel_rain
        print(f"Using DESIGNED kernel: Periodic(SquaredExponential) * RBF + Matern12")
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    model = gpflow.models.GPR(data=(X_train_tf, Y_train_tf), 
                              kernel=kernel, mean_function=None)
    #optimise hyperparameters
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, 
                 model.trainable_variables, 
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
    
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
        "Coverage (%)": coverage * 100,
        "Entropy (nats)": mean_entropy
    }

def plot_results(timestamps_train, Y_train, Y_pred_train, std_train,
                 timestamps_test, Y_test, Y_pred_test, std_test):
    """
    Plot training and test results
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=15, 
               label='Training Data', alpha=0.5, zorder=3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='GPR Fit (Training)', linewidth=2, zorder=4)
    ax.fill_between(timestamps_train, 
                    Y_pred_train.ravel() - 1.96 * std_train.ravel(),
                    Y_pred_train.ravel() + 1.96 * std_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Training)', zorder=2)
    
    # Plot test data
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
    ax.set_ylabel('WWTP Inflow', fontsize=12)
    ax.set_title(f'GPR: WWTP Inflow Prediction (Train: {train_days} days, Test: {test_hours} hours)', 
                 fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_comparison(timestamps_train_naive, Y_train, Y_pred_train_naive, std_train_naive,
                    timestamps_test_naive, Y_test, Y_pred_test_naive, std_test_naive,
                    Y_pred_train_designed, std_train_designed,
                    Y_pred_test_designed, std_test_designed):
    """
    Plot side-by-side comparison of naive vs designed kernel
    """
    fig, (ax_naive, ax_designed) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============== NAIVE KERNEL ==============
    ax_naive.scatter(timestamps_train_naive, Y_train.ravel(), c='blue', s=15, 
                     label='Training Data', alpha=0.5, zorder=3)
    ax_naive.plot(timestamps_train_naive, Y_pred_train_naive.ravel(), 'green', 
                  label='GPR Fit (Training)', linewidth=2, zorder=4)
    ax_naive.fill_between(timestamps_train_naive, 
                          Y_pred_train_naive.ravel() - 1.96 * std_train_naive.ravel(),
                          Y_pred_train_naive.ravel() + 1.96 * std_train_naive.ravel(),
                          alpha=0.2, color='green', label='95% CI (Training)', zorder=2)
    
    ax_naive.scatter(timestamps_test_naive, Y_test.ravel(), c='orange', s=15,
                     label='Test Data (Actual)', alpha=0.7, zorder=3)
    ax_naive.plot(timestamps_test_naive, Y_pred_test_naive.ravel(), 'red', 
                  label='GPR Prediction (Test)', linewidth=2, zorder=4)
    ax_naive.fill_between(timestamps_test_naive,
                          Y_pred_test_naive.ravel() - 1.96 * std_test_naive.ravel(),
                          Y_pred_test_naive.ravel() + 1.96 * std_test_naive.ravel(),
                          alpha=0.2, color='red', label='95% CI (Test)', zorder=2)
    
    ax_naive.axvline(x=timestamps_train_naive[-1], color='black', linestyle='--', 
                     linewidth=1.5, label='Train/Test Split', zorder=5)
    
    ax_naive.set_xlabel('Date', fontsize=12, fontfamily='Times New Roman')
    ax_naive.set_ylabel('WWTP Inflow (L/s)', fontsize=12, fontfamily='Times New Roman')
    ax_naive.set_title('Naive Kernel (SquaredExponential)', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    ax_naive.legend(fontsize=10, loc='best', prop={'family': 'Times New Roman'})
    ax_naive.grid(True, alpha=0.3)
    ax_naive.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_naive.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_naive.tick_params(axis='x', rotation=45)
    
    # ============== DESIGNED KERNEL ==============
    ax_designed.scatter(timestamps_train_naive, Y_train.ravel(), c='blue', s=15, 
                        label='Training Data', alpha=0.5, zorder=3)
    ax_designed.plot(timestamps_train_naive, Y_pred_train_designed.ravel(), 'green', 
                     label='GPR Fit (Training)', linewidth=2, zorder=4)
    ax_designed.fill_between(timestamps_train_naive, 
                             Y_pred_train_designed.ravel() - 1.96 * std_train_designed.ravel(),
                             Y_pred_train_designed.ravel() + 1.96 * std_train_designed.ravel(),
                             alpha=0.2, color='green', label='95% CI (Training)', zorder=2)
    
    ax_designed.scatter(timestamps_test_naive, Y_test.ravel(), c='orange', s=15,
                        label='Test Data (Actual)', alpha=0.7, zorder=3)
    ax_designed.plot(timestamps_test_naive, Y_pred_test_designed.ravel(), 'red', 
                     label='GPR Prediction (Test)', linewidth=2, zorder=4)
    ax_designed.fill_between(timestamps_test_naive,
                             Y_pred_test_designed.ravel() - 1.96 * std_test_designed.ravel(),
                             Y_pred_test_designed.ravel() + 1.96 * std_test_designed.ravel(),
                             alpha=0.2, color='red', label='95% CI (Test)', zorder=2)
    
    ax_designed.axvline(x=timestamps_train_naive[-1], color='black', linestyle='--', 
                        linewidth=1.5, label='Train/Test Split', zorder=5)
    
    ax_designed.set_xlabel('Date', fontsize=12, fontfamily='Times New Roman')
    ax_designed.set_ylabel('WWTP Inflow (L/s)', fontsize=12, fontfamily='Times New Roman')
    ax_designed.set_title('Designed Kernel (Periodic*RBF + Matern12)', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    ax_designed.legend(fontsize=10, loc='best', prop={'family': 'Times New Roman'})
    ax_designed.grid(True, alpha=0.3)
    ax_designed.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_designed.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_designed.tick_params(axis='x', rotation=45)
    
    fig.suptitle(f'GPR Model Comparison: Naive vs Designed Kernel (Train: {train_days} days, Test: {test_hours} hours)', 
                 fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    plt.tight_layout()
    
    # Save the comparison figure
    output_path = Path(__file__).parent / "WWTP_GPR_Kernel_Comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison figure saved to: {output_path}")
    
    plt.show()
    
def main():
    total_start = time.perf_counter()
    print("="*70)
    print("GPR Model for WWTP Inflow Prediction - Kernel Comparison")
    print("="*70)
    
    # Find optimal lag
    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_rain_lag(merged_train, timeinterval)
    
    # Prepare training data
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, lag_minutes=optimal_lag)
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, lag_minutes=optimal_lag)
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # ============== NAIVE KERNEL ==============
    print("\n" + "="*70)
    print("TRAINING NAIVE KERNEL MODEL")
    print("="*70)
    model_naive = build_gpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0], kernel_type='naive')
    
    # Predictions with naive kernel
    X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float64)
    X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float64)
    
    mean_train_naive_sc, var_train_naive_sc = model_naive.predict_y(X_train_tf)
    Y_pred_train_naive = scaler_Y.inverse_transform(mean_train_naive_sc.numpy())
    std_train_naive = np.sqrt(var_train_naive_sc.numpy()) * scaler_Y.scale_
    
    mean_test_naive_sc, var_test_naive_sc = model_naive.predict_y(X_test_tf)
    Y_pred_test_naive = scaler_Y.inverse_transform(mean_test_naive_sc.numpy())
    std_test_naive = np.sqrt(var_test_naive_sc.numpy()) * scaler_Y.scale_
    
    # Evaluation for naive kernel
    train_metrics_naive = model_evaluation(Y_train, Y_pred_train_naive, std_train_naive)
    test_metrics_naive = model_evaluation(Y_test, Y_pred_test_naive, std_test_naive)
    
    print("\n" + "="*50)
    print("NAIVE KERNEL - MODEL PERFORMANCE")
    print("="*50)
    results_naive = pd.DataFrame({
        'Training Set': train_metrics_naive,
        'Test Set': test_metrics_naive
    })
    print(results_naive.round(4))
    print("="*50)
    
    # ============== DESIGNED KERNEL ==============
    print("\n" + "="*70)
    print("TRAINING DESIGNED KERNEL MODEL")
    print("="*70)
    model_designed = build_gpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0], kernel_type='designed')
    
    # Predictions with designed kernel
    mean_train_designed_sc, var_train_designed_sc = model_designed.predict_y(X_train_tf)
    Y_pred_train_designed = scaler_Y.inverse_transform(mean_train_designed_sc.numpy())
    std_train_designed = np.sqrt(var_train_designed_sc.numpy()) * scaler_Y.scale_
    
    mean_test_designed_sc, var_test_designed_sc = model_designed.predict_y(X_test_tf)
    Y_pred_test_designed = scaler_Y.inverse_transform(mean_test_designed_sc.numpy())
    std_test_designed = np.sqrt(var_test_designed_sc.numpy()) * scaler_Y.scale_
    
    # Evaluation for designed kernel
    train_metrics_designed = model_evaluation(Y_train, Y_pred_train_designed, std_train_designed)
    test_metrics_designed = model_evaluation(Y_test, Y_pred_test_designed, std_test_designed)
    
    print("\n" + "="*50)
    print("DESIGNED KERNEL - MODEL PERFORMANCE")
    print("="*50)
    results_designed = pd.DataFrame({
        'Training Set': train_metrics_designed,
        'Test Set': test_metrics_designed
    })
    print(results_designed.round(4))
    print("="*50)
    
    # ============== COMPARISON TABLE ==============
    print("\n" + "="*70)
    print("COMPARISON: NAIVE vs DESIGNED KERNEL")
    print("="*70)
    comparison_df = pd.DataFrame({
        'Naive Kernel (Test)': test_metrics_naive,
        'Designed Kernel (Test)': test_metrics_designed,
        'Difference': {k: test_metrics_designed[k] - test_metrics_naive[k] for k in test_metrics_naive}
    })
    print(comparison_df.round(4))
    print("="*70)
    
    total_time = time.perf_counter() - total_start
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    
    # ============== SIDE-BY-SIDE PLOT ==============
    print("\nGenerating side-by-side comparison plot...")
    plot_comparison(timestamps_train, Y_train, Y_pred_train_naive, std_train_naive,
                    timestamps_test, Y_test, Y_pred_test_naive, std_test_naive,
                    Y_pred_train_designed, std_train_designed,
                    Y_pred_test_designed, std_test_designed)
    
    

if __name__ == "__main__":
    main()


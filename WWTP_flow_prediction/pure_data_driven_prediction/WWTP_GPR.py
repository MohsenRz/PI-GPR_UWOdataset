""" 
making a Gaussian Process Regression model for WWTP data prediction
sensor data is used 
poitn of interest is added to the plot which shows the full distribution of the prediction for that specific timestamp and compares it with the actual value.
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
from scipy.stats import truncnorm, norm 
from pathlib import Path
import tensorflow_probability as tfp

## Load Data
BASE = Path(__file__).parent.parent.parent

data_path = BASE / "data" / "RAW_data" / "pickled_data"

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
    """
    # Plot results
    plt.figure(figsize=(10, 5))
    lags = list(correlations.keys())
    corrs = list(correlations.values())
    plt.plot(lags, corrs, 'b-o', linewidth=2, markersize=4)
    plt.axvline(x=optimal_lag, color='r', linestyle='--', 
                label=f'Optimal lag: {optimal_lag} min')
    plt.xlabel('Precipitation Accumulation Window (minutes)', fontsize=12)
    plt.ylabel('Cross-correlation with WWTP Inflow', fontsize=12)
    plt.title('Cross-Correlation Analysis: Rain Lag vs WWTP Response', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    
    return optimal_lag, correlations

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, lag_minutes=120,
                   full_rain_series=None): 
    #creating multi dimensional input as timestamps and precipitation data
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)  # time in minutes
    ## I sum up the previous 60 mins precipitation to consider lag effect
    # Calculate window size dynamically (Target 60 mins / Interval)
    window_size = int(lag_minutes / interval_minutes) 
    if window_size < 1: window_size = 1
    
    # Use external full rain series if provided
    if full_rain_series is None:
        rain_series = merged_data[precip_col]
        rain_accum = rain_series.rolling(window=window_size).sum().fillna(0)
    else:
        rain_accum_full = full_rain_series.rolling(window=window_size).sum().fillna(0)
        rain_accum = rain_accum_full.loc[timestamps]
    
    rain_accum = rain_accum.values.reshape(-1, 1)
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
        
    X_multi = np.hstack((time_feat, rain_accum))
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
    ### noise kernel
    kernel_noise = gpflow.kernels.White()
    
    kernel_daily.active_dims = [0]
    kernel_long_term.active_dims = [0]
    kernel_rain.active_dims = [1]   
    
    gpflow.set_trainable(kernel_daily.period, False)
    
    kernel = kernel_daily * kernel_long_term + kernel_rain  # kernel noise is removed because the likelihhod variance takes the overal noise
    
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
                 timestamps_test, Y_test, Y_pred_test, std_test,
                 min_inflow=None, max_inflow=None, sample_date=None):
    """
    Plot training and test results
    shows the bounds of 95% confidence interval and the actual data points for both train and test sets.
    If sample_date is provided, it will also plot the PDF for that specific timestamp.
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
    
    # vertical line showing the sample date
    if sample_date is not None:
        sample_ts = pd.to_datetime(sample_date)
        ax.axvline(x=sample_ts, color='purple', linestyle=':', 
                   linewidth=1.5, label='Point of Interest', zorder=5)
    
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
    
def plot_point_of_interest(target_date_str, timestamps, X_scaled, model, 
                           scaler_Y, Y_actual, min_val=None, max_val=None):
    """
    Plots the full probability distribution (PDF) for a specific timestamp.
    Visualizes the difference between Raw Mean, Truncated Mean, and Mode.
    """
    target_ts = pd.to_datetime(target_date_str)
    time_diffs = np.abs(timestamps - target_ts)
    idx = np.argmin(time_diffs)
    actual_ts = timestamps[idx]
    actual_value = Y_actual[idx][0]
    
    print(f"Plotting Point: {actual_ts}")
    
    # Getting Raw Parameters
    x_input = X_scaled[idx].reshape(1, -1)
    X_tf = tf.convert_to_tensor(x_input, dtype=tf.float64)
    mean_sc, var_sc = model.predict_y(X_tf)
    
    # Unscale
    mu_raw = scaler_Y.inverse_transform(mean_sc.numpy())[0][0]
    sigma_raw = (np.sqrt(var_sc.numpy()) * scaler_Y.scale_)[0][0]
    
    # Truncation Bounds (Z-scores)
    a, b = -np.inf, np.inf
    if min_val is not None: a = (min_val - mu_raw) / sigma_raw
    if max_val is not None: b = (max_val - mu_raw) / sigma_raw
    
    # MEAN (Center of Mass) - This is what your prediction() function returns
    mu_truncated = truncnorm.mean(a, b, loc=mu_raw, scale=sigma_raw)
       
    # MODE (Highest Peak) - Visually where the curve is highest
    if mu_raw < (min_val if min_val else -np.inf):
        mode_truncated = min_val
    elif mu_raw > (max_val if max_val else np.inf):
        mode_truncated = max_val
    else:
        mode_truncated = mu_raw
 
    # x-axis range
    x_min = mu_raw - 4*sigma_raw
    if min_val is not None: x_min = min(x_min, min_val - 10)
    x_max = mu_raw + 4*sigma_raw
    if max_val is not None: x_max = max(x_max, max_val + 10)
    
    x_axis = np.linspace(x_min, x_max, 1000)
    y_pdf = truncnorm.pdf(x_axis, a, b, loc=mu_raw, scale=sigma_raw)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting Distribution
    ax.plot(x_axis, y_pdf, 'b-', lw=2, label='Probability Density')
    ax.fill_between(x_axis, y_pdf, alpha=0.1, color='blue')
    
    # A. Raw Mean (Where the bell curve WANTS to be)
    #ax.axvline(mu_raw, color='red', linestyle=':', linewidth=2, 
    #           label=f'Raw Mean ({mu_raw:.1f})')
    
    # B. Truncated Mean (Prediction - Center of Mass)
    ax.axvline(mu_truncated, color='green', linestyle='-', linewidth=2, 
               label=f'Prediction ({mu_truncated:.1f})')
    
    # --- ADDED: Plotting the Actual Value ---
    ax.axvline(actual_value, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Actual Value ({actual_value:.1f})')
    
    # Plot Constraints
    if min_val is not None:
        ax.axvline(min_val, color='k', linewidth=3, label='Min Constraint')
    if max_val is not None:
        ax.axvline(max_val, color='k', linewidth=3, label='Max Constraint')

    ax.set_title(f"Prediction Distribution at {actual_ts}", fontsize=14)
    ax.set_xlabel("Inflow (L/s)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    total_start = time.perf_counter()
    print("="*70)
    print("GPR Model for WWTP Inflow Prediction")
    print("="*70)
    
    merged_all = pd.concat([merged_train, merged_test]).sort_index()
    merged_all = merged_all[~merged_all.index.duplicated(keep='first')]
    precip_col = [col for col in merged_all.columns if 'precipitation' in col.lower()][0]
    full_rain_series = merged_all[precip_col]

    # Find optimal lag
    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_rain_lag(merged_train, timeinterval)
    
    # Prepare training data (Pass interval to calc window size)
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, 
                                                        lag_minutes=optimal_lag, full_rain_series=full_rain_series)
    
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
                                                     lag_minutes=optimal_lag, full_rain_series=full_rain_series)
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
    sample_date = str(train_end_time + pd.Timedelta(hours=36))  # it shows the 36th hour of the predictions
    plot_results(timestamps_train, Y_train, Y_pred_train, std_train_y,
                 timestamps_test, Y_test, Y_pred_test, std_test_y, 
                 min_inflow=None, max_inflow=None, sample_date=sample_date)
    
    plot_point_of_interest(sample_date, timestamps_test,
                           X_test_scaled, model, scaler_Y,
                           Y_test, min_val=None, max_val=None)

if __name__ == "__main__":
    main()


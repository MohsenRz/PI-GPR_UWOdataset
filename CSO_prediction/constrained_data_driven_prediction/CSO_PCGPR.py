"""
making a Gaussian Process Regression model for CSO prediction
sensor data is used 
this is a naive model that only takes CSO data for predictions 
automatic lag time detection is used to find the optimal lag time for precipitation data in short term
a new attribute is added: long-term accumulated precipitation, to see the effect in filling the upstream tank
mean function and min/max values have been added to the model 
Author: Mohsen 
Date: 06/03/2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from scipy.stats import truncnorm
from pathlib import Path
import tensorflow_probability as tfp

# load data
BASE = Path(__file__).parent.parent.parent

data_path = BASE / "data" / "RAW_data" / "pickled_data"

CSO = pd.read_pickle(
    data_path / "overflow_to_CSO" / "sensor_bf_plsRKBA1101_rubbasin_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

# preprocess data
CSO['timestamp'] = pd.to_datetime(CSO['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters 
train_start_time = pd.to_datetime("2019-06-01 00:00:00")
train_days = 30  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

# constraints 
mean_value = 0 # L/s
min_flow = 0 # L/s
max_flow = None # No upper limit 

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
    plt.show()
    """
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    print(f"Max correlation: {correlations[optimal_lag]:.3f}")
    
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

def build_gpr_model(X_train, Y_train, time_std_dev, scaler_Y=None):
    """
    Build and train GPR model with constraints 
    X_train: Scaled training data 
    Y_train: Scaled target data
    time_std_dev: the scaling factor (std) of the Time Column from scalara_X.scale_[0]
    """ 
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev  # Adjust period based on scaling
    print(f"Scaled period for daily cycle: {scaled_period}")
    
    ### kernel design 
    time_kernel = gpflow.kernels.RBF(lengthscales=scaled_period) 
    
    ### rain kernel 
    kernel_rain_short = gpflow.kernels.Matern12(variance=1, active_dims=[1])
    bounded_transform_rain_short = tfp.bijectors.Sigmoid(
        low=tf.constant(0.1, dtype=tf.float64), 
        high=tf.constant(1, dtype=tf.float64)) 
    kernel_rain_short.lengthscales = gpflow.Parameter(0.5, transform=bounded_transform_rain_short)
    
    kernel_rain_long = gpflow.kernels.Matern52(variance=1, active_dims=[2])
    bounded_transform_rain_long = tfp.bijectors.Sigmoid(
        low=tf.constant(1, dtype=tf.float64), 
        high=tf.constant(5, dtype=tf.float64)) 
    kernel_rain_long.lengthscales = gpflow.Parameter(3, transform=bounded_transform_rain_long)
    
    kernel = kernel_rain_short +kernel_rain_long + time_kernel
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    # Scale mean_value to match Y_train scaling
    if scaler_Y is not None:
        mean_scaled = scaler_Y.transform([[mean_value]])[0, 0]
        print(f"Mean flow: {mean_value} L/s (scaled: {mean_scaled:.4f})")
    else:
        mean_scaled = mean_value
    
    model = gpflow.models.GPR(data=(X_train_tf, Y_train_tf), 
                              kernel=kernel, 
                              mean_function=gpflow.mean_functions.Constant(mean_scaled))
    
    # optimisation 
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 method='L-BFGS-B')
    print("\nModel Summary:")
    gpflow.utilities.print_summary(model)
    
    return model

def prediction(model, X_scaled, scaler_Y, min_val=None, max_val=None):
    """
    Make predictions with the GPR model
    min and max constraints will be applied here with TRUNCATED Gaussian distribution 
    """
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float64)
    mean_sc, var_y_sc = model.predict_y(X_tf)
    _, var_f_sc = model.predict_f(X_tf)
    
    Y_pred = scaler_Y.inverse_transform(mean_sc.numpy())
    std_y = np.sqrt(var_y_sc.numpy()) * scaler_Y.scale_
    std_f = np.sqrt(var_f_sc.numpy()) * scaler_Y.scale_
    
    if min_val is None and max_val is None:
        return Y_pred, std_y, std_f
    
    # Flatten arrays for scipy truncnorm (expects 1D arrays)
    Y_pred_flat = Y_pred.ravel()
    std_y_flat = std_y.ravel()
    
    # I use truncated Gaussian distribution for the test predictions only 
    if min_val is not None:
        a = (min_val - Y_pred_flat) / std_y_flat
    else: 
        a = -np.inf * np.ones_like(Y_pred_flat)
    
    if max_val is not None:
        b = (max_val - Y_pred_flat) / std_y_flat
    else:
        b = np.inf * np.ones_like(Y_pred_flat)
    
    Y_pred_constrained = truncnorm.mean(a=a, b=b, loc=Y_pred_flat, scale=std_y_flat)
    std_y_constrained = truncnorm.std(a=a, b=b, loc=Y_pred_flat, scale=std_y_flat)
    
    # Reshape back to original shape
    Y_pred_constrained = Y_pred_constrained.reshape(Y_pred.shape)
    std_y_constrained = std_y_constrained.reshape(std_y.shape)
        
    return Y_pred_constrained, std_y_constrained, std_f

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
                 timestamps_test, Y_test, Y_pred_test, std_test, 
                 min_level=None, max_level=None, sample_date=None):
    """
    Plot training and test results with uncertainty.
    Credible Intervals are clipped 
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # plotting training data 
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=10, 
               label='Training Data', alpha=0.3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='GPR Fit (Training)', linewidth=1.5)
    ax.fill_between(timestamps_train, 
                    Y_pred_train.ravel() - 1.96 * std_train.ravel(),
                    Y_pred_train.ravel() + 1.96 * std_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Training)')
    
    # Plot test data
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data (Actual)', alpha=0.6)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='GPR Prediction (Test)', linewidth=1.5)
    lower_test = Y_pred_test.ravel() - 1.96 * std_test.ravel()
    upper_test = Y_pred_test.ravel() + 1.96 * std_test.ravel()
    if min_level is not None:
        lower_test = np.maximum(lower_test, min_level)
    if max_level is not None:
        upper_test = np.minimum(upper_test, max_level)
    
    ax.fill_between(timestamps_test, lower_test, upper_test,
                    alpha=0.2, color='red', label='95% CI (Test)')
    
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
    ax.set_ylabel('Tank Level (mm)', fontsize=12)
    ax.set_title(f'GPR: Tank Level Prediction (Train: {train_days} days, Test: {test_hours} hours)', 
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
                           scaler_Y, Y_actual=None, min_val=None, max_val=None):
    """
    Plots the full probability distribution (PDF) for a specific timestamp.
    Visualizes the difference between Raw Mean, Truncated Mean, and Mode.
    Also shows the actual sensor value and its position in the distribution.
    """
    target_ts = pd.to_datetime(target_date_str)
    time_diffs = np.abs(timestamps - target_ts)
    idx = np.argmin(time_diffs)
    actual_ts = timestamps[idx]

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
    ax.plot(x_axis, y_pdf, 'b-', lw=2, label='Probability Density')
    ax.fill_between(x_axis, y_pdf, alpha=0.1, color='blue')
    
    # A. Raw Mean (Where the bell curve WANTS to be)
    ax.axvline(mu_raw, color='red', linestyle=':', linewidth=2, 
               label=f'Raw Mean ({mu_raw:.1f})')
    
    # B. Truncated Mean (Prediction - Center of Mass)
    ax.axvline(mu_truncated, color='green', linestyle='-', linewidth=2, 
               label=f'Constrained Prediction ({mu_truncated:.1f})')
    
    # C. Actual Sensor Value (if provided)
    if Y_actual is not None:
        actual_value = Y_actual[idx]
        # Handle numpy array by converting to scalar
        if isinstance(actual_value, np.ndarray):
            actual_value = actual_value.item() if actual_value.size == 1 else actual_value[0]
        # Calculate the PDF value at the actual point
        pdf_at_actual = truncnorm.pdf(actual_value, a, b, loc=mu_raw, scale=sigma_raw)
        ax.plot(actual_value, pdf_at_actual, 'o', color='orange', markersize=12, 
                label=f'Actual Sensor Value ({actual_value:.1f})', zorder=5, markeredgewidth=2, 
                markeredgecolor='darkorange')
    
    # Plot Constraints
    if min_val is not None:
        ax.axvline(min_val, color='k', linewidth=3, label='Min Constraint')
    if max_val is not None:
        ax.axvline(max_val, color='k', linewidth=3, label='Max Constraint')
    
    ax.set_title(f"Prediction Distribution at {actual_ts}", fontsize=14)
    ax.set_xlabel("CSO (l/s)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    model = build_gpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0], scaler_Y)
    
    # --- PREDICTION ON TRAIN ---
    Y_pred_train, std_train_y, std_train_f = prediction(model, X_train_scaled, scaler_Y,
                                                         min_val=min_flow, 
                                                         max_val=max_flow)
    
    # --- PREDICTION ON TEST ---
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, 
                                                     short_lag_minutes=optimal_lag, long_lag_hours=24)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    # Use prediction function with truncated Gaussian constraints
    Y_pred_test, std_test_y, std_test_f = prediction(model, X_test_scaled, scaler_Y,
                                                      min_val=min_flow,
                                                      max_val=max_flow)
    
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
    sample_date = str(train_end_time + pd.Timedelta(hours=24))  # it shows the 24th hour of the predictions
    plot_results(timestamps_train, Y_train, Y_pred_train, std_train_y,
                 timestamps_test, Y_test, Y_pred_test, std_test_y,
                 min_level=min_flow, max_level=max_flow,
                 sample_date=sample_date)
    plot_point_of_interest(sample_date, timestamps_test,
                           X_test_scaled, model, scaler_Y, Y_actual=Y_test,
                           min_val=min_flow, max_val=max_flow)

if __name__ == "__main__":
    main()

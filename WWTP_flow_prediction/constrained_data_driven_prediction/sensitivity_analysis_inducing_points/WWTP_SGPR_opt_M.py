""" 
Making a Sparse Gaussian Process Regression model for WWTP data prediction
Sensor data is used. Mean function is created based on the SWMM data.
Constraints are added on this code and plotted based on TRUNCATED GAUSSIAN distribution.
Sparsed GP is evaluated for sensitivity to the number of inducing points (M).
Author: Mohsen 
Updated at: 29/07/2026
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from scipy.stats import norm, truncnorm
from scipy.cluster.vq import kmeans 
from pathlib import Path
import tensorflow_probability as tfp
import pickle

## Load Data
BASE = Path(__file__).parent.parent.parent.parent

data_path = BASE / "data" / "RAW_data" / "pickled_data"

WWTP_inflow = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")
mean_data = pd.read_csv( BASE / "data" / "faf_model" / "DWF_Mean_Function_WWTP_in_Seconds.csv")

##plotting font 
plt.rcParams['font.family'] = 'times new roman'  

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

# constraints 
min_inflow = 0.0  # Minimum WWTP inflow (L/s)
max_inflow = 180.0  # Maximum WWTP inflow (L/s) based on the throttle setting

# Inducing points for Sensitivity Analysis
M_values = [50, 75, 100, 150, 200, 300, 400, 500]

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
inflow = merged_train.filter(like='inflow').iloc[:, 0]
average_inflow = inflow.mean()
p5 = inflow.quantile(0.05)
p95 = inflow.quantile(0.95)
average_inflow_trimmed = inflow[(inflow >= p5) & (inflow <= p95)].mean()

print(f"Training period: {train_start_time} to {train_end_time} ({train_days} days)")
print(f"Test period: {train_end_time} to {test_end_time} ({test_hours} hours)")
print(f"Training samples: {len(merged_train)}")
print(f"Test samples: {len(merged_test)}")
print(f"Data statistics: mean inflow = {merged_train.filter(like='inflow').mean().values[0]:.2f} L/s,\
      mean precipitation = {merged_train.filter(like='precipitation').mean().values[0]*24/timeinterval:.2f} mm/day")
print(f"Trimmed mean inflow (5-95th percentile): {average_inflow_trimmed:.2f} L/s")

def find_optimal_rain_lag(merged_data, interval_minutes, max_lag_hours=2, lag_step_minutes=15):
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    inflow = merged_data[inflow_col].values
    precip = merged_data[precip_col].values
    
    lag_range = range(0, max_lag_hours * 60 + 1, lag_step_minutes)
    correlations = {}
    
    for lag_minutes in lag_range:
        window_size = int(lag_minutes / interval_minutes)
        if window_size < 1: window_size = 1
        rain_accum = pd.Series(precip).rolling(window=window_size).sum().fillna(0).values
        valid_idx = ~(np.isnan(inflow) | np.isnan(rain_accum))
        if valid_idx.sum() > 0:
            corr = np.corrcoef(inflow[valid_idx], rain_accum[valid_idx])[0, 1]
            correlations[lag_minutes] = corr
    
    optimal_lag = max(correlations, key=correlations.get)
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    return optimal_lag, correlations

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, lag_minutes=120, full_rain_series=None): 
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)
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
        
    X_multi = np.hstack((time_feat, rain_accum))
    Y = inflow
    return X_multi, Y, timestamps

def create_lookup_function(pattern_values, time_mean, time_scale):
    pattern_tensor = tf.constant(pattern_values.reshape(-1), dtype=tf.float64)
    num_points = tf.cast(tf.shape(pattern_tensor)[0], tf.float64)
    mu = tf.cast(time_mean, tf.float64)
    sigma = tf.cast(time_scale, tf.float64)
    day_minutes = tf.constant(24.0 * 60.0, dtype=tf.float64)

    def _mean_func(X):
        t_scaled = X[:, 0]
        t_physical = (t_scaled * sigma) + mu
        t_mod = tf.math.floormod(t_physical, day_minutes)
        indices = (t_mod / day_minutes) * (num_points - 1.0)
        val = tfp.math.interp_regular_1d_grid(
            x=indices, x_ref_min=0.0, x_ref_max=num_points - 1.0,
            y_ref=pattern_tensor, fill_value='extrapolate'
        )
        return tf.reshape(val, (-1, 1))
    return _mean_func

def mean_function(mean_data, scaler_Y, timestep, average_inflow):
    timesteps_seconds = timestep * 60 
    mean_times = mean_data['Time_Seconds'].values
    mean_values = mean_data['Mean_WWTP_Inflow'].values
    seconds_in_day = 24 * 60 * 60
    target_times = np.arange(0, seconds_in_day, timesteps_seconds)
    
    mean_pattern = []
    for t in target_times:
        idx = np.argmin(np.abs(mean_times - t))
        mean_pattern.append(mean_values[idx])
    
    mean_pattern = np.array(mean_pattern)
    mean_pattern_scaled = mean_pattern * average_inflow / np.mean(mean_pattern)
    mean_values_reshaped = mean_pattern_scaled.reshape(-1, 1)
    daily_pattern = scaler_Y.transform(mean_values_reshaped)
    return daily_pattern 

def build_gpr_model(X_train, Y_train, time_std_dev, M, mean_func=None):
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev
    
    kernel_daily = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[0], variance=1), period=scaled_period)
    bounded_transform_daily = tfp.bijectors.Sigmoid(low=tf.constant(scaled_period/(24), dtype=tf.float64), high=tf.constant(scaled_period/3, dtype=tf.float64))
    kernel_daily.base_kernel.lengthscales = gpflow.Parameter(0.01, transform=bounded_transform_daily)
    
    kernel_long_term = gpflow.kernels.RBF(variance=1.0, active_dims=[0])
    bounded_transform_long_term = tfp.bijectors.Sigmoid(low=tf.constant(2.0 * scaled_period, dtype=tf.float64), high=tf.constant(8.0 * scaled_period, dtype=tf.float64))
    kernel_long_term.lengthscales = gpflow.Parameter(7.0 * scaled_period, transform=bounded_transform_long_term)
    
    kernel_short_term = gpflow.kernels.Matern32(variance=0.5, active_dims=[0])
    bounded_transform_short_term = tfp.bijectors.Sigmoid(low=tf.constant(scaled_period / 24, dtype=tf.float64), high=tf.constant(scaled_period / 4, dtype=tf.float64))
    kernel_short_term.lengthscales = gpflow.Parameter(scaled_period/12, transform=bounded_transform_short_term)

    kernel_rain = gpflow.kernels.Matern12(variance=1, active_dims=[1])
    bounded_transform_rain = tfp.bijectors.Sigmoid(low=tf.constant(0.1, dtype=tf.float64), high=tf.constant(1, dtype=tf.float64)) 
    kernel_rain.lengthscales = gpflow.Parameter(0.5, transform=bounded_transform_rain)
    
    kernel_daily.active_dims = [0]
    kernel_short_term.active_dims = [0]
    kernel_long_term.active_dims = [0]
    kernel_rain.active_dims = [1]   
    
    gpflow.set_trainable(kernel_daily.period, False)
    kernel = kernel_long_term * kernel_daily + kernel_rain
    
    # MODIFIED: Dynamically uses M from function arguments
    num_inducing = min(M, X_train.shape[0]//2) 
    Z_init, _ = kmeans(X_train, num_inducing)
    Z = tf.convert_to_tensor(Z_init, dtype=tf.float64)
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    model = gpflow.models.SGPR(data=(X_train_tf, Y_train_tf), 
                               kernel=kernel, mean_function=mean_func, 
                               inducing_variable=Z)
    gpflow.set_trainable(model.inducing_variable, True)
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, method='L-BFGS-B')
    return model

def prediction(model, X_scaled, scaler_Y, min_val=None, max_val=None):
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float64)
    mean_sc, var_y_sc = model.predict_y(X_tf)
    _, var_f_sc = model.predict_f(X_tf)
    
    Y_pred = scaler_Y.inverse_transform(mean_sc.numpy())
    std_y = np.sqrt(var_y_sc.numpy()) * scaler_Y.scale_
    std_f = np.sqrt(var_f_sc.numpy()) * scaler_Y.scale_
    
    if min_val is None and max_val is None:
        return Y_pred, std_y, std_f
    
    a = (min_val - Y_pred) / std_y if min_val is not None else -np.inf
    b = (max_val - Y_pred) / std_y if max_val is not None else np.inf
    
    Y_pred_constrained = truncnorm.mean(a=a, b=b, loc=Y_pred, scale=std_y)
    std_y_constrained = truncnorm.std(a=a, b=b, loc=Y_pred, scale=std_y)
    return Y_pred_constrained, std_y_constrained, std_f

def model_evaluation(Y_true, Y_pred, std_pred):
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    lower_bound = Y_pred.ravel() - 1.96 * std_pred.ravel()
    upper_bound = Y_pred.ravel() + 1.96 * std_pred.ravel()
    points_inside = np.sum((Y_true.ravel() >= lower_bound) & (Y_true.ravel() <= upper_bound))
    coverage = points_inside / len(Y_true)
    
    variance = std_pred.ravel() ** 2
    entropy_per_point = 0.5 * np.log2(2 * np.pi * np.e * variance)
    mean_entropy = np.mean(entropy_per_point)
    
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
        "Coverage (%)": coverage * 100,
        "Entropy (nats)": mean_entropy
    }

def plot_sensitivity_elbow(results_df):
    """
    Plots the Pareto front (Elbow curve) for the reviewer.
    Left Y-axis: Test RMSE
    Right Y-axis: Runtime (seconds)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_rmse = 'tab:blue'
    ax1.set_xlabel('Number of Inducing Points (M)', fontsize=12)
    ax1.set_ylabel('Test RMSE (L/s)', color=color_rmse, fontsize=12)
    ax1.plot(results_df['M'], results_df['Test_RMSE'], marker='o', color=color_rmse, linewidth=2, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color_time = 'tab:red'
    ax2.set_ylabel('Runtime (Seconds)', color=color_time, fontsize=12)  
    ax2.plot(results_df['M'], results_df['Runtime_sec'], marker='s', color=color_time, linewidth=2, linestyle='--', label='Runtime')
    ax2.tick_params(axis='y', labelcolor=color_time)

    fig.suptitle('Sensitivity Analysis: Inducing Points (M) vs. Accuracy and Runtime', fontsize=14)
    fig.tight_layout()  
    plt.show()

def main():
    print("="*70)
    print("SGPR SENSITIVITY ANALYSIS: INDUCING POINTS (M)")
    print("="*70)
    
    merged_all = pd.concat([merged_train, merged_test]).sort_index()
    merged_all = merged_all[~merged_all.index.duplicated(keep='first')]
    precip_col = [col for col in merged_all.columns if 'precipitation' in col.lower()][0]
    full_rain_series = merged_all[precip_col]

    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_rain_lag(merged_train, timeinterval)
    
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, 
                                                        lag_minutes=optimal_lag, full_rain_series=full_rain_series)
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, 
                                                     lag_minutes=optimal_lag, full_rain_series=full_rain_series)
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    daily_pattern = mean_function(mean_data, scaler_Y, timestep=timeinterval, average_inflow=average_inflow_trimmed)
    time_mean, time_scale = scaler_X.mean_[0], scaler_X.scale_[0]
    mean_func = create_lookup_function(daily_pattern, time_mean, time_scale)
    
    # Store sensitivity analysis metrics
    sensitivity_results = []
    
    # Loop over M values
    for M in M_values:
        print(f"\n{'-'*50}")
        print(f"Training SGPR Model with M = {M}")
        print(f"{'-'*50}")
        
        start_time = time.perf_counter()
        
        # Build and Train Model
        model = build_gpr_model(X_train_scaled, Y_train_scaled, time_scale, M, mean_func=mean_func)
        
        # Predict on Train & Test
        Y_pred_train, std_train_y, _ = prediction(model, X_train_scaled, scaler_Y)
        Y_pred_test, std_test_y, _ = prediction(model, X_test_scaled, scaler_Y, min_val=min_inflow, max_val=max_inflow)
        
        # Evaluate
        train_metrics = model_evaluation(Y_train, Y_pred_train, std_train_y)
        test_metrics = model_evaluation(Y_test, Y_pred_test, std_test_y)
        
        run_time = time.perf_counter() - start_time
        
        print(f"Finished M={M} | Test RMSE: {test_metrics['RMSE (L/s)']:.3f} | Runtime: {run_time:.1f} sec")
        
        # Append to results
        sensitivity_results.append({
            'M': M,
            'Train_RMSE': train_metrics['RMSE (L/s)'],
            'Test_RMSE': test_metrics['RMSE (L/s)'],
            'Test_MAE': test_metrics['MAE (L/s)'],
            'Runtime_sec': run_time
        })
    
    # Create Summary DataFrame
    results_df = pd.DataFrame(sensitivity_results)
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(results_df.round(3))
    
    # Plot the Elbow Curve
    plot_sensitivity_elbow(results_df)
    
    # Save the sensitivity data to a CSV for manuscript tables/plotting later
    save_path = BASE / "WWTP_flow_prediction"/ "results" / "SGPR" / "SGPR_Sensitivity_Analysis_2019.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nSensitivity analysis data saved to: {save_path}")

if __name__ == "__main__":
    main()
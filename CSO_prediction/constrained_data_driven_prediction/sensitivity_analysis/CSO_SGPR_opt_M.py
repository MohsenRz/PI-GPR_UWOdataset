""" 
Making a Sparse Gaussian Process Regression model for CSO data prediction
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

CSO = pd.read_pickle(
    data_path / "overflow_to_CSO" / "sensor_bf_plsRKBA1101_rubbasin_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")


##plotting font 
plt.rcParams['font.family'] = 'times new roman'  

## Preprocess Data
CSO['timestamp'] = pd.to_datetime(CSO['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters 
train_start_time = pd.to_datetime("2019-09-30 00:00:00") #- pd.Timedelta(days=150)
train_days = 30  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 5 # minutes 

# inducing points 
M = 250    

# constraints 
mean_value = 0 # L/s
min_flow = None # L/s
max_flow = None # No upper limit 

# Inducing points for Sensitivity Analysis
M_values = [50, 75, 100, 150, 200, 300, 400, 500]

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

def initialize_smart_inducing_points(X_train, Y_train, total_M, scaled_threshold):
    # print("\nInitializing Stratified Inducing Points...")
    
    # Identify indices above and below the threshold
    is_above = (Y_train > scaled_threshold).flatten()
    is_below = ~is_above
    
    X_above = X_train[is_above]
    X_below = X_train[is_below]
    
    # Allocate 50% to active periods, 50% to flat periods
    M_above = int(total_M * 0.5)
    M_below = total_M - M_above
    
    # print(f"  - Points > average level: {len(X_above)}. Allocating {M_above} inducing points.")
    # print(f"  - Points <= average level: {len(X_below)}. Allocating {M_below} inducing points.")
    
    # K-means for ABOVE average
    if len(X_above) > M_above:
        Z_above, _ = kmeans(X_above, M_above)
    elif len(X_above) > 0:
        Z_above = X_above
    else:
        Z_above = np.empty((0, X_train.shape[1]))
        
    # K-means for BELOW average
    if len(X_below) > M_below:
        Z_below, _ = kmeans(X_below, M_below)
    elif len(X_below) > 0:
        Z_below = X_below
    else:
        Z_below = np.empty((0, X_train.shape[1]))
        
    Z_combined = np.vstack([Z_above, Z_below])
    
    # Safety net: ensure we have exactly total_M points (in case of shortages)
    if len(Z_combined) < total_M:
        shortage = total_M - len(Z_combined)
        idx = np.random.choice(len(X_train), shortage, replace=False)
        Z_combined = np.vstack([Z_combined, X_train[idx]])
    elif len(Z_combined) > total_M:
        Z_combined = Z_combined[:total_M, :]
        
    return Z_combined

def build_gpr_model(X_train, Y_train, time_std_dev, scaler_Y, threshold_scaled, M):
    """
    Build and train SGPR model
    X_train: Scaled training data 
    Y_train: Scaled target data
    time_std_dev: the scaling factor (std) of the Time Column from scalara_X.scale_[0]
    Added threshold_scaled argument for smart initialisation of inducing points
    """ 
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev  # Adjust period based on scaling
    # print(f"Scaled period for daily cycle: {scaled_period}")
    
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
    
    # --- STRATIFIED SPARSIFICATION ---
    num_inducing = min(M, X_train.shape[0]//2)
    Z_init = initialize_smart_inducing_points(X_train, Y_train, num_inducing, threshold_scaled)
    Z = tf.convert_to_tensor(Z_init, dtype=tf.float64)
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    # Scale mean_value to match Y_train scaling
    if scaler_Y is not None:
        mean_scaled = scaler_Y.transform([[mean_value]])[0, 0]
        # print(f"Mean function: {mean_value} L/s (scaled: {mean_scaled:.4f})")
    else:
        mean_scaled = mean_value
    
    model = gpflow.models.SGPR(data=(X_train_tf, Y_train_tf), 
                              inducing_variable=Z,
                              kernel=kernel, 
                              mean_function=gpflow.mean_functions.Constant(mean_scaled)
                              )
    
    #We force the model to maintain a minimum baseline noise so the bounds don't disappear
    bounded_transform_noise = tfp.bijectors.Sigmoid(
        low=tf.constant(0.05, dtype=tf.float64),  # Lower bound for noise variance
        high=tf.constant(2.0, dtype=tf.float64))
    model.likelihood.variance = gpflow.Parameter(0.1, transform=bounded_transform_noise)
    # freezing inducing points 
    gpflow.set_trainable(model.inducing_variable, True)
    
    # optimisation 
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 method='L-BFGS-B')
    # print("\nModel Summary:")
    # gpflow.utilities.print_summary(model)
    
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
    total_start = time.perf_counter()
    print("="*70)
    print("SGPR SENSITIVITY ANALYSIS: INDUCING POINTS (M) FOR CSO")
    print("="*70)
    
    # Find optimal lag
    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_short_term_rain_lag(merged_train, timeinterval)
    
    # Prepare training data with optimal lag
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, 
                                                        short_lag_minutes=optimal_lag, long_lag_hours=24)
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, 
                                                     short_lag_minutes=optimal_lag, long_lag_hours=24)
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # The average_CSO is physical data. We must scale it to match Y_train_scaled.
    threshold_scaled = scaler_Y.transform([[average_CSO]])[0, 0]
    time_scale = scaler_X.scale_[0]
    
    # Store sensitivity analysis metrics
    sensitivity_results = []
    
    # Loop over M values
    for M in M_values:
        print(f"\n{'-'*50}")
        print(f"Training SGPR Model with M = {M}")
        print(f"{'-'*50}")
        
        start_time = time.perf_counter()
        
        # Build and Train Model
        model = build_gpr_model(X_train_scaled, Y_train_scaled, time_scale, scaler_Y, threshold_scaled, M)
        
        # Predict on Train & Test
        Y_pred_train, std_train_y, _ = prediction(model, X_train_scaled, scaler_Y, min_val=min_flow, max_val=max_flow)
        Y_pred_test, std_test_y, _ = prediction(model, X_test_scaled, scaler_Y, min_val=min_flow, max_val=max_flow)
        
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
    save_path = BASE / "CSO_prediction"/ "constrained_data_driven_prediction" / "sensitivity_analysis" / "CSO_SGPR_Sensitivity_Analysis_2019.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nSensitivity analysis data saved to: {save_path}")

if __name__ == "__main__":
    main()
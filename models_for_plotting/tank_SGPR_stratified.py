"""
making a Gaussian Process Regression model for CSO prediction
Water level in the upstream tank of the CSO should be predicted using GPR. 
sensor data is used  
Author: Mohsen 
Date: 06/02/2026
Updated: 13/04/2026 for saving results 
###########################################the code has a serious peoblem in the kernel part ##############################################
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from scipy.stats import truncnorm, norm
from scipy.cluster.vq import kmeans
from pathlib import Path
import tensorflow_probability as tfp
import pickle

# Load the data
BASE = Path(__file__).parent.parent

data_path = BASE / "RAW_data" / "pickled_data"

Tank_level = pd.read_pickle(
    data_path / "RB59_retention_tank_water_level" / "sensor_bl_plsRKBA1201_rubbasin_ara_2021-01-01_to_2021-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2021_cleaned.pkl")

# Convert timestamp columns to datetime
Tank_level['timestamp'] = pd.to_datetime(Tank_level['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters 
train_start_time = pd.to_datetime("2021-04-10 00:00:00") - pd.Timedelta(days=60)
train_days = 90  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

# insducing points 
M = 250 

# constraints
min_tank_level = 0 # mm
max_tank_level = 3800 # mm
mean_value = 100 # mm
overflow_level = 3200 # mm

tank_train = Tank_level[(Tank_level['timestamp'] >= train_start_time) & (Tank_level['timestamp'] <= train_end_time)]
precipitation_train = precipitation[(precipitation['timestamp'] >= train_start_time) & (precipitation['timestamp'] <= train_end_time)]

tank_test = Tank_level[(Tank_level['timestamp'] >= train_end_time) & (Tank_level['timestamp'] <= test_end_time)]
precipitation_test = precipitation[(precipitation['timestamp'] >= train_end_time) & (precipitation['timestamp'] <= test_end_time)]

tank_train = tank_train.set_index('timestamp')
precipitation_train = precipitation_train.set_index('timestamp')
tank_test = tank_test.set_index('timestamp')
precipitation_test = precipitation_test.set_index('timestamp')

interval_string = f'{timeinterval}min' # resample interval
tank_train_resampled = tank_train.resample(interval_string).mean().fillna(0)
tank_test_resampled = tank_test.resample(interval_string).mean().fillna(0)
precipitation_train_resampled = precipitation_train.resample(interval_string).mean().fillna(0)  
precipitation_test_resampled = precipitation_test.resample(interval_string).mean().fillna(0)

# Merge data 
merged_train = tank_train_resampled.join(precipitation_train_resampled, 
                                         lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_test = tank_test_resampled.join(precipitation_test_resampled, 
                                       lsuffix='_inflow', rsuffix='_precipitation', how='inner')
merged_train.dropna(inplace=True)
merged_test.dropna(inplace=True)
average_tank_level = Tank_level['value'].mean()
average_precipitation = precipitation['value'].mean()

print(f"Training period: {train_start_time} to {train_end_time} ({train_days} days)")
print(f"Test period: {train_end_time} to {test_end_time} ({test_hours} hours)")
print(f"Training samples: {len(merged_train)}")
print(f"Test samples: {len(merged_test)}")
print(f"Data statistics: mean daily water level = {average_tank_level:.1f} mm,\
      mean precipitation = {merged_train.filter(like='precipitation').mean().values[0]*60*24/timeinterval:.2f} mm/day")

def find_optimal_rain_lag(merged_data, interval_minutes, max_lag_hours=2, lag_step_minutes=15):
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
    plt.ylabel('Cross-correlation with Tank Level ', fontsize=12)
    plt.title('Cross-Correlation Analysis: Rain Lag vs Tank Level Response', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    print(f"\nOptimal precipitation lag: {optimal_lag} minutes")
    
    return optimal_lag, correlations

def preparing_data(merged_data, start_time, interval_minutes=timeinterval, lag_minutes=120): 
    #creating multi dimensional input as timestamps and precipitation data
    timestamps = merged_data.index
    level_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)  # time in minutes
    ## I sum up the previous 60 mins precipitation to consider lag effect
    # Calculate window size dynamically (Target 60 mins / Interval)
    window_size = int(lag_minutes / interval_minutes) 
    if window_size < 1: window_size = 1
    
    rain_series = merged_data[precip_col]
    # rolling sum, fill NaN at start with 0
    rain_accum = rain_series.rolling(window=window_size).sum().fillna(0).values.reshape(-1, 1)
    
    level = merged_data[level_col].values.reshape(-1, 1)
    
    X_multi = np.hstack((time_feat, rain_accum))
    Y = level
        
    return X_multi, Y, timestamps

def initialize_smart_inducing_points(X_train, Y_train, total_M, scaled_threshold):
    print("\nInitializing Stratified Inducing Points...")
    
    # Identify indices above and below the threshold
    is_above = (Y_train > scaled_threshold).flatten()
    is_below = ~is_above
    
    X_above = X_train[is_above]
    X_below = X_train[is_below]
    
    # Allocate 50% to active periods, 50% to flat periods
    M_above = int(total_M * 0.5)
    M_below = total_M - M_above
    
    print(f"  - Points > average level: {len(X_above)}. Allocating {M_above} inducing points.")
    print(f"  - Points <= average level: {len(X_below)}. Allocating {M_below} inducing points.")
    
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

def build_gpr_model(X_train, Y_train,time_std_dev, scaler_Y=None, threshold_scaled=None):
    """
    Build and train GPR model
    X_train: Scaled training data 
    Y_train: Scaled target data
    time_std_dev: the scaling factor (std) of the Time Column from scalara_X.scale_[0]
    Added threshold_scaled argument for smart initialisation of inducing points
    """ 
    
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev  # Adjust period based on scaling
    print(f"Scaled period for daily cycle: {scaled_period}")
    
    ### naive kernel 
    naive_kernel = gpflow.kernels.RBF()
    
    ### kernel design 
    time_kernel = gpflow.kernels.RBF(lengthscales=scaled_period/6, variance=0.5, active_dims=[0]) 
    
    ### rain kernel 
    kernel_rain = gpflow.kernels.Matern12(variance=10, active_dims=[1])
    bounded_transform_rain = tfp.bijectors.Sigmoid(
        low=tf.constant(0.01, dtype=tf.float64), 
        high=tf.constant(0.5, dtype=tf.float64)) 
    kernel_rain.lengthscales = gpflow.Parameter(0.1, transform=bounded_transform_rain)
    
    kernel = time_kernel + kernel_rain
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    # --- STRATIFIED SPARSIFICATION ---
    num_inducing = min(M, X_train.shape[0]//2)
    Z_init = initialize_smart_inducing_points(X_train, Y_train, num_inducing, threshold_scaled)
    Z = tf.convert_to_tensor(Z_init, dtype=tf.float64)
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    # Scale mean_value to match Y_train scaling
    if scaler_Y is not None:
        mean_scaled = scaler_Y.transform([[mean_value]])[0, 0]
        print(f"Mean function: {mean_value} mm (scaled: {mean_scaled:.4f})")
    else:
        mean_scaled = mean_value
    
    model = gpflow.models.SGPR(data=(X_train_tf, Y_train_tf), 
                              inducing_variable=Z,
                              kernel=kernel, 
                              mean_function=gpflow.mean_functions.Constant(mean_scaled)
                              )
    gpflow.set_trainable(model.mean_function, False)
    # freezing inducing points or not
    gpflow.set_trainable(model.inducing_variable, True)
    
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
    
    return {
        "RMSE (mm)": RMSE,
        "MAE (mm)": MAE,
        "Coverage (%)": coverage * 100,
        "Entropy (bits)": mean_entropy
    }
    
def plot_results(timestamps_train, Y_train, Y_pred_train, std_train,
                 timestamps_test, Y_test, Y_pred_test, std_test, 
                 min_level=None, max_level=None, sample_date=None,
                 Z_timestamps=None, Z_values=None):
    """
    Plot training and test results with uncertainty.
    Credible Intervals are clipped 
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    ax = axs[0]
    
    # --- existing plot code on ax instead of ax --- 
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=10, 
               label='Training Data', alpha=0.3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='GPR Fit (Training)', linewidth=1.5)
    ax.fill_between(timestamps_train, 
                    Y_pred_train.ravel() - 1.96 * std_train.ravel(),
                    Y_pred_train.ravel() + 1.96 * std_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Training)')
    # plotting inducing points 
    if Z_timestamps is not None and Z_values is not None:
        ax.scatter(Z_timestamps, Z_values.ravel(), c='purple', s=50, 
                   label='Inducing Points', marker='|', zorder=5)
    
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
    ax.axvline(x=timestamps_train[-1], color='black', linestyle='--', 
               linewidth=1.5, label='Train/Test Split', zorder=5)
    if sample_date is not None:
        sample_ts = pd.to_datetime(sample_date)
        ax.axvline(x=sample_ts, color='purple', linestyle=':', 
                   linewidth=1.5, label='Point of Interest', zorder=5)
    ax.set_ylabel('Tank Level (mm)', fontsize=12)
    ax.set_title(f'GPR: Tank Level Prediction (Train: {train_days} days, Test: {test_hours} hours)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # --- CSO probability subplot ---
    Z_scores = (overflow_level - Y_pred_test.ravel()) / std_test.ravel()
    cso_probabilities = (1 - norm.cdf(Z_scores)) * 100

    axs[1].plot(timestamps_test, cso_probabilities, 'r-', linewidth=2, label='CSO Probability (%)')
    #axs[1].axhline(y=10, color='yellow', linestyle='--', alpha=0.7, label='Low Risk (10%)')
    #axs[1].axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Medium Risk (25%)')
    #axs[1].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='High Risk (50%)')
    axs[1].set_ylabel('CSO Probability (%)', fontsize=12)
    axs[1].set_xlabel('Date', fontsize=12)
    axs[1].set_title(f'Probability of CSO (Tank Level > {overflow_level} mm)', fontsize=14)
    axs[1].set_ylim(0, 100)
    axs[1].legend(fontsize=11)
    axs[1].grid(True, alpha=0.3)

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def main():
    total_start = time.perf_counter()
    print("="*70 + "\nTank water prediction\n" + "="*70)
    
    # Find optimal lag
    print("\nPerforming cross-correlation analysis...")
    optimal_lag, correlations = find_optimal_rain_lag(merged_train, timeinterval)
    
    # Prepare training data (Pass interval to calc window size)
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval, lag_minutes=optimal_lag)
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    # The average tank level is physical data. We must scale it to match Y_train_scaled.
    threshold_scaled = scaler_Y.transform([[average_tank_level]])[0, 0]
    
    print("\nTraining GPR model...")
    # Pass Time Std Dev for Period Calculation
    model = build_gpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0], scaler_Y, threshold_scaled)
    
    # --- PREDICTION ON TRAIN ---
    Y_pred_train, std_train_y, std_train_f = prediction(model, X_train_scaled, scaler_Y,
                                                         min_val=min_tank_level, 
                                                         max_val=max_tank_level)
    
    # --- PREDICTION ON TEST ---
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval, lag_minutes=optimal_lag)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    # Use prediction function with truncated Gaussian constraints
    Y_pred_test, std_test_y, std_test_f = prediction(model, X_test_scaled, scaler_Y,
                                                      min_val=min_tank_level,
                                                      max_val=max_tank_level)
    
    # Extracting inducing points for plotting
    Z_scaled = model.inducing_variable.Z.numpy()
    Z_unscaled = scaler_X.inverse_transform(Z_scaled)
    Z_time_minutes = Z_unscaled[:, 0]
    Z_timestamps = [train_start_time + pd.Timedelta(minutes=float(tm)) for tm in Z_time_minutes]
    mu_Z_scaled, _ = model.predict_f(Z_scaled)
    Z_values = scaler_Y.inverse_transform(mu_Z_scaled.numpy())
    
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
    sample_date = None
    plot_results(timestamps_train, Y_train, Y_pred_train, std_train_y,
                 timestamps_test, Y_test, Y_pred_test, std_test_y,
                 min_level=min_tank_level, max_level=max_tank_level,
                 sample_date=sample_date, 
                 Z_timestamps=Z_timestamps, Z_values=Z_values)

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
    save_path = BASE / "results" / "Tank_level_outputs" / "2021_Tank_SGPR_stratified_90days_train.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Data successfully saved to: {save_path}")
    
    
    
if __name__ == "__main__":
    main()
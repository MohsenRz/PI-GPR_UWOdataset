""" 
making a Gaussian Process Regression model for WWTP data prediction
sensor data is used 
sparsification is applied here using SGPR model from GPflow library
Warping is applied to ensure the flow doesn't pass 0 or 180 L/s in all train and test points 
Author: Mohsen 
Date: 18/12/2025
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from scipy.cluster.vq import kmeans 
from scipy.stats import norm
from pathlib import Path
import tensorflow_probability as tfp

# Warping ============================
class LogitWarper:
    """
    Transforms bounded data [min_val, max_val] to unconstrained real space (-inf, inf)
    using a Logit transformation, and inverses it using a Sigmoid. 
    """ 
    def __init__(self, min_val, max_val, epsilon=1e-5):
        self.min_val = min_val
        self.max_val = max_val
        self.epsilon = epsilon # Prevents log(0)
        
    def transform(self, y): 
        """Forward: Physical -> Latent"""
        # 1. Normalize to [0, 1]
        y_norm = (y - self.min_val) / (self.max_val - self.min_val)
        # 2. Clip to avoid exactly 0 or 1
        y_clamped = np.clip(y_norm, self.epsilon, 1.0 - self.epsilon)
        # 3. Logit transform
        y_warped = np.log(y_clamped / (1.0 - y_clamped))
        return y_warped

    def inverse_transform(self, y_warped):
        """Inverse: Latent -> Physical"""
        # 1. Sigmoid
        y_sigmoid = 1.0 / (1.0 + np.exp(-y_warped))
        # 2. Scale back to physical units
        y_physical = self.min_val + (y_sigmoid * (self.max_val - self.min_val))
        return y_physical

## Load Data
BASE = Path(__file__).parent.parent

data_path = BASE / "RAW_data" / "pickled_data"

WWTP_inflow = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

## Preprocess Data
WWTP_inflow['timestamp'] = pd.to_datetime(WWTP_inflow['timestamp']) # L/s
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp']) # mm/hr

# train parameters
train_start_time = pd.to_datetime("2019-02-01 00:00:00")
train_days = 30  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

# constraints 
min_inflow = 0.0  # Minimum WWTP inflow (L/s)
max_inflow = 180.0  # Maximum WWTP inflow (L/s) based on the throttle setting

# inducing points 
M = 1000  # number of inducing points

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

def preparing_data(merged_data, start_time, interval_minutes=timeinterval): 
    #creating multi dimensional input as timestamps and precipitation data
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)  # time in minutes
    window_size = int(60 / interval_minutes) 
    if window_size < 1: window_size = 1
    
    rain_series = merged_data[precip_col]
    rain_accum = rain_series.rolling(window=window_size).sum().fillna(0).values.reshape(-1, 1)
    
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
    
    X_multi = np.hstack((time_feat, rain_accum))
    Y = inflow
    
    return X_multi, Y, timestamps

def build_sgpr_model(X_train, Y_train, time_std_dev):
    """
    Build and train sparse GPR model 
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
    kernel_daily.base_kernel.lengthscales = gpflow.Parameter(0.02 * scaled_period, transform=bounded_transform_daily)
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
    
    kernel = kernel_daily * kernel_long_term + kernel_rain # noise is removed because the likelihood variance takes the overal noise
    
    # sparsification 
    num_inducing = min(M, X_train.shape[0]//2)  # choose number of inducing points
    Z_init, _ = kmeans(X_train, num_inducing)
    Z = tf.convert_to_tensor(Z_init, dtype=tf.float64)
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    model = gpflow.models.SGPR(
        data=(X_train_tf, Y_train_tf),
        kernel=kernel, mean_function=None, 
        inducing_variable=Z
        )
    # I can freeze the inducing points location for small datasets 
    gpflow.set_trainable(model.inducing_variable, True)
    
    #optimise hyperparameters
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, 
                 model.trainable_variables, 
                 method='L-BFGS-B')
    
    print("\nModel Summary:")
    gpflow.utilities.print_summary(model)
 
    return model

def prediction(model, X_scaled, scaler_Y, warper, n_samples=10000):
    """
    making predictions considering global constraints 
    the kernel is using inducing points for sparsification
    """
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float64)
    
    # Predict in the Latent (Warped + Scaled) Space
    # predict_y includes the likelihood noise, which is correct for predictive intervals
    mean_lat_scaled, var_lat_scaled = model.predict_y(X_tf)
    
    # Unscale (back to just Warped Space)
    mu_lat = scaler_Y.inverse_transform(mean_lat_scaled.numpy())
    std_lat = np.sqrt(var_lat_scaled.numpy()) * scaler_Y.scale_
    
    # Monte Carlo Sampling
    # Shape: (n_samples, n_test_points)
    rng = np.random.default_rng(42)
    # We create standard normal samples and scale them
    z_samples = rng.standard_normal((n_samples, len(X_scaled)))
    # Broadcast mu and std
    lat_samples = mu_lat.T + (std_lat.T * z_samples)
    
    # INVERSE WARP: Transform samples back to physical space [0, 180]
    phys_samples = warper.inverse_transform(lat_samples)
    
    # Calculate Statistics in Physical Space
    Y_pred_mean = np.mean(phys_samples, axis=0).reshape(-1, 1)
    
    # Asymmetric Confidence Intervals (e.g. 95%)
    lower_ci = np.percentile(phys_samples, 2.5, axis=0).reshape(-1, 1)
    upper_ci = np.percentile(phys_samples, 97.5, axis=0).reshape(-1, 1)
    
    # Std for metric calculation (approximate, since distribution is skewed)
    std_y = np.std(phys_samples, axis=0).reshape(-1, 1)
        
    return Y_pred_mean, std_y, lower_ci, upper_ci, phys_samples
    
def model_evaluation(Y_true, Y_pred, lower_ci, upper_ci):
    """
    Updated metrics for asymmetric CIs
    """
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    # Coverage (using calculated CIs)
    points_inside = np.sum((Y_true.ravel() >= lower_ci.ravel()) & 
                           (Y_true.ravel() <= upper_ci.ravel()))
    coverage = points_inside / len(Y_true)
    
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
        "Coverage (%)": coverage * 100
    }

def plot_results(timestamps_train, Y_train, Y_pred_train, lower_train, upper_train,
                 timestamps_test, Y_test, Y_pred_test, lower_test, upper_test,
                 Z_timestamps=None, sample_date=None):
    """
    Plot training and test results with inducing points 
    point of interest is shown on the plot, if defined 
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=10, 
               label='Training Data', alpha=0.3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='SGPR Fit', linewidth=1.5)
    ax.fill_between(timestamps_train, lower_train.ravel(), upper_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Train)')
    
    # 2. Plot Inducing Points (Stick to x-axis/minimum value)
    if Z_timestamps is not None:
        # We place them at the very bottom of the data range
        y_bottom = Y_train.min() 
        ax.plot(Z_timestamps, [y_bottom] * len(Z_timestamps), '|', c='k', 
                markersize=15, markeredgewidth=1.5, 
                label='Inducing Points (Z)', zorder=5)

    # 3. Plot Test Data
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data', alpha=0.6)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='Prediction', linewidth=1.5)
    ax.fill_between(timestamps_test, lower_test.ravel(), upper_test.ravel(),
                    alpha=0.2, color='red', label='95% CI (Test)')
    
    # Separators
    ax.axvline(x=timestamps_train[-1], color='black', linestyle='--', linewidth=1.5)
    
    if sample_date:
        ax.axvline(x=pd.to_datetime(sample_date), color='purple', linestyle=':', lw=1.5)
    
    ax.set_title(f'Sparse Warped GPR: Inflow Prediction', fontsize=14)
    ax.set_ylabel('Inflow (L/s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.show()

def plot_point_of_interest(target_date_str, timestamps, phys_samples_matrix, Y_actual=None): 
    """
    Plots the histograms of the predicted sample point with Monte Carlo method 
    """
    target_ts = pd.to_datetime(target_date_str)
    idx = np.argmin(np.abs(timestamps - target_ts))
    actual_ts = timestamps[idx]
    
    # Get samples for this specific time step
    # phys_samples matrix is (n_samples, n_timesteps) - wait, from prediction it was (samples, points)
    # Check transpose in prediction func: lat_samples was (samples, len). Correct.
    samples_at_t = phys_samples_matrix[:, idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of samples
    ax.hist(samples_at_t, bins=50, density=True, alpha=0.6, color='skyblue', label='MC Samples')

    mean_val = np.mean(samples_at_t)
    median_val = np.median(samples_at_t)
    
    ax.axvline(mean_val, color='red', linestyle='-', lw=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', lw=2, label=f'Median: {median_val:.1f}')
    
    # Constraints
    ax.axvline(min_inflow, color='k', lw=3, label='Min Constraint')
    ax.axvline(max_inflow, color='k', lw=3, label='Max Constraint')
    # Actual value
    if Y_actual is not None:
        actual_val = Y_actual[idx][0]
        ax.axvline(actual_val, color='Orange', linestyle=':', lw=2, label=f'Actual Sensor Value: {actual_val:.1f}')
    
    ax.set_title(f"Predictive Distribution at {actual_ts}", fontsize=14)
    ax.set_xlabel("Inflow (L/s)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
     
def main():
    total_start = time.perf_counter()
    print("="*70)
    print("Sparse Warped GPR Model for WWTP Inflow Prediction")
    print("="*70)
    
    # Prepare training data (Pass interval to calc window size)
    X_train, Y_train_phys, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval)
    
    # Transform Physical Y (0-180) -> Latent Y (-inf, inf)
    warper = LogitWarper(min_val=min_inflow, max_val=max_inflow)
    Y_train_warped = warper.transform(Y_train_phys)
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y_warped = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y_warped.fit_transform(Y_train_warped)
    
    print("\nTraining GPR model...")
    # Pass Time Std Dev for Period Calculation
    model = build_sgpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0])
    
    # --- PREDICTION ON TRAIN ---
    Y_pred_train, _, lower_train, upper_train, _ = prediction(
        model, X_train_scaled, scaler_Y_warped, warper)
    
    # --- PREDICTION ON TEST ---
    X_test, Y_test_phys, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    Y_pred_test, _, lower_test, upper_test, samples_test = prediction(
        model, X_test_scaled, scaler_Y_warped, warper)
        
    # EXTRACT INDUCING POINTS FOR PLOTTING ---
    print("\nExtracting Inducing Points...")

    Z_scaled = model.inducing_variable.Z.numpy()
    
    # Unscale X-coordinates (Time and Rain)
    Z_unscaled = scaler_X.inverse_transform(Z_scaled)
    Z_time_minutes = Z_unscaled[:, 0]
    Z_timestamps = [train_start_time + pd.Timedelta(minutes=float(m)) for m in Z_time_minutes]
    
    # --- EVALUATION TABLE ---
    train_metrics = model_evaluation(Y_train_phys, Y_pred_train, lower_train, upper_train)
    test_metrics = model_evaluation(Y_test_phys, Y_pred_test, lower_test, upper_test)
    
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
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Prediction complete!")
    # --- PLOTTING ---
    sample_date = str(train_end_time + pd.Timedelta(hours=24))  # it shows the 24th hour of the predictions 

    plot_results(timestamps_train, Y_train_phys, Y_pred_train, lower_train, upper_train,
                 timestamps_test, Y_test_phys, Y_pred_test, lower_test, upper_test,
                 Z_timestamps=Z_timestamps,
                 sample_date=sample_date)

    plot_point_of_interest(sample_date, timestamps_test, samples_test, Y_actual=Y_test_phys)

if __name__ == "__main__":
    main()


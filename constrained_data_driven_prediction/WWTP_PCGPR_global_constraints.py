""" 
making a Gaussian Process Regression model for WWTP data prediction
sensor data is used 
physical constraints are added to get better performance
here we have min and max constraints for the WWTP inflow predictions 
the global constraints are used in this implementation with WARPING method 
Author: Mohsen 
Date: 08/12/2025
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
WWTP_inflow['timestamp'] = pd.to_datetime(WWTP_inflow['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

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
    ## I sum up the previous 60 mins precipitation to consider lag effect
    # Calculate window size dynamically (Target 60 mins / Interval)
    window_size = int(60 / interval_minutes) 
    if window_size < 1: window_size = 1
    
    rain_series = merged_data[precip_col]
    # rolling sum, fill NaN at start with 0
    rain_accum = rain_series.rolling(window=window_size).sum().fillna(0).values.reshape(-1, 1)
    
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
    scaled_period = minutes_in_day / time_std_dev  # Adjusting period based on scaling
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

# main changes come here 
def prediction_warped (model, X_scaled, scaler_Y_warped, warper, n_samples=2000):
    """
    Performs predictions by:
    1. Predicting latent Gaussian distribution (Mean/Var) in warped space.
    2. Sampling from this latent distribution.
    3. Inverse warping the samples to physical space.
    4. Calculating statistics (Mean, CI) on the physical samples.
    """
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float64)
    # Predict in the Latent (Warped + Scaled) Space
    # predict_y includes the likelihood noise, which is correct for predictive intervals
    mean_lat_scaled, var_lat_scaled = model.predict_y(X_tf)
    mu_lat = scaler_Y_warped.inverse_transform(mean_lat_scaled.numpy())
    std_lat = np.sqrt(var_lat_scaled.numpy()) * scaler_Y_warped.scale_
    
    # Monte Carlo Sampling
    # Shape: (n_samples, n_test_points)
    rng = np.random.default_rng(42)
    # We create standard normal samples and scale/shift them
    z_samples = rng.standard_normal((n_samples, len(X_scaled)))
    
    # Broadcast mu and std
    lat_samples = mu_lat.T + (std_lat.T * z_samples)
    
    # INVERSE WARP: Transform samples back to physical space [0, 180]
    phys_samples = warper.inverse_transform(lat_samples)
    
    # Calculate Statistics in Physical Space
    Y_pred_mean = np.mean(phys_samples, axis=0).reshape(-1, 1)
    Y_pred_median = np.median(phys_samples, axis=0).reshape(-1, 1)
    
    # Asymmetric Confidence Intervals (e.g. 95%)
    lower_ci = np.percentile(phys_samples, 2.5, axis=0).reshape(-1, 1)
    upper_ci = np.percentile(phys_samples, 97.5, axis=0).reshape(-1, 1)
    
    # Std for metric calculation (approximate, since distribution is skewed)
    std_y = np.std(phys_samples, axis=0).reshape(-1, 1)
     
    return Y_pred_mean, std_y, lower_ci, upper_ci, phys_samples

############################ to here 



def model_evaluation(Y_true, Y_pred, lower_ci, upper_ci):
    """
    Returns a dictionary of metrics for easy table formatting.
    """
    # Standard Metrics
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    # Coverage
    points_inside = np.sum((Y_true.ravel() >= lower_ci.ravel()) & 
                           (Y_true.ravel() <= upper_ci.ravel()))
    coverage = points_inside / len(Y_true)
    """
    # Entropy (Nats)
    variance = std_pred.ravel() ** 2
    # Use log2 for Bits
    entropy_per_point = 0.5 * np.log2(2 * np.pi * np.e * variance)
    mean_entropy = np.mean(entropy_per_point)
    """
    return {
        "RMSE (L/s)": RMSE,
        "MAE (L/s)": MAE,
        "Coverage (%)": coverage * 100,
        #"Entropy (nats)": mean_entropy
    }

def plot_results(timestamps_train, Y_train, Y_pred_train, lower_train, upper_train,
                 timestamps_test, Y_test, Y_pred_test, lower_test, upper_test,
                 sample_date=None):
    """
    Plot training and test results
    credible intervals are clipped 
    sample date is shown on the plot 
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=10, 
               label='Training Data', alpha=0.3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='Warped GP Fit', linewidth=1.5)
    ax.fill_between(timestamps_train, lower_train.ravel(), upper_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Asymmetric)')
    
    # Plot test data
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data', alpha=0.6)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='Warped GP Prediction', linewidth=1.5)
    ax.fill_between(timestamps_test, lower_test.ravel(), upper_test.ravel(),
                    alpha=0.2, color='red')
    
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

def plot_point_of_interest(target_date_str, timestamps, phys_samples_matrix):
    """
    Plots the full probability distribution (PDF) for a specific timestamp.
    Visualizes the difference between Raw Mean, Truncated Mean, and Mode.
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
    
    # KDE for smooth line
    try:
        density = norm.pdf(np.linspace(min(samples_at_t), max(samples_at_t), 100), 
                           np.mean(samples_at_t), np.std(samples_at_t))
        # ax.plot(np.linspace(min(samples_at_t), max(samples_at_t), 100), density, 'b--', label='Gaussian Approx')
    except: pass

    mean_val = np.mean(samples_at_t)
    median_val = np.median(samples_at_t)
    
    ax.axvline(mean_val, color='red', linestyle='-', lw=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', lw=2, label=f'Median: {median_val:.1f}')
    
    # Constraints
    ax.axvline(MIN_INFLOW, color='k', lw=3, label='Min Constraint')
    ax.axvline(MAX_INFLOW, color='k', lw=3, label='Max Constraint')
    
    ax.set_title(f"Predictive Distribution at {actual_ts}\n(Note non-Gaussian skew near bounds)", fontsize=14)
    ax.set_xlabel("Inflow (L/s)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def main():
    total_start = time.perf_counter()
    print("="*70) 
    print("GPR Model for WWTP Inflow Prediction")
    print("="*70)
    
    # Prepare training data (Pass interval to calc window size)
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval)
    range_val = max_inflow - min_inflow
    warp = tfp.bijectors.Chain([
        tfp.bijectors.AffineScalar(shift=-min_inflow),
        tfp.bijectors.Scale(scale=1.0 / range_val),
        tfp.bijectors.Sigmoid()
    ])
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    print("\nTraining GPR model...")
    # Pass Time Std Dev for Period Calculation
    model = build_gpr_model(X_train_scaled, Y_train_scaled, scaler_X.scale_[0])
    
    # --- PREDICTION ON TRAIN ---
    Y_pred_train, std_train_y, std_train_f = prediction(model, X_train_scaled, scaler_Y,
                                                        min_val=None, max_val=None)

    # --- PREDICTION ON TEST ---
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions...")
    Y_pred_test, std_test_y, std_test_f = prediction(model, X_test_scaled, scaler_Y, 
                                                     min_val=min_inflow, max_val=max_inflow)
    
    # --- COMPARISON TABLE ---
    train_metrics = model_evaluation(Y_train, Y_pred_train, std_train_y)
    test_metrics = model_evaluation(Y_test, Y_pred_test, std_test_y)
    
    results_df = pd.DataFrame({
        'Training Set': train_metrics,
        'Test Set': test_metrics})
    
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
                 min_inflow, max_inflow, sample_date=sample_date)
    
    plot_point_of_interest(sample_date, timestamps_test,
                           X_test_scaled, model, scaler_Y,
                           min_val=min_inflow, max_val=max_inflow)

if __name__ == "__main__":
    main()

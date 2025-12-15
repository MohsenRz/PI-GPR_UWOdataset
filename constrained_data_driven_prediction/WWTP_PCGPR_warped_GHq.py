""" 
Gaussian Process Regression model for WWTP data prediction with Global Warping
- Sensor data is used 
- Physical constraints (0 to 180) are enforced globally via Warping (Logit Transform)
- Training is performed in the unconstrained latent space
- Predictions use Gauss-Hermite quadrature to get the unscaled physical mean and variance
Author: Mohsen
Date: 15/12/2025
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow 
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import time
from pathlib import Path
from scipy.stats import norm
import tensorflow_probability as tfp
from numpy.polynomial.hermite import hermgauss

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

## Loading Data    ----------------------------------
BASE = Path(__file__).parent.parent

data_path = BASE / "RAW_data" / "pickled_data"

# NOTE: Ensure these paths match your local structure
WWTP_inflow = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

## Preprocess Data
WWTP_inflow['timestamp'] = pd.to_datetime(WWTP_inflow['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

# train parameters
train_start_time = pd.to_datetime("2019-02-01 00:00:00")
train_days = 30 # days 
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 5 * 24  # hours 
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15 # minutes 

# GLOBAL Constraints
MIN_INFLOW = 0.0
MAX_INFLOW = 180.0 

WWTP_inflow_train = WWTP_inflow[(WWTP_inflow['timestamp'] >= train_start_time) & (WWTP_inflow['timestamp'] <= train_end_time)]
precipitation_train = precipitation[(precipitation['timestamp'] >= train_start_time) & (precipitation['timestamp'] <= train_end_time)]

WWTP_inflow_test = WWTP_inflow[(WWTP_inflow['timestamp'] >= train_end_time) & (WWTP_inflow['timestamp'] <= test_end_time)]
precipitation_test = precipitation[(precipitation['timestamp'] >= train_end_time) & (precipitation['timestamp'] <= test_end_time)]

WWTP_inflow_train.set_index('timestamp', inplace=True)
precipitation_train.set_index('timestamp', inplace=True)
WWTP_inflow_test.set_index('timestamp', inplace=True)
precipitation_test.set_index('timestamp', inplace=True)

interval_string = f'{timeinterval}min'
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

print(f"Training period: {train_start_time} to {train_end_time}")
print(f"Data statistics: mean inflow = {merged_train.filter(like='inflow').mean().values[0]:.2f} L/s")

# preparing data for building the model ------------------------------
def preparing_data(merged_data, start_time, interval_minutes=timeinterval): 
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time_feat = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)
    
    window_size = int(60 / interval_minutes) 
    if window_size < 1: window_size = 1
    
    rain_series = merged_data[precip_col]
    rain_accum = rain_series.rolling(window=window_size).sum().fillna(0).values.reshape(-1, 1)
    
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
    
    X_multi = np.hstack((time_feat, rain_accum))
    Y = inflow
        
    return X_multi, Y, timestamps

def build_gpr_model(X_train, Y_train, time_std_dev):
    """
    Build and train GPR model.
    Note: Y_train here is already Warped and Scaled.
    """ 
    minutes_in_day = 24 * 60
    scaled_period = minutes_in_day / time_std_dev 
    
    # --- Kernels ---
    # Daily Periodic
    kernel_daily = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[0], variance=1), 
                                            period=scaled_period)
    bounded_transform_daily = tfp.bijectors.Sigmoid(
        low=tf.constant(scaled_period/(24*60), dtype=tf.float64),
        high=tf.constant(scaled_period/3, dtype=tf.float64))
    kernel_daily.base_kernel.lengthscales = gpflow.Parameter(0.01, transform=bounded_transform_daily)
    
    # Long term Trend
    kernel_long_term = gpflow.kernels.RBF(variance=1.0, active_dims=[0])
    bounded_transform_long_term = tfp.bijectors.Sigmoid(
        low=tf.constant(4.0 * scaled_period, dtype=tf.float64),
        high=tf.constant(8.0 * scaled_period, dtype=tf.float64))
    kernel_long_term.lengthscales = gpflow.Parameter(7.0 * scaled_period, transform=bounded_transform_long_term)
    
    # Rain Effect
    kernel_rain = gpflow.kernels.Matern12(variance=1, active_dims=[1])
    bounded_transform_rain = tfp.bijectors.Sigmoid(
        low=tf.constant(0.1, dtype=tf.float64), 
        high=tf.constant(1, dtype=tf.float64)) 
    kernel_rain.lengthscales = gpflow.Parameter(0.5, transform=bounded_transform_rain)
    
    # Noise is handled by the Likelihood variance, but we can add a small jitter kernel if needed
    
    kernel_daily.active_dims = [0]
    kernel_long_term.active_dims = [0]
    kernel_rain.active_dims = [1]   
    
    gpflow.set_trainable(kernel_daily.period, False)
    
    kernel = kernel_daily * kernel_long_term + kernel_rain 
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    model = gpflow.models.GPR(data=(X_train_tf, Y_train_tf), 
                              kernel=kernel, mean_function=None)
    
    # Optimise
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, 
                 model.trainable_variables, 
                 method='L-BFGS-B')
    
    print("\nModel Summary:")
    gpflow.utilities.print_summary(model)
 
    return model

def prediction_warped(model, X_scaled, scaler_Y_warped, warper):
    """
    Performs predictions using Gauss-Hermite quadrature for moments (mean/Var) and Quantile mapping for CI.
    n_quad (int): number of quadrature points.
    """
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float64)
    
    # Predict in the Latent (Warped + Scaled) Space
    # predict_y includes the likelihood noise, which is correct for predictive intervals
    mean_lat_scaled, var_lat_scaled = model.predict_y(X_tf)
    
    # Unscale (back to just Warped Space)
    mu_lat = scaler_Y_warped.inverse_transform(mean_lat_scaled.numpy())
    std_lat = np.sqrt(var_lat_scaled.numpy()) * scaler_Y_warped.scale_
    
    # Gauss-Hermite Quadrature for mean and variance 
    # weights (2) and roots (x) 
    n_quad = 30 
    x_GH, w_GH = hermgauss(n_quad)   
    # Broadcast mu and std
    x_GH = x_GH.reshape(-1, 1)  # (n_quad, 1)
    mu_lat_T = mu_lat.T  # (1, n_points)
    std_lat_T = std_lat.T  # (1, n_points)
    
    # change of variables: z = mu + sqrt(2)*sigma*x 
    f_grid = mu_lat_T + np.sqrt(2) * std_lat_T * x_GH  # (n_quad, n_points)
    
    # Transform to Physical Space
    phys_grid = warper.inverse_transform(f_grid)  # (n_quad, n_points)
    
    # Integration: E[y] = (1/sqrt(phi)) * sum(w_i * y_i)
    factor = 1.0 / np.sqrt(np.pi)
    Y_pred_mean = factor * np.sum(w_GH.reshape(-1, 1) * phys_grid, axis=0).reshape(-1, 1)
    
    # Integrating for E[y^2]
    Y_pred_sq_mean = factor * np.sum(w_GH.reshape(-1, 1) * (phys_grid**2), axis=0).reshape(-1, 1)
    
    # variance = E[y^2] - (E[y])^2
    Y_pred_var = Y_pred_sq_mean - (Y_pred_mean ** 2)
    Y_pred_std = np.sqrt(np.maximum(Y_pred_var, 1e-10))  # Prevent negative variance due to numerical issues
    
    # Quantile mapping for CIs 
    lat_lower = mu_lat - 1.96 * std_lat
    lat_upper = mu_lat + 1.96 * std_lat
    
    # transform to physical space 
    lower_ci = warper.inverse_transform(lat_lower)
    upper_ci = warper.inverse_transform(lat_upper)
    
    
    return Y_pred_mean, Y_pred_std, lower_ci, upper_ci

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
                 sample_date=None):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training
    ax.scatter(timestamps_train, Y_train.ravel(), c='blue', s=10, 
               label='Training Data', alpha=0.3)
    ax.plot(timestamps_train, Y_pred_train.ravel(), 'green', 
            label='Warped GP Fit', linewidth=1.5)
    ax.fill_between(timestamps_train, lower_train.ravel(), upper_train.ravel(),
                    alpha=0.2, color='green', label='95% CI (Asymmetric)')
    
    # Plot test
    ax.scatter(timestamps_test, Y_test.ravel(), c='orange', s=15,
               label='Test Data', alpha=0.6)
    ax.plot(timestamps_test, Y_pred_test.ravel(), 'red', 
            label='Warped GP Prediction', linewidth=1.5)
    ax.fill_between(timestamps_test, lower_test.ravel(), upper_test.ravel(),
                    alpha=0.2, color='red')
    
    # Separator
    ax.axvline(x=timestamps_train[-1], color='black', linestyle='--', linewidth=1.5)
    
    if sample_date:
        ax.axvline(x=pd.to_datetime(sample_date), color='purple', linestyle=':', lw=1.5)

    ax.set_title(f'Warped GP: WWTP Inflow Prediction', fontsize=14)
    ax.set_ylabel('Inflow (L/s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.show()

def plot_warped_distribution(target_date_str, timestamps, 
                             X_scaled, model, scaler_Y_warped, warper):
    """
    Generates a local histogram for a specific point without MC sampling. 
    """
    target_ts = pd.to_datetime(target_date_str)
    # Find index
    diffs = np.abs(timestamps - target_ts)
    idx = np.argmin(diffs)
    actual_ts = timestamps[idx]
    
    print(f"\n--- DISTRIBUTION PLOT ---")
    print(f"Target: {target_ts}, Actual: {actual_ts}")
    
    # 1. Get Latent Prediction for this specific point
    x_input = X_scaled[idx].reshape(1, -1)
    mean_lat_sc, var_lat_sc = model.predict_y(x_input)
    
    # 2. Unscale Latent parameters
    mu_lat = scaler_Y_warped.inverse_transform(mean_lat_sc.numpy())[0][0]
    std_lat = (np.sqrt(var_lat_sc.numpy()) * scaler_Y_warped.scale_)[0][0]
    
    # 3. Generate local samples for visualization (Latent -> Physical)
    rng = np.random.default_rng(42)
    lat_samples = rng.normal(mu_lat, std_lat, 10000)
    phys_samples = warper.inverse_transform(lat_samples)
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of samples
    ax.hist(phys_samples, bins=50, density=True, alpha=0.6, color='skyblue', label='Predictive Dist.')
    
    mean_val = np.mean(phys_samples)
    median_val = np.median(phys_samples)
    
    ax.axvline(mean_val, color='red', linestyle='-', lw=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', lw=2, label=f'Median: {median_val:.1f}')
    
    # Constraints
    ax.axvline(MIN_INFLOW, color='k', lw=3, label='Min Constraint')
    ax.axvline(MAX_INFLOW, color='k', lw=3, label='Max Constraint')
    
    ax.set_title(f"Predictive Distribution at {actual_ts}\n(Non-Gaussian Skew)", fontsize=14)
    ax.set_xlabel("Inflow (L/s)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

# Main Execution ------------------------------------------------------------------------------
def main():
    total_start = time.perf_counter()
    print("="*70) 
    print("WARPED Gaussian Process for WWTP Inflow Prediction")
    print("="*70)
    
    # Prepare Data
    X_train, Y_train_phys, timestamps_train = preparing_data(merged_train, train_start_time, timeinterval)
    
    # WARPING 
    # Transform Physical Y (0-180) -> Latent Y (-inf, inf)
    warper = LogitWarper(min_val=MIN_INFLOW, max_val=MAX_INFLOW)
    Y_train_warped = warper.transform(Y_train_phys)
    
    print("Warping Check:")
    print(f"Physical range: {Y_train_phys.min():.2f} to {Y_train_phys.max():.2f}")
    print(f"Warped range:   {Y_train_warped.min():.2f} to {Y_train_warped.max():.2f}")
    
    # Standard Scaling (On top of warped data for numerical stability)
    scaler_X = StandardScaler()
    scaler_Y_warped = StandardScaler() # Scales the Warped Y
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_final = scaler_Y_warped.fit_transform(Y_train_warped)
    
    # Training Model
    print("\nTraining GPR model on Warped Space...")
    model = build_gpr_model(X_train_scaled, Y_train_final, scaler_X.scale_[0])
    
    # Prediction (Train)
    print("Predicting on Training set...")
    Y_pred_train, _, lower_train, upper_train = prediction_warped(
        model, X_train_scaled, scaler_Y_warped, warper)
    
    # Prediction (Test)
    X_test, Y_test_phys, timestamps_test = preparing_data(merged_test, train_start_time, timeinterval)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"\nGenerating {test_hours}-hour predictions ...")
    Y_pred_test, std_test, lower_test, upper_test = prediction_warped(
        model, X_test_scaled, scaler_Y_warped, warper)
    
    # Evaluation
    train_metrics = model_evaluation(Y_train_phys, Y_pred_train, lower_train, upper_train)
    test_metrics = model_evaluation(Y_test_phys, Y_pred_test, lower_test, upper_test)
    
    results_df = pd.DataFrame({
        'Training Set': train_metrics,
        'Test Set': test_metrics})
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE (Warped GP)")
    print("="*50)
    print(results_df.round(4))
    print("="*50)
    
    total_time = time.perf_counter() - total_start
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print("Prediction complete!")
    # 8. Plotting
    sample_date = str(train_end_time + pd.Timedelta(hours=24))
    
    plot_results(timestamps_train, Y_train_phys, Y_pred_train, lower_train, upper_train,
                 timestamps_test, Y_test_phys, Y_pred_test, lower_test, upper_test,
                 sample_date=sample_date)
    
    plot_warped_distribution(sample_date, timestamps_test, X_test_scaled,
                             model, scaler_Y_warped, warper)

if __name__ == "__main__":
    main()
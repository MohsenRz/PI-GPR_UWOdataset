# making a Gaussian Process Regression model for WWTP data prediction
# sensor data is used 

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
train_start_time = pd.to_datetime("2019-01-01 00:00:00")
train_days = 7  # Number of days for training
train_end_time = train_start_time + pd.Timedelta(days=train_days)

# test parameters
test_hours = 12  # Hours to predict
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 10 # minutes 

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
# Corrected Merge Logic
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

def preparing_data(merged_data, start_time): 
    #creating multi dimensional input as timestamps and precipitation data
    timestamps = merged_data.index
    inflow_col = [col for col in merged_data.columns if 'inflow' in col.lower()][0]
    precip_col = [col for col in merged_data.columns if 'precipitation' in col.lower()][0]
    
    time = np.array([(ts - start_time).total_seconds() / 60.0 for ts in timestamps]).reshape(-1, 1)  # time in minutes
    rainfall = merged_data[precip_col].values.reshape(-1, 1)
    inflow = merged_data[inflow_col].values.reshape(-1, 1)
    
    X_multi = np.hstack((time, rainfall))
    Y = inflow
    
    X_train_original = X_multi.copy()
    timestamp_train = timestamps.copy()
    
    #standardised data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_multi)
    Scalar_Y = StandardScaler()
    Y_train_scaled = Scalar_Y.fit_transform(Y)
    
    return X_multi, Y, timestamps

def build_gpr_model(X_train, Y_train):
    # training the Gaussian process mdoel 
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 1.0], variance=1.0)
    kernel_noise = gpflow.kernels.White()
    kernel = kernel1 + kernel_noise
    
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    
    model = gpflow.models.GPR(data=(X_train_tf, Y_train_tf), 
                              kernel=kernel, mean_function=None)
    #optimise hyperparameters
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, 
                 model.trainable_variables, 
                 method='L-BFGS-B')
    
    print(f"\nOptimized kernel parameters:")
    print(f"Lengthscales: {model.kernel.kernels[0].lengthscales.numpy()}")
    print(f"Variance: {model.kernel.kernels[0].variance.numpy()}")
    print(f"Noise variance: {model.kernel.kernels[1].variance.numpy()}")
    
    return model


def model_evaluation(Y_true, Y_pred, std_pred, dataset_name=""):
    """
    Evaluate model performance
    """
    # Calculate metrics
    MSE = np.mean((Y_true.ravel() - Y_pred.ravel())**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_true.ravel() - Y_pred.ravel()))
    
    # Calculate coverage (95% confidence interval)
    lower_bound = Y_pred.ravel() - 1.96 * std_pred.ravel()
    upper_bound = Y_pred.ravel() + 1.96 * std_pred.ravel()
    points_inside = np.sum((Y_true.ravel() >= lower_bound) & 
                           (Y_true.ravel() <= upper_bound))
    coverage = points_inside / len(Y_true)
    
    print(f"\n{dataset_name} Evaluation:")
    print(f"RMSE: {RMSE:.4f}")
    print(f"MAE: {MAE:.4f}")
    print(f"Coverage (95% CI): {coverage*100:.1f}%")
    
    return RMSE, MAE, coverage 

def plot_results(timestamps_train, Y_train, Y_pred_train, std_train,
                 timestamps_test, Y_test, Y_pred_test, std_test):
    """
    Plot training and test results
    """
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    
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
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('WWTP Inflow', fontsize=14)
    ax.set_title(f'GPR: WWTP Inflow Prediction (Train: {train_days} days, Test: {test_hours} hours)', 
                 fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def main():
    total_start = time.time()
    print("="*70)
    print("GPR Model for WWTP Inflow Prediction")
    print("="*70)
    
    # Prepare training data
    X_train, Y_train, timestamps_train = preparing_data(merged_train, train_start_time)
    
    # Standardize training data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    # Build and train model
    print("\nTraining GPR model...")
    model = build_gpr_model(X_train_scaled, Y_train_scaled)
    
    # Predict on training data
    X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float64)
    mean_train, var_train = model.predict_f(X_train_tf)
    Y_pred_train_scaled = mean_train.numpy()
    std_train_scaled = np.sqrt(var_train.numpy())
    
    # Transform back to original scale
    Y_pred_train = scaler_Y.inverse_transform(Y_pred_train_scaled)
    std_train = std_train_scaled * scaler_Y.scale_
    
    # Evaluate on training data
    model_evaluation(Y_train, Y_pred_train, std_train, "Training Set")
    
    # Prepare test data (with actual precipitation)
    X_test, Y_test, timestamps_test = preparing_data(merged_test, train_start_time)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Predict on test data
    print(f"\nGenerating {test_hours}-hour predictions...")
    X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float64)
    mean_test, var_test = model.predict_f(X_test_tf)
    Y_pred_test_scaled = mean_test.numpy()
    std_test_scaled = np.sqrt(var_test.numpy())
    
    # Transform back to original scale
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_scaled)
    std_test = std_test_scaled * scaler_Y.scale_
    
    # Evaluate on test data
    model_evaluation(Y_test, Y_pred_test, std_test, "Test Set")
    
    # Plot results
    plot_results(timestamps_train, Y_train, Y_pred_train, std_train,
                 timestamps_test, Y_test, Y_pred_test, std_test)
    
    total_time = time.time() - total_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\n" + "="*70)
    print("Prediction complete!")
    print("="*70)

if __name__ == "__main__":
    main()


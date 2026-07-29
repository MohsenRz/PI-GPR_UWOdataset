""" 
making a persistence model for WWTP data prediction
sensor data is used 
the model only uses the previous data to predict the next data point
IT IS A REAL-TIME PREDICTIVE MODEL, NOT AN EXOGENEOUS MODEL 
Author: Mohsen 
created at: 29/07/2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import time

## Load Data
BASE = Path(__file__).parent.parent.parent
data_path = BASE / "data" / "RAW_data" / "pickled_data"

WWTP_inflow = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2021-01-01_to_2021-12-31.pkl")

WWTP_inflow['timestamp'] = pd.to_datetime(WWTP_inflow['timestamp'])

# train/test parameters (same as GPR script)
train_start_time = pd.to_datetime("2021-04-10 00:00:00")
train_days = 30
train_end_time = train_start_time + pd.Timedelta(days=train_days)

test_hours = 5 * 24
test_end_time = train_end_time + pd.Timedelta(hours=test_hours)
timeinterval = 15  # minutes

# NEW: real-world data latency -> how many intervals back is the "last known" value
latency_minutes = 15
lag_steps = max(1, int(round(latency_minutes / timeinterval)))  # e.g. 30/15 = 2

WWTP_inflow_train = WWTP_inflow[(WWTP_inflow['timestamp'] >= train_start_time) &
                                 (WWTP_inflow['timestamp'] <= train_end_time)]
WWTP_inflow_test = WWTP_inflow[(WWTP_inflow['timestamp'] > train_end_time) &
                                (WWTP_inflow['timestamp'] <= test_end_time)]

WWTP_inflow_train = WWTP_inflow_train.set_index('timestamp')
WWTP_inflow_test = WWTP_inflow_test.set_index('timestamp')

interval_string = f'{timeinterval}min'
WWTP_train_resampled = WWTP_inflow_train.resample(interval_string).mean()
WWTP_test_resampled = WWTP_inflow_test.resample(interval_string).mean()

inflow_col = WWTP_train_resampled.columns[0]

# Build one continuous series spanning train+test so the first test predictions
# can pull real observed values from the end of the training period.
full_series = pd.concat([WWTP_train_resampled[inflow_col],
                          WWTP_test_resampled[inflow_col]]).sort_index()
full_series = full_series[~full_series.index.duplicated(keep='first')]


def persistence_predict(series, target_index, lag_steps):
    """
    Operational persistence: prediction for time t uses the observed value
    from lag_steps back in the full (train+test) series -- i.e. the last
    value actually available given real-world data latency.
    """
    shifted = series.shift(lag_steps)
    return shifted.loc[target_index]


def model_evaluation(Y_true, Y_pred):
    valid = ~(Y_true.isna() | Y_pred.isna())
    y_t = Y_true[valid].values
    y_p = Y_pred[valid].values
    RMSE = np.sqrt(np.mean((y_t - y_p) ** 2))
    MAE = np.mean(np.abs(y_t - y_p))
    return {"RMSE (L/s)": RMSE, "MAE (L/s)": MAE, "N points": valid.sum()}


def plot_persistence(timestamps_train, Y_train, Y_pred_train,
                      timestamps_test, Y_test, Y_pred_test):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(timestamps_train, Y_train, c='blue', s=10, alpha=0.4,
               label='Training Data (Actual)', zorder=2)
    ax.plot(timestamps_train, Y_pred_train, color='green', linewidth=1,
            alpha=0.7, label='Persistence (Training)', zorder=3)

    ax.scatter(timestamps_test, Y_test, c='orange', s=15, alpha=0.7,
               label='Test Data (Actual)', zorder=2)
    ax.plot(timestamps_test, Y_pred_test, color='red', linewidth=2,
            label='Persistence Prediction (Test)', zorder=3)

    ax.axvline(x=timestamps_train[-1], color='black', linestyle='--',
               linewidth=1.5, label='Train/Test Split', zorder=4)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WWTP Inflow', fontsize=12)
    ax.set_title(f'Persistence Baseline (lag = {lag_steps} steps, '
                 f'{lag_steps * timeinterval} min)', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 70)
    total_start = time.perf_counter()
    print(f"Persistence Baseline (lag_steps={lag_steps}, "
          f"i.e. {lag_steps * timeinterval} min latency)")
    print("=" * 70)

    Y_train = WWTP_train_resampled[inflow_col]
    Y_test = WWTP_test_resampled[inflow_col]

    Y_pred_train = persistence_predict(full_series, Y_train.index, lag_steps)
    Y_pred_test = persistence_predict(full_series, Y_test.index, lag_steps)

    train_metrics = model_evaluation(Y_train, Y_pred_train)
    test_metrics = model_evaluation(Y_test, Y_pred_test)

    results_df = pd.DataFrame({'Training Set': train_metrics, 'Test Set': test_metrics})
    print("\n" + "=" * 50)
    print("PERSISTENCE MODEL PERFORMANCE")
    print("=" * 50)
    print(results_df.round(4))
    print("=" * 50)
    total_time = time.perf_counter() - total_start
    print(f"\nTotal execution time: {total_time:.1f} seconds")
        
    plot_persistence(Y_train.index, Y_train.values, Y_pred_train.values,
                      Y_test.index, Y_test.values, Y_pred_test.values)


if __name__ == "__main__":
    main()
"""
drawing CSO or tank level on time series with precipitation data on the same plot 
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

## Load Data
BASE = Path(__file__).parent

data_path = BASE / "RAW_data" / "pickled_data"

CSO = pd.read_pickle(
    data_path / "overflow_to_CSO" / "sensor_bf_plsRKBA1101_rubbasin_ara_2019-01-01_to_2019-12-31.pkl")
Tank_level = pd.read_pickle(
    data_path / "RB59_retention_tank_water_level" / "sensor_bl_plsRKBA1201_rubbasin_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

## Preprocess Data
Tank_level['timestamp'] = pd.to_datetime(Tank_level['timestamp'])
precipitation['timestamp'] = pd.to_datetime(precipitation['timestamp'])

plot_days = 20
start_date = pd.to_datetime("2019-01-01 00:00:00")

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'

# Filter data for the selected period
end_date = start_date + pd.Timedelta(days=plot_days)
inflow_filtered = Tank_level[(Tank_level['timestamp'] >= start_date) & 
                               (Tank_level['timestamp'] <= end_date)]
precip_filtered = precipitation[(precipitation['timestamp'] >= start_date) & 
                                 (precipitation['timestamp'] <= end_date)]

# Plot inflow on the left y-axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Tank Level (mm)', color=color)
ax1.plot(inflow_filtered['timestamp'], inflow_filtered['value'], 
         color=color, label='Tank Level', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Create a second y-axis for precipitation
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Precipitation (mm)', color=color)
ax2.bar(precip_filtered['timestamp'], precip_filtered['value'], 
        color=color, alpha=0.6, label='Precipitation', width=0.02)
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
fig.suptitle(f'Tank Level vs Precipitation ({start_date.date()} to {end_date.date()})', 
             fontsize=14, fontweight='bold')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()
plt.show()
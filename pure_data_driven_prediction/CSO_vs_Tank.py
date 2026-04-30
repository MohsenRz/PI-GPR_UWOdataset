"""
in this code I draw the plots showing CSO flow and Tank level for a period of time 
Author: Mohsen
Date: 06/02/2026
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
BASE = Path(__file__).parent.parent

data_path = BASE / "RAW_data" / "pickled_data"
Tank_level = pd.read_pickle(
    data_path / "RB59_retention_tank_water_level" / "sensor_bl_plsRKBA1201_rubbasin_ara_2019-01-01_to_2019-12-31.pkl")
CSO = pd.read_pickle(
    data_path / "overflow_to_CSO" / "sensor_bf_plsRKBA1101_rubbasin_ara_2019-01-01_to_2019-12-31.pkl")
precipitation = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

# Convert index to datetime if it isn't already
Tank_level['timestamp'] = pd.to_datetime(Tank_level['timestamp'])
CSO['timestamp'] = pd.to_datetime(CSO['timestamp'])

# Set timestamp as index
Tank_level = Tank_level.set_index('timestamp')
CSO = CSO.set_index('timestamp')

# time period 
start_time = "2019-01-01"
end_time = "2019-12-31"

# Filter data for the specified time period
tank_filtered = Tank_level[start_time:end_time]
cso_filtered = CSO[start_time:end_time]

print("\nFiltered tank_filtered shape:", tank_filtered.shape)
print("Filtered cso_filtered shape:", cso_filtered.shape)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot Tank Level on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Tank Water Level (m)', color=color)
line1 = ax1.plot(tank_filtered.index, tank_filtered['value'], color=color, linewidth=1.5, label='Tank Level')
ax1.tick_params(axis='y', labelcolor=color)

# Create secondary y-axis for CSO Flow
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('CSO Flow (L/s)', color=color)
line2 = ax2.plot(cso_filtered.index, cso_filtered['value'], color=color, linewidth=1.5, label='CSO Flow', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
fig.suptitle('Tank Water Level vs CSO Flow (2019)', fontsize=14, fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Improve layout
fig.tight_layout()
plt.show()
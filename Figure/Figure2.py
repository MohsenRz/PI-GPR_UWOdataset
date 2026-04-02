"""
drawing WWTP inflow on time series with precipitation data on the same plot 
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Set font type globally (change 'serif' to 'sans-serif', 'monospace', or a specific font like 'Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'  
plt.rcParams['font.size'] = 10

## Load Data
BASE = Path(__file__).parent.parent

data_path = BASE / "RAW_data" / "pickled_data"

WWTP_inflow2019 = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2019-01-01_to_2019-12-31.pkl")
precipitation2019 = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2019_cleaned.pkl")

WWTP_inflow2021 = pd.read_pickle(
    data_path / "inflow_WWTP" / "sensor_bf_plsZUL1100_inflow_ara_2021-01-01_to_2021-12-31.pkl")
precipitation2021 = pd.read_pickle(
    data_path / "precipitation" / "sensor_bn_r02_school_chatzenrainstr_2021_cleaned.pkl")

## Preprocess Data
WWTP_inflow2019['timestamp'] = pd.to_datetime(WWTP_inflow2019['timestamp'])
precipitation2019['timestamp'] = pd.to_datetime(precipitation2019['timestamp'])
WWTP_inflow2021['timestamp'] = pd.to_datetime(WWTP_inflow2021['timestamp'])
precipitation2021['timestamp'] = pd.to_datetime(precipitation2021['timestamp'])

plot_days = 35
start_date_2019 = pd.to_datetime("2019-09-30 00:00:00")
start_date_2021 = pd.to_datetime("2021-04-10 00:00:00")

# Create top and bottom plots
fig, (ax1_left, ax1_right) = plt.subplots(2, 1, figsize=(12, 12))

# ============== FIRST PLOT (2019) ==============
end_date_2019 = start_date_2019 + pd.Timedelta(days=plot_days)
inflow_filtered_2019 = WWTP_inflow2019[(WWTP_inflow2019['timestamp'] >= start_date_2019) & 
                                        (WWTP_inflow2019['timestamp'] <= end_date_2019)]
precip_filtered_2019 = precipitation2019[(precipitation2019['timestamp'] >= start_date_2019) & 
                                          (precipitation2019['timestamp'] <= end_date_2019)]

# Plot inflow on the left y-axis (2019)
color = 'tab:blue'
#ax1_left.set_xlabel('Time', fontsize=12, fontfamily='serif')
ax1_left.set_ylabel('WWTP Inflow (L/s)', color='black', fontsize=12, fontfamily='serif')
ax1_left.plot(inflow_filtered_2019['timestamp'], inflow_filtered_2019['value'], 
              color=color, label='WWTP Inflow', linewidth=2)
ax1_left.tick_params(axis='y', labelcolor='black')
ax1_left.grid(True, alpha=0.3)

# Create a second y-axis for precipitation (2019)
ax2_left = ax1_left.twinx()
color = 'tab:orange'
ax2_left.set_ylabel('Precipitation (mm)', color='black', fontsize=12, fontfamily='serif')
ax2_left.bar(precip_filtered_2019['timestamp'], precip_filtered_2019['value'], 
             color=color, alpha=0.6, label='Precipitation', width=0.02)
ax2_left.tick_params(axis='y', labelcolor='black')

# Add vertical line to show prediction zone (final 5 days) - 2019
prediction_start_2019 = end_date_2019 - pd.Timedelta(days=5)
ax1_left.axvline(x=prediction_start_2019, color='red', linestyle='--', linewidth=2, 
                 label='Prediction Zone', alpha=0.7)

# Add title - 2019
ax1_left.set_title(f'WWTP Inflow vs Precipitation \n{start_date_2019.date()} to {end_date_2019.date()}', 
                   fontsize=12, fontweight='bold', fontfamily='serif')

# Combine legends from both axes - 2019
lines1_left, labels1_left = ax1_left.get_legend_handles_labels()
lines2_left, labels2_left = ax2_left.get_legend_handles_labels()
ax1_left.legend(lines1_left + lines2_left, labels1_left + labels2_left, loc='upper left', 
                fontsize=10, prop={'family': 'serif'})

# Format x-axis for better spacing - 2019
ax1_left.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax1_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1_left.tick_params(axis='x', rotation=45)

# ============== SECOND PLOT (2021) ==============
end_date_2021 = start_date_2021 + pd.Timedelta(days=plot_days)
inflow_filtered_2021 = WWTP_inflow2021[(WWTP_inflow2021['timestamp'] >= start_date_2021) & 
                                        (WWTP_inflow2021['timestamp'] <= end_date_2021)]
precip_filtered_2021 = precipitation2021[(precipitation2021['timestamp'] >= start_date_2021) & 
                                          (precipitation2021['timestamp'] <= end_date_2021)]

# Plot inflow on the left y-axis (2021)
color = 'tab:blue'
ax1_right.set_xlabel('Time', fontsize=12, fontfamily='serif')
ax1_right.set_ylabel('WWTP Inflow (L/s)', color='black', fontsize=12, fontfamily='serif')
ax1_right.plot(inflow_filtered_2021['timestamp'], inflow_filtered_2021['value'], 
               color=color, label='WWTP Inflow', linewidth=2)
ax1_right.tick_params(axis='y', labelcolor='black')
ax1_right.grid(True, alpha=0.3)

# Create a second y-axis for precipitation (2021)
ax2_right = ax1_right.twinx()
color = 'tab:orange'
ax2_right.set_ylabel('Precipitation (mm)', color='black', fontsize=12, fontfamily='serif')
ax2_right.bar(precip_filtered_2021['timestamp'], precip_filtered_2021['value'], 
              color=color, alpha=0.6, label='Precipitation', width=0.02)
ax2_right.tick_params(axis='y', labelcolor='black')

# Add vertical line to show prediction zone (final 5 days) - 2021
prediction_start_2021 = end_date_2021 - pd.Timedelta(days=5)
ax1_right.axvline(x=prediction_start_2021, color='red', linestyle='--', linewidth=2, 
                  label='Prediction Zone', alpha=0.7)

# Add title - 2021
ax1_right.set_title(f'{start_date_2021.date()} to {end_date_2021.date()}', 
                    fontsize=12, fontweight='bold', fontfamily='serif')

# Combine legends from both axes - 2021
lines1_right, labels1_right = ax1_right.get_legend_handles_labels()
lines2_right, labels2_right = ax2_right.get_legend_handles_labels()
ax1_right.legend(lines1_right + lines2_right, labels1_right + labels2_right, loc='upper left', 
                 fontsize=10, prop={'family': 'serif'})

# Format x-axis for better spacing - 2021
ax1_right.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax1_right.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1_right.tick_params(axis='x', rotation=45)

fig.tight_layout()

# Save the combined figure to current folder
output_path = Path(__file__).parent / "WWTP_Inflow_vs_Precipitation_Combined.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined figure saved to: {output_path}")

plt.show()


"""
I get the data from the SWMM model for a zero-rain period to get the DWF mean function 
the output is imported to the main GPR model as a mean function
Author: Mohsen
Date: 24/12/2025
"""

import pyswmm
from pyswmm import Simulation, Nodes, Links
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

start_time = datetime.datetime(2018, 2, 1, 0, 0)
end_time = datetime.datetime(2018, 2, 15, 0, 0)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sim_file = os.path.join(script_dir, 'faf_for_pyswmm.inp')

with Simulation(sim_file) as sim:
    sim.start_time = start_time
    sim.end_time = end_time
    
    flow_units = sim.flow_units
    system_units = sim.system_units
    print(f"Flow units: {flow_units}, System units: {system_units}")
    
    simtime = []
    WWTP = Nodes(sim)["ARA_Fehraltorf"]
    WWTP_inflow = []
    
    timeinterval = 180 # seconds 
    for step in enumerate(sim):
        current_time = sim.current_time
        simtime.append(current_time)
        WWTP_inflow.append(WWTP.total_inflow)
        sim.step_advance(timeinterval)  # advance by 180 seconds
        
# Create a DataFrame to hold the results
data = pd.DataFrame({
    'Time': simtime,
    'WWTP_Inflow': WWTP_inflow
})

# plotting the results 
plt.figure(figsize=(12, 6))
plt.plot(data['Time'], data['WWTP_Inflow'], label='WWTP Inflow (lps)', color='blue')
plt.xlabel('Time')  
plt.ylabel('Inflow (lps)')
plt.title('WWTP Inflow Over Time During Zero-Rain Period')
plt.legend()
plt.grid()
plt.show()

# creating an average mean function for one day 
timesteps_per_day = int(24 * 3600 / timeinterval)  # number of timesteps in one day
num_days = len(data) // timesteps_per_day
mean_function = np.zeros(timesteps_per_day)
for day in range(num_days):
    start_idx = day * timesteps_per_day
    end_idx = start_idx + timesteps_per_day
    mean_function += data['WWTP_Inflow'].iloc[start_idx:end_idx].values
mean_function /= num_days
# save as a CSV file 
mean_df = pd.DataFrame({
    'Time_Seconds': np.arange(0, timesteps_per_day * timeinterval, timeinterval),
    'Mean_WWTP_Inflow': mean_function
})
mean_df.to_csv(os.path.join(script_dir, 'DWF_Mean_Function_WWTP_in_Seconds.csv'), index=False)
print("DWF Mean function saved to 'DWF_Mean_Function_WWTP_in_Seconds.csv'")

# plotting the mean function
plt.figure(figsize=(12, 6))
plt.plot(mean_df['Time_Seconds'], mean_df['Mean_WWTP_Inflow'], label='Mean WWTP Inflow (lps)', color='green')
plt.xlabel('Time (seconds)')    
plt.ylabel('Mean Inflow (lps)')
plt.title('DWF Mean Function for WWTP Inflow Over One Day')
plt.legend()
plt.grid()
plt.show()

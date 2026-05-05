"""
I want to remove anomalies from precipitation data.
This script loads the data, removes anomalies, and saves the cleaned data."""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_pickle(r'RAW_data\pickled_data\precipitation\sensor_bn_r02_school_chatzenrainstr_2021-01-01_to_2021-12-31.pkl')
replacing_data = pd.read_pickle(r'RAW_data\pickled_data\precipitation\airport\sensor_bn_r04_airport_speck_2021-01-01_to_2021-12-31.pkl')

data['timestamp'] = pd.to_datetime(data['timestamp'])
replacing_data['timestamp'] = pd.to_datetime(replacing_data['timestamp'])
start_time = pd.to_datetime("2021-01-01 00:00:00")
end_time = pd.to_datetime("2021-12-30 23:59:59")
data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
replacing_data = replacing_data[(replacing_data['timestamp'] >= start_time) & (replacing_data['timestamp'] <= end_time)]
# list the negative values 
anomalies = data[(data['value'] < 0) | (data['value'] > 200)]
print("Anomalies found:")

# replace the anomalies with the data from the replacing_data dataframe
for index, row in anomalies.iterrows():
    timestamp = row['timestamp']
    replacement_value = replacing_data[replacing_data['timestamp'] == timestamp]['value']
    if not replacement_value.empty:
        data.at[index, 'value'] = replacement_value.values[0]
        print(f"Replaced anomaly at {timestamp} with value {replacement_value.values[0]}")
    elif replacement_value.empty:
        print(f"No replacement found for anomaly at {timestamp}, leaving original value {row['value']}")
# Save the cleaned data
data.to_pickle(r'RAW_data\pickled_data\precipitation\sensor_bn_r02_school_chatzenrainstr_2021_cleaned.pkl')

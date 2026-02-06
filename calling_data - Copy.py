import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_pickle(r'RAW_data\pickled_data\RB59_retention_tank_water_level\sensor_bl_plsRKBA1201_rubbasin_ara_2019-01-01_to_2019-12-31.pkl')
datatype = "level" 
data['timestamp'] = pd.to_datetime(data['timestamp'])
start_time = pd.to_datetime("2019-01-01 00:00:00")
end_time = pd.to_datetime("2019-12-30 23:59:59")
data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

if datatype == "flow":
    unit = "L/s"
elif datatype == "level":
    unit = "mm"
else:
    unit = "unknown"

def plot(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label=datatype, color='blue')
    plt.xlabel('Date')
    plt.ylabel(f'{datatype} ({unit})')
    plt.title('Water Level in Retention Tank (2019)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    data.set_index('timestamp', inplace=True)
    plot(data)
if __name__ == "__main__":
    main()

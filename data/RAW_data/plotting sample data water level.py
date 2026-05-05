import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

# --- CONFIGURATION ---
# !!! Change this to the actual path of your SQLite file
DB_FILE = 'data_uwo_2019.sqlite' 
TABLE_NAME = 'signal' # The table name you confirmed

# IDs you want to filter by
VARIABLE_ID = 6 # variable_id=4 for flow, =6 for water level
SOURCE_ID = 199   # source_id: the sensor's ID    ### get it from SOURCE dataset 


# The time period you want to query
START_TIME = '2019-06-16 00:00:00'
END_TIME = '2019-06-22 23:59:00'
# --- END CONFIGURATION ---

def fetch_data_by_source(db_path, table, variable, source, start, end):
    """
    Connects to the SQLite database and fetches data based on
    variable_id and source_id for a specific time period.
    """
    print(f"Connecting to database: {db_path}...")
    try:
        # Create the database connection
        with sqlite3.connect(db_path) as conn:
            # Construct the SQL query
            query = f"""
                SELECT 
                    timestamp, 
                    value, 
                    site_id 
                FROM 
                    {table}
                WHERE 
                    variable_id = ?
                AND 
                    source_id = ?
                AND 
                    timestamp >= ?
                AND 
                    timestamp < ?
                ORDER BY 
                    timestamp;
            """
            
            # Use pandas to execute the query and load data into a DataFrame
            # Pass the variables as parameters to the query
            params = (variable, source, start, end)
            df = pd.read_sql_query(query, conn, params=params)
            
            print(f"Successfully fetched {len(df)} records for variable_id={variable} & source_id={source}.")
            return df

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        if "no such table" in str(e):
            print(f"HINT: Check if the TABLE_NAME '{table}' is correct.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        print(f"HINT: Make sure the file path '{db_path}' is correct.", file=sys.stderr)
        return None

def plot_data(df, variable, source):
    """
    Plots the fetched data, creating a separate line for each
    unique site (site_id) found.
    """
    if df.empty:
        print(f"No data found for variable_id={variable} & source_id={source} in this time range.")
        return

    print("Generating plot...")
    
    # Ensure the timestamp column is in datetime format for plotting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- ADDED: print maximum value and its details ---
    if df['value'].notna().any():
        #MAX
        idx = df['value'].idxmax()
        max_row = df.loc[idx]
        max_value = max_row['value']
        max_time = pd.to_datetime(max_row['timestamp'])
        max_site = max_row.get('site_id', 'N/A')
        print(f"Maximum value: {max_value} at {max_time} (site_id={max_site})")
        
        #MIN
        idx_min = df['value'].idxmin()
        min_row = df.loc[idx_min]
        min_value = min_row['value']
        min_time = pd.to_datetime(min_row['timestamp'])
        min_site = min_row.get('site_id', 'N/A')
        print(f"Minimum value: {min_value} at {min_time} (site_id={min_site})")
    else:
        print("No numeric 'value' data available to compute maximum.")
    # --- END ADDED ---
    
    # Find all unique sites that reported data
    unique_sites = df['site_id'].unique()
    
    plt.figure(figsize=(14, 7))
    
    # Plot data for each site
    for site in unique_sites:
        site_data = df[df['site_id'] == site]
        plt.plot(site_data['timestamp'], site_data['value'], label=f'Site: {site}')
    
    # The PDF (Table S2) says flow 'bf' is in [l/s].
    unit = "mm" 
    plt.title(f'Water Level for Source ID: {source} (Variable: {variable})')
    plt.xlabel('Timestamp')
    plt.ylabel(f'Water Level ({unit})')
    
    # Format the x-axis to show dates nicely
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate() # Auto-rotate dates
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Main execution
if __name__ == "__main__":
    # 1. Fetch the data using the new function
    data_df = fetch_data_by_source(DB_FILE, TABLE_NAME, VARIABLE_ID, SOURCE_ID, START_TIME, END_TIME)
    
    # 2. If data was fetched successfully, plot it
    if data_df is not None:
        plot_data(data_df, VARIABLE_ID, SOURCE_ID)
    else:
        print("Failed to fetch data. Exiting.", file=sys.stderr)
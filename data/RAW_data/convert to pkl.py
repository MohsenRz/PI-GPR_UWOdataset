import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os # <-- ADDED IMPORT

# --- CONFIGURATION ---
# !!! Change this to the actual path of your SQLite file
DB_FILE = 'data_uwo_2019.sqlite' 
TABLE_NAME = 'signal' # The table name you confirmed
OUTPUT_DIR = 'pickled_data' # <-- ADDED: Directory to store .pkl files

# IDs you want to filter by
Snesor_name = "bn_r04_airport_speck" 
VARIABLE_ID = 20 # variable_id=4 for flow, =6 for water level, =8 for bucket content, =20 rain intensity
SOURCE_ID = 4   # source_id: the sensor's ID      ### get it from SOURCE dataset 


# The time period you want to query
START_TIME = '2019-01-01 00:00:00'
END_TIME = '2019-12-31 23:59:59'
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
    
    # Find all unique sites that reported data
    unique_sites = df['site_id'].unique()
    
    plt.figure(figsize=(14, 7))
    
    # Plot data for each site
    for site in unique_sites:
        site_data = df[df['site_id'] == site]
        plt.plot(site_data['timestamp'], site_data['value'], label=f'Site: {site}')
    
    if variable==6 : 
        unit = "mm" 
        plt.title(f'Water Level for Source ID: {source} (Variable: {variable})')
        plt.ylabel(f'water level ({unit})')
    elif variable==4 :
        unit = "L/s" 
        plt.title(f'Flow Rate for Source ID: {source} (Variable: {variable})')
        plt.ylabel(f'flow rate ({unit})')
    else:
        plt.title(f'Data for Source ID: {source} (Variable: {variable})')
        plt.ylabel('Value')
    plt.xlabel('Timestamp')
    
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
        
        # --- NEW: SAVE TO .PKL FILE ---
        print("\n--- Saving data to .pkl file ---")
        
        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create a descriptive filename
        start_date = START_TIME[:10]
        end_date = END_TIME[:10]
        filename = f"sensor_{Snesor_name}_{start_date}_to_{end_date}.pkl"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            # Save the DataFrame to a pickle file
            data_df.to_pickle(output_path)
            
            # Check the file size
            size_bytes = os.path.getsize(output_path)
            size_mb = size_bytes / (1024 * 1024)
            
            print(f"Successfully saved data to: {output_path}")
            print(f"File size: {size_mb:.4f} MB ({size_bytes} bytes)")
            
        except Exception as e:
            print(f"Error saving pickle file: {e}", file=sys.stderr)
            
        print("---------------------------------\n")
        # --- END NEW SECTION ---
        
        # 3. Plot the data
        plot_data(data_df, VARIABLE_ID, SOURCE_ID)
    else:
        print("Failed to fetch data. Exiting.", file=sys.stderr)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import List, Dict, Optional, Tuple

# --- Configuration and Initialization ---
st.set_page_config(
    page_title="🔋 Battery Swap & Power Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. UPDATED CONSTANTS ---
SOC_BINS = [-float('inf'), 9.999, 30, 50, 70, 85, 100]
SOC_BUCKETS = [
    'Less than 10%',
    'Between 10 - 30%',
    'Between 30.1- 50%',
    'Between 50.1 - 70%',
    'Between 70.1 - 85%',
    'Between 85.1 - 100%'
]
PREDEF_BP_COUNTS = [1, 2, 3] # Assumes max 4 BPs per swap
TIME_RANGE_HOURS = list(range(7, 23)) # 7 AM to 11 PM
SOC_TOLERANCE = 0.10 # 10% tolerance

# --- Column Mapping ---
SOC_COL_MAP = {
    'Swap Start Time': 'Timestamp',
    'Station Id': 'Station_Name',
    'Vehicle Id': 'Vehicle_ID',
    'Received Batteries - BPID': 'BP_ID',
    'Incoming BP SOC': 'Incoming_SOC'
}
POWER_COL_MAP = {
    'Date': 'Timestamp',
    'Hour': 'Hour_of_Day',
    'Station ID': 'Station_Name',
    'Station Energy': 'Cumulative_Power_kWh'
}

# --- 2. Data Loading and Preprocessing (Unchanged) ---

@st.cache_data(show_spinner="Loading and merging data...")
def load_and_merge_data(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], file_type: str) -> Optional[pd.DataFrame]:
    if not uploaded_files: return None
    all_dfs = [pd.read_csv(file) for file in uploaded_files if file]
    if not all_dfs: return None
    merged_df = pd.concat(all_dfs, ignore_index=True)
    st.success(f"Successfully loaded and merged {len(all_dfs)} {file_type} files.")
    return merged_df

def preprocess_soc_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.rename(columns=SOC_COL_MAP, inplace=True)
    required_cols = list(SOC_COL_MAP.values())
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"**SOC Data Error:** Missing required columns after renaming: {missing}.")
        return None

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df['Date'] = df['Timestamp'].dt.date
    df = df[df['Hour'].between(TIME_RANGE_HOURS[0], TIME_RANGE_HOURS[-1])]
    
    df.rename(columns={'Incoming_SOC': 'SOC'}, inplace=True)
    df['SOC'] = pd.to_numeric(df['SOC'], errors='coerce')
    df.dropna(subset=['SOC'], inplace=True)
    return df

def preprocess_power_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.rename(columns=POWER_COL_MAP, inplace=True)
    required_cols = list(POWER_COL_MAP.values())
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"**Power Data Error:** Missing required columns after renaming: {missing}.")
        return None

    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'].astype(str) + ' ' + df['Hour_of_Day'].astype(str) + ':00:00', errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        df['Hour'] = df['Timestamp'].dt.hour
        df['Date'] = df['Timestamp'].dt.date
    except Exception as e:
        st.error(f"Error creating Timestamp from Date/Hour in Power Data: {e}")
        return None

    df = df[df['Hour'].between(TIME_RANGE_HOURS[0], TIME_RANGE_HOURS[-1])]
    df.rename(columns={'Cumulative_Power_kWh': 'Cumulative_Power'}, inplace=True)
    df['Cumulative_Power'] = pd.to_numeric(df['Cumulative_Power'], errors='coerce')
    df.dropna(subset=['Cumulative_Power'], inplace=True)
    return df

# --- 3. Core Analysis Logic (Unchanged) ---

def apply_swap_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies logic to calculate SOC, determines BP count, and buckets swaps.
    """
    swap_groups = df.groupby(['Date', 'Hour', 'Station_Name', 'Vehicle_ID'])

    def calculate_soc_and_bp_count(group):
        socs = group['SOC'].values
        bp_count = len(socs)
        max_soc = np.max(socs)
        min_soc = np.min(socs)
        
        if (max_soc - min_soc) / 100.0 > SOC_TOLERANCE:
            final_soc = np.mean(socs)
        else:
            final_soc = min_soc
            
        return pd.Series({'Representative_SOC': final_soc, 'BP_Count': bp_count})

    vehicle_soc_metrics = swap_groups.apply(calculate_soc_and_bp_count).reset_index()

    vehicle_soc_metrics['SOC_Bucket'] = pd.cut(
        vehicle_soc_metrics['Representative_SOC'],
        bins=SOC_BINS,
        labels=SOC_BUCKETS,
        right=True,
        include_lowest=True
    ).astype(str)

    hourly_swaps = vehicle_soc_metrics.groupby(
        ['Date', 'Hour', 'Station_Name', 'BP_Count', 'SOC_Bucket']
    ).size().reset_index(name='Swap_Count')
    
    return hourly_swaps


def calculate_hourly_power(df: pd.DataFrame) -> pd.DataFrame:
    """ Calculates hourly power consumption (unchanged). """
    hourly_readings = df.loc[df.groupby(['Date', 'Hour', 'Station_Name'])['Timestamp'].idxmin()]
    hourly_readings.sort_values(by=['Station_Name', 'Date', 'Hour'], inplace=True)
    hourly_readings['Prev_Cumulative_Power'] = hourly_readings.groupby(['Station_Name'])['Cumulative_Power'].shift(1)
    hourly_readings['Power_Consumed'] = hourly_readings['Cumulative_Power'] - hourly_readings['Prev_Cumulative_Power']
    hourly_readings.dropna(subset=['Prev_Cumulative_Power'], inplace=True)
    hourly_power = hourly_readings[['Date', 'Hour', 'Station_Name', 'Power_Consumed']].copy()
    hourly_power.rename(columns={'Power_Consumed': 'Power_Consumed_kWh'}, inplace=True)
    return hourly_power

# --- 4. Final Aggregation and Merging (Unchanged) ---
# This function already provides the 'Date' column needed for splitting.

def get_all_swap_columns() -> List[str]:
    """Generates a list of all possible swap column names for the template."""
    cols = []
    for bp in PREDEF_BP_COUNTS:
        for bucket in SOC_BUCKETS:
            cols.append(f"{bp}BP_{bucket}")
    return cols

def combine_analysis_results(
    hourly_swaps_df: pd.DataFrame, 
    hourly_power_df: pd.DataFrame, 
    selected_stations: List[str], 
    selected_dates: List[pd.Timestamp]
) -> Dict[str, pd.DataFrame]:
    """
    Combines data and returns a dictionary of hourly dataframes.
    Each dataframe contains hourly data AND a 'Total' row for each date.
    """
    if hourly_swaps_df is None and hourly_power_df is None:
        return {}

    all_dates = [pd.to_datetime(d).date() for d in selected_dates]
    
    # --- SWAP DATA PREP ---
    all_swap_cols = get_all_swap_columns()
    if hourly_swaps_df is not None:
        hourly_swaps_df['Date'] = pd.to_datetime(hourly_swaps_df['Date']).dt.date
        hourly_swaps_df = hourly_swaps_df[
            (hourly_swaps_df['Date'].isin(all_dates)) & 
            (hourly_swaps_df['Station_Name'].isin(selected_stations))
        ]
        pivoted_swaps = hourly_swaps_df.pivot_table(
            index=['Date', 'Hour', 'Station_Name'],
            columns=['BP_Count', 'SOC_Bucket'],
            values='Swap_Count',
            fill_value=0
        )
        pivoted_swaps.columns = [f"{int(bp_count)}BP_{soc_bucket}" for bp_count, soc_bucket in pivoted_swaps.columns.values]
        pivoted_swaps.reset_index(inplace=True)
    else:
        pivoted_swaps = pd.DataFrame(columns=['Date', 'Hour', 'Station_Name'] + all_swap_cols)
    
    # --- POWER DATA PREP (Unchanged) ---
    if hourly_power_df is not None:
        hourly_power_df['Date'] = pd.to_datetime(hourly_power_df['Date']).dt.date
        hourly_power_df = hourly_power_df[
            (hourly_power_df['Date'].isin(all_dates)) & 
            (hourly_power_df['Station_Name'].isin(selected_stations))
        ]
    else:
        hourly_power_df = pd.DataFrame(columns=['Date', 'Hour', 'Station_Name', 'Power_Consumed_kWh'])

    # --- MERGE (Unchanged) ---
    if not pivoted_swaps.empty and not hourly_power_df.empty:
        combined_df = pd.merge(pivoted_swaps, hourly_power_df, on=['Date', 'Hour', 'Station_Name'], how='outer')
    elif not pivoted_swaps.empty:
        combined_df = pivoted_swaps
    elif not hourly_power_df.empty:
        combined_df = hourly_power_df
    else:
        combined_df = pd.DataFrame()

    # --- ZERO-FILLING & TEMPLATE CREATION ---
    hourly_dfs: Dict[str, pd.DataFrame] = {}
    fill_cols = ['Power_Consumed_kWh'] + all_swap_cols

    for station in selected_stations:
        template_data = []
        for date in all_dates:
            for hour in TIME_RANGE_HOURS:
                template_data.append({'Date': date, 'Hour': hour, 'Station_Name': station})
        station_template = pd.DataFrame(template_data)
        
        station_df = combined_df[combined_df['Station_Name'] == station]
        
        final_df = pd.merge(station_template, station_df.drop(columns=['Station_Name'], errors='ignore'), 
                            on=['Date', 'Hour'], how='left')
        
        for col in fill_cols:
            if col not in final_df.columns:
                final_df[col] = 0
            final_df[col] = final_df[col].fillna(0).astype(np.float64)

        final_df['Time Slot'] = final_df['Hour'].apply(lambda h: f"{h:02d}:00 - {h+1:02d}:00")
        
        # --- Build table with totals for EACH date ---
        all_date_chunks = []
        sorted_dates = sorted(all_dates)
        
        for date in sorted_dates:
            date_df = final_df[final_df['Date'] == date].sort_values(by='Hour')
            total_row = date_df[fill_cols].sum().to_frame().T
            total_row['Time Slot'] = 'Total'
            total_row['Date'] = date 
            total_row['Hour'] = 99
            all_date_chunks.append(pd.concat([date_df, total_row], ignore_index=True))
        
        if all_date_chunks:
            hourly_dfs[station] = pd.concat(all_date_chunks, ignore_index=True)
            hourly_dfs[station]['Date'] = pd.to_datetime(hourly_dfs[station]['Date']).dt.strftime('%Y-%m-%d')
            # Keep 'Hour' for visualizations, but 'Date' is now the key for splitting
        else:
            hourly_dfs[station] = pd.DataFrame()

    return hourly_dfs


# --- 5. Visualization Functions (Unchanged) ---

def create_power_chart(df: pd.DataFrame, station_name: str) -> alt.Chart:
    """Creates an Altair line chart for hourly power consumption."""
    chart_df = df[df['Time Slot'] != 'Total'].copy()
    chart_df = chart_df.groupby('Time Slot')['Power_Consumed_kWh'].mean().reset_index()
    time_slot_order = [f"{h:02d}:00 - {h+1:02d}:00" for h in TIME_RANGE_HOURS]
    
    base = alt.Chart(chart_df).encode(
        x=alt.X('Time Slot:N', sort=time_slot_order, title="Time Slot"),
        tooltip=[
            'Time Slot:N',
            alt.Tooltip('Power_Consumed_kWh:Q', format='.2f', title='Avg Power (kWh)')
        ]
    ).properties(
        title=f"Avg. Hourly Power Consumption - {station_name}",
        height=300
    )
    line_chart = base.mark_line(point=True).encode(
        y=alt.Y('Power_Consumed_kWh:Q', title="Power Consumed (kWh)"),
        color=alt.value("#4c78a8")
    )
    return (line_chart).interactive()

def create_swap_chart(df: pd.DataFrame, station_name: str) -> alt.Chart:
    """Creates an Altair faceted bar chart for hourly swap distribution."""
    chart_df = df[df['Time Slot'] != 'Total'].copy()
    swap_cols = get_all_swap_columns()
    chart_df = chart_df.groupby('Time Slot')[swap_cols].sum().reset_index()
    chart_long = chart_df.melt(
        id_vars=['Time Slot'],
        value_vars=swap_cols,
        var_name='Metric',
        value_name='Swap_Count'
    )
    chart_long = chart_long[chart_long['Swap_Count'] > 0]
    chart_long['BP_Count'] = chart_long['Metric'].str.extract(r'(\d+)BP_').astype(int)
    chart_long['SOC_Bucket'] = chart_long['Metric'].str.replace(r'^\d+BP_', '')
    time_slot_order = [f"{h:02d}:00 - {h+1:02d}:00" for h in TIME_RANGE_HOURS]

    chart = alt.Chart(chart_long).mark_bar().encode(
        x=alt.X('Time Slot:N', sort=time_slot_order, title="Time Slot", axis=None),
        y=alt.Y('Swap_Count:Q', title="Total Swap Count"),
        color=alt.Color('SOC_Bucket:N', title="SOC Range", legend=alt.Legend(orient="bottom")),
        column=alt.Column(
            'BP_Count:N', 
            title="Swaps by Battery Pack (BP) Count",
            header=alt.Header(titleOrient="bottom", labelOrient="bottom")
        ),
        tooltip=[
            'Time Slot:N',
            'SOC_Bucket:N',
            'BP_Count:N',
            'Swap_Count:Q'
        ]
    ).properties(
        title=f"Hourly Swap Distribution - {station_name}"
    ).interactive()
    
    return chart


# --- 6. Streamlit Dashboard App (DISPLAY SECTION UPDATED) ---

def main():
    st.title("🔋 Professional Battery Swap and Power Data Analysis Dashboard")
    st.markdown("Analyzes hourly power (kWh) and swap distribution by **SOC Bucket** and **BP Count**.")
    st.markdown("---")

    # 1. File Upload Section
    with st.sidebar:
        st.header("1. Data Upload (CSV)")
        st.info("The code automatically maps your custom columns for processing.")
        soc_files = st.file_uploader("Upload **Vehicle SOC CSV** files", type=['csv'], accept_multiple_files=True, key='soc_uploader')
        power_files = st.file_uploader("Upload **Power Log CSV** files", type=['csv'], accept_multiple_files=True, key='power_uploader')

    # Load and Preprocess Data
    soc_df_raw = load_and_merge_data(soc_files, "SOC/Swap")
    power_df_raw = load_and_merge_data(power_files, "Power Log")
    
    soc_df = preprocess_soc_data(soc_df_raw.copy()) if soc_df_raw is not None else None
    power_df = preprocess_power_data(power_df_raw.copy()) if power_df_raw is not None else None

    # Determine available stations and dates
    available_stations = set()
    available_dates = set()
    if soc_df is not None:
        available_stations.update(soc_df['Station_Name'].unique())
        available_dates.update(soc_df['Date'].unique())
    if power_df is not None:
        available_stations.update(power_df['Station_Name'].unique())
        available_dates.update(power_df['Date'].unique())

    all_stations = sorted(list(available_stations))
    all_dates = sorted(list(available_dates))
    
    # 2. Filters Section
    with st.sidebar:
        st.header("2. Analysis Filters")
        selected_stations = st.multiselect("Select Station(s)", options=all_stations, default=all_stations[0] if all_stations else [])
        selected_dates_raw = st.multiselect("Select Date(s)", options=all_dates, default=all_dates if all_dates else [])
        st.info(f"**Fixed Time Range:** {TIME_RANGE_HOURS[0]:02d}:00 to {TIME_RANGE_HOURS[-1]+1:02d}:00")

    if not selected_stations or not selected_dates_raw:
        st.warning("Please upload data and select at least one station and one date to proceed.")
        return

    # --- Core Analysis and Merging ---
    
    hourly_swaps_df = apply_swap_logic(soc_df.copy()) if soc_df is not None else None
    hourly_power_df = calculate_hourly_power(power_df.copy()) if power_df is not None else None
            
    if hourly_swaps_df is None and hourly_power_df is None:
        st.error("Cannot perform analysis. Check data upload.")
        return

    with st.spinner("Finalizing Dashboard Data..."):
        final_hourly_data = combine_analysis_results(
            hourly_swaps_df, hourly_power_df, selected_stations, selected_dates_raw
        )

    # --- 3 & 4. Output Display and Visualization (UPDATED) ---
    
    st.header("📊 Hourly Station Data Analysis")
    st.markdown("---")

    for station_name, df_output in final_hourly_data.items():
        if df_output.empty:
            st.error(f"No data found for station: **{station_name}** in the selected date range.")
            continue
            
        st.subheader(f"Station: {station_name}")
        
        # --- NEW LOGIC: Iterate and display a table for each date ---
        all_dates_in_df = df_output['Date'].unique()
        
        for date_str in all_dates_in_df:
            if date_str is pd.NaT or date_str is None:
                continue

            st.markdown(f"#### Hourly Summary for Date: {date_str}")
            
            # Filter the main dataframe for just this date
            date_specific_df = df_output[df_output['Date'] == date_str].copy()

            # Rename columns for presentation
            display_df = date_specific_df.copy()
            display_df.rename(columns={'Power_Consumed_kWh': 'Power (kWh)'}, inplace=True)
            for bp in PREDEF_BP_COUNTS:
                for bucket in SOC_BUCKETS:
                    display_df.rename(columns={f"{bp}BP_{bucket}": f"{bp}BP: {bucket}"}, inplace=True)
            
            # Define columns to show (now without 'Date')
            final_display_cols = ['Time Slot', 'Power (kWh)'] + [f"{bp}BP: {bucket}" for bp in PREDEF_BP_COUNTS for bucket in SOC_BUCKETS]
            final_display_cols_exist = [col for col in final_display_cols if col in display_df.columns]
            
            st.dataframe(display_df[final_display_cols_exist], use_container_width=True, hide_index=True)
        # --- END OF NEW TABLE LOGIC ---

        
        # 4. Visualization
        # Visuals show the aggregate for the *entire* selection
        #st.markdown("### Visual Insights (Aggregated for all selected dates)")
        
        #power_chart = create_power_chart(df_output, station_name)
       # st.altair_chart(power_chart, use_container_width=True)

       # swap_chart = create_swap_chart(df_output, station_name)
       # st.altair_chart(swap_chart, use_container_width=True)
        
        #st.markdown("---")

if __name__ == "__main__":
    main()
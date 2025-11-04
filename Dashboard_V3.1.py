import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import List, Dict, Optional, Tuple
# No AgGrid import needed
import os # <-- To check if cache files exist
import io  # <-- To prepare files for download

# --- Configuration and Initialization ---
st.set_page_config(
    page_title="🔋 Battery Swap & Power Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CONSTANTS ---
SOC_BINS = [-float('inf'), 9.999, 30, 50, 70, 85, 100]
SOC_BUCKETS = [
    'Less than 10%', 'Between 10 - 30%', 'Between 30.1- 50%',
    'Between 50.1 - 70%', 'Between 70.1 - 85%', 'Between 85.1 - 100%'
]
PREDEF_BP_COUNTS = [1, 2, 3] 
TIME_RANGE_HOURS = list(range(7, 24))
SOC_TOLERANCE = 0.10

# --- CACHE FILENAMES ---
SWAP_CACHE_FILE = "processed_swaps.parquet"
POWER_CACHE_FILE = "processed_power.parquet"

# --- Column Mapping ---
SOC_COL_MAP = {
    'Swap Start Time': 'Timestamp', 'Station Id': 'Station_Name',
    'Vehicle Id': 'Vehicle_ID', 'Received Batteries - BPID': 'BP_ID',
    'Incoming BP SOC': 'Incoming_SOC'
}
POWER_COL_MAP = {
    'Date': 'Timestamp', 'Hour': 'Hour_of_Day',
    'Station ID': 'Station_Name', 'Station Energy': 'Cumulative_Power_kWh'
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
        bins=SOC_BINS, labels=SOC_BUCKETS, right=True, include_lowest=True
    ).astype(str)
    hourly_swaps = vehicle_soc_metrics.groupby(
        ['Date', 'Hour', 'Station_Name', 'BP_Count', 'SOC_Bucket']
    ).size().reset_index(name='Swap_Count')
    return hourly_swaps

def calculate_hourly_power(df: pd.DataFrame) -> pd.DataFrame:
    hourly_readings = df.loc[df.groupby(['Date', 'Hour', 'Station_Name'])['Timestamp'].idxmin()]
    hourly_readings.sort_values(by=['Station_Name', 'Date', 'Hour'], inplace=True)
    hourly_readings['Prev_Cumulative_Power'] = hourly_readings.groupby(['Station_Name'])['Cumulative_Power'].shift(1)
    hourly_readings['Power_Consumed'] = hourly_readings['Cumulative_Power'] - hourly_readings['Prev_Cumulative_Power']
    hourly_readings.dropna(subset=['Prev_Cumulative_Power'], inplace=True)
    hourly_power = hourly_readings[['Date', 'Hour', 'Station_Name', 'Power_Consumed']].copy()
    hourly_power.rename(columns={'Power_Consumed': 'Power_Consumed_kWh'}, inplace=True)
    return hourly_power

# --- 4. Final Aggregation and Merging (Unchanged) ---
def get_all_swap_columns() -> List[str]:
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
    if hourly_swaps_df is None and hourly_power_df is None: return {}
    all_dates = [pd.to_datetime(d).date() for d in selected_dates]
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
            values='Swap_Count', fill_value=0
        )
        pivoted_swaps.columns = [f"{int(bp_count)}BP_{soc_bucket}" for bp_count, soc_bucket in pivoted_swaps.columns.values]
        pivoted_swaps.reset_index(inplace=True)
    else:
        pivoted_swaps = pd.DataFrame(columns=['Date', 'Hour', 'Station_Name'] + all_swap_cols)
    
    if hourly_power_df is not None:
        hourly_power_df['Date'] = pd.to_datetime(hourly_power_df['Date']).dt.date
        hourly_power_df = hourly_power_df[
            (hourly_power_df['Date'].isin(all_dates)) & 
            (hourly_power_df['Station_Name'].isin(selected_stations))
        ]
    else:
        hourly_power_df = pd.DataFrame(columns=['Date', 'Hour', 'Station_Name', 'Power_Consumed_kWh'])

    if not pivoted_swaps.empty and not hourly_power_df.empty:
        combined_df = pd.merge(pivoted_swaps, hourly_power_df, on=['Date', 'Hour', 'Station_Name'], how='outer')
    elif not pivoted_swaps.empty: combined_df = pivoted_swaps
    elif not hourly_power_df.empty: combined_df = hourly_power_df
    else: combined_df = pd.DataFrame()

    hourly_dfs: Dict[str, pd.DataFrame] = {}
    fill_cols = ['Power_Consumed_kWh'] + all_swap_cols
    for station in selected_stations:
        template_data = []
        for date in all_dates:
            for hour in TIME_RANGE_HOURS:
                template_data.append({'Date': date, 'Hour': hour, 'Station_Name': station})
        station_template = pd.DataFrame(template_data)
        station_df = combined_df[combined_df['Station_Name'] == station]
        
        # Merge template with data, this retains 'Station_Name' from the template
        final_df = pd.merge(station_template, station_df.drop(columns=['Station_Name'], errors='ignore'), 
                            on=['Date', 'Hour'], how='left')
        
        for col in fill_cols:
            if col not in final_df.columns: final_df[col] = 0
            final_df[col] = final_df[col].fillna(0).astype(np.float64)
        final_df['Time Slot'] = final_df['Hour'].apply(lambda h: f"{h:02d}:00 - {h+1:02d}:00")
        
        all_date_chunks = []
        sorted_dates = sorted(all_dates)
        for date in sorted_dates:
            date_df = final_df[final_df['Date'] == date].sort_values(by='Hour')
            total_row = date_df[fill_cols].sum().to_frame().T
            total_row['Time Slot'] = 'Total'
            total_row['Date'] = date 
            total_row['Hour'] = 99
            # Manually add Station_Name to total row to match columns
            total_row['Station_Name'] = station 
            all_date_chunks.append(pd.concat([date_df, total_row], ignore_index=True))
        
        if all_date_chunks:
            hourly_dfs[station] = pd.concat(all_date_chunks, ignore_index=True)
            hourly_dfs[station]['Date'] = pd.to_datetime(hourly_dfs[station]['Date']).dt.strftime('%Y-%m-%d')
        else:
            hourly_dfs[station] = pd.DataFrame()
    return hourly_dfs

# --- 5. Visualization Functions (Unchanged) ---
def create_power_chart(df: pd.DataFrame, station_name: str) -> alt.Chart:
    chart_df = df[df['Time Slot'] != 'Total'].copy()
    chart_df = chart_df.groupby('Time Slot')['Power_Consumed_kWh'].mean().reset_index()
    time_slot_order = [f"{h:02d}:00 - {h+1:02d}:00" for h in TIME_RANGE_HOURS]
    base = alt.Chart(chart_df).encode(
        x=alt.X('Time Slot:N', sort=time_slot_order, title="Time Slot"),
        tooltip=[
            'Time Slot:N',
            alt.Tooltip('Power_Consumed_kWh:Q', format='.2f', title='Avg Power (kWh)')
        ]
    ).properties(title=f"Avg. Hourly Power Consumption - {station_name}", height=300)
    line_chart = base.mark_line(point=True).encode(
        y=alt.Y('Power_Consumed_kWh:Q', title="Power Consumed (kWh)"),
        color=alt.value("#4c78a8")
    )
    return (line_chart).interactive()

def create_swap_chart(df: pd.DataFrame, station_name: str) -> alt.Chart:
    chart_df = df[df['Time Slot'] != 'Total'].copy()
    swap_cols = get_all_swap_columns()
    for col in swap_cols:
        if col not in chart_df.columns: chart_df[col] = 0
    chart_df = chart_df.groupby('Time Slot')[swap_cols].sum().reset_index()
    chart_long = chart_df.melt(
        id_vars=['Time Slot'], value_vars=swap_cols,
        var_name='Metric', value_name='Swap_Count'
    )
    chart_long = chart_long[chart_long['Swap_Count'] > 0]
    if chart_long.empty:
        return alt.Chart(chart_long).mark_bar().properties(title=f"No Swap Data - {station_name}", height=300)
    chart_long['BP_Count'] = chart_long['Metric'].str.extract(r'(\d+)BP_').astype(int)
    chart_long['SOC_Bucket'] = chart_long['Metric'].str.replace(r'^\d+BP_', '')
    time_slot_order = [f"{h:02d}:00 - {h+1:02d}:00" for h in TIME_RANGE_HOURS]
    chart = alt.Chart(chart_long).mark_bar().encode(
        x=alt.X('Time Slot:N', sort=time_slot_order, title="Time Slot", axis=None),
        y=alt.Y('Swap_Count:Q', title="Total Swap Count"),
        color=alt.Color('SOC_Bucket:N', title="SOC Range", legend=alt.Legend(orient="bottom")),
        column=alt.Column(
            'BP_Count:N', title="Swaps by Battery Pack (BP) Count",
            header=alt.Header(titleOrient="bottom", labelOrient="bottom")
        ),
        tooltip=['Time Slot:N', 'SOC_Bucket:N', 'BP_Count:N', 'Swap_Count:Q']
    ).properties(title=f"Hourly Swap Distribution - {station_name}").interactive()
    return chart

# --- 6. Custom CSS Injection (Unchanged) ---
def inject_custom_css():
    st.markdown(
        """
        <style>
            /* Base */
            .stApp {}
            /* Sidebar */
            [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
            /* KPI Cards */
            [data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.04); }
            [data-testid="stMetricValue"] { font-size: 2.5em; }
            /* Chart Cards */
            [data-testid="stAltairChart"] { background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.04); }
            /* Expander (for tables) */
            [data-testid="stExpander"] { background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 10px; }
            [data-testid="stExpander"] summary { font-size: 1.1em; font-weight: 500; }
            /* Tabs */
            [data-testid="stTabs"] [data-baseweb="tab"] { font-size: 1.1em; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 7. Streamlit Dashboard App (REVERTED) ---

def main():
    inject_custom_css()

    st.title("🔋 Battery Swap & Power Dashboard")
    st.markdown("Advanced analysis of hourly power consumption and swap distribution.")

    # --- NEW CACHING LOGIC ---
    hourly_swaps_df = None
    hourly_power_df = None
    data_loaded = False

    # Check if pre-processed cache files exist (in the GitHub repo)
    if os.path.exists(SWAP_CACHE_FILE) and os.path.exists(POWER_CACHE_FILE):
        with st.sidebar:
            st.success("✅ Pre-processed data found! Loading...")
        try:
            hourly_swaps_df = pd.read_parquet(SWAP_CACHE_FILE)
            hourly_power_df = pd.read_parquet(POWER_CACHE_FILE)
            data_loaded = True
        except Exception as e:
            st.sidebar.error(f"Error loading cache files: {e}")
            st.sidebar.warning("Please clear cache files from GitHub and re-upload.")
            st.stop()
    
    # IF CACHE IS NOT FOUND -> Show Admin Upload Section
    else:
        with st.sidebar:
            st.warning("⚠️ No pre-processed data found.")
            st.info("This 'Admin' section is for uploading new raw data. Stakeholders will not see this if cache files are on GitHub.")
            
            with st.expander("Upload & Process New Data (Admin)"):
                soc_files = st.file_uploader("Upload **Vehicle SOC CSV** files", type=['csv'], accept_multiple_files=True, key='soc_uploader')
                power_files = st.file_uploader("Upload **Power Log CSV** files", type=['csv'], accept_multiple_files=True, key='power_uploader')
                
                if st.button("Process Data to Create Cache"):
                    if not soc_files or not power_files:
                        st.error("Please upload both SOC and Power files.")
                    else:
                        with st.spinner("Processing new data..."):
                            # 1. Load and process swaps
                            soc_df_raw = load_and_merge_data(soc_files, "SOC/Swap")
                            soc_df = preprocess_soc_data(soc_df_raw.copy())
                            final_swaps = apply_swap_logic(soc_df.copy())
                            final_swaps.to_parquet(SWAP_CACHE_FILE)

                            # 2. Load and process power
                            power_df_raw = load_and_merge_data(power_files, "Power Log")
                            power_df = preprocess_power_data(power_df_raw.copy())
                            final_power = calculate_hourly_power(power_df.copy())
                            final_power.to_parquet(POWER_CACHE_FILE)
                            
                            st.success("✅ Data Processed and Saved!")
                            st.warning("👇 **IMPORTANT:** Download both files and upload them to your GitHub repository.")
                        
                # After processing, offer files for download
                if os.path.exists(SWAP_CACHE_FILE):
                    with open(SWAP_CACHE_FILE, 'rb') as f:
                        st.download_button(
                            label="Download Swap Cache (processed_swaps.parquet)",
                            data=f,
                            file_name=SWAP_CACHE_FILE,
                            mime='application/octet-stream'
                        )
                if os.path.exists(POWER_CACHE_FILE):
                    with open(POWER_CACHE_FILE, 'rb') as f:
                        st.download_button(
                            label="Download Power Cache (processed_power.parquet)",
                            data=f,
                            file_name=POWER_CACHE_FILE,
                            mime='application/octet-stream'
                        )

        st.info("Please use the sidebar to upload and process new data to build the cache.")
        st.stop() # Stop the app here. It will re-run once files are processed or found.

    # --- This section only runs if data_loaded = True ---
    
    # Determine available stations and dates from loaded cache
    available_stations = set(hourly_swaps_df['Station_Name'].unique()) | set(hourly_power_df['Station_Name'].unique())
    available_dates = set(pd.to_datetime(hourly_swaps_df['Date']).dt.date.unique()) | set(pd.to_datetime(hourly_power_df['Date']).dt.date.unique())
    all_stations = sorted(list(available_stations))
    all_dates = sorted(list(available_dates))
    
    # 2. Filters Section
    with st.sidebar:
        st.header("⚙️ Analysis Filters")
        selected_stations = st.multiselect("Select Station(s)", options=all_stations, default=all_stations[0] if all_stations else [])
        selected_dates_raw = st.multiselect("Select Date(s)", options=all_dates, default=all_dates if all_dates else [])
        st.info(f"**Fixed Time Range:** {TIME_RANGE_HOURS[0]:02d}:00 to {TIME_RANGE_HOURS[-1]+1:02d}:00")

    if not selected_stations or not selected_dates_raw:
        st.warning("Please select at least one station and one date to proceed.")
        st.stop()

    # --- Core Analysis and Merging (NO CHANGE TO LOGIC) ---
    with st.spinner("Finalizing Dashboard Data..."):
        final_hourly_data = combine_analysis_results(
            hourly_swaps_df, hourly_power_df, selected_stations, selected_dates_raw
        )

    # --- 3 & 4. Output Display and Visualization (Unchanged) ---
    st.header("📈 Overall Performance (All Selected Data)")
    st.markdown("---")

    total_power = 0
    total_swaps = 0
    all_swap_cols = get_all_swap_columns()
    for station_name, df in final_hourly_data.items():
        total_rows = df[df['Time Slot'] == 'Total']
        total_power += total_rows['Power_Consumed_kWh'].sum()
        for col in all_swap_cols:
             if col in total_rows.columns:
                total_swaps += total_rows[col].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="Total Power Consumed (kWh)", value=f"{total_power:,.2f}")
    with col2: st.metric(label="Total Swaps Performed", value=f"{int(total_swaps)}")
    with col3:
        avg_power_per_swap = (total_power / total_swaps) if total_swaps > 0 else 0
        st.metric(label="Avg. Power per Swap (kWh)", value=f"{avg_power_per_swap:,.2f}")

    st.header("📊 Station-by-Station Analysis")
    st.markdown("---")
    if not final_hourly_data:
        st.error("No data found for the selected stations and dates.")
        st.stop()

    tab_list = st.tabs([station for station in final_hourly_data.keys()])
    
    for i, tab in enumerate(tab_list):
        with tab:
            station_name = selected_stations[i]
            df_output = final_hourly_data[station_name]
            
            if df_output.empty:
                st.warning("No data for this station in the selected period.")
                continue

            st.subheader(f"Visual Insights: {station_name}")
            power_chart = create_power_chart(df_output, station_name)
            st.altair_chart(power_chart, use_container_width=True)
            swap_chart = create_swap_chart(df_output, station_name)
            st.altair_chart(swap_chart, use_container_width=True)

            with st.expander(f"View Hourly Data Tables for {station_name}"):
                all_dates_in_df = df_output['Date'].unique()
                for date_str in all_dates_in_df:
                    if date_str is pd.NaT or date_str is None: continue

                    st.markdown(f"#### Hourly Summary for Date: {date_str}")
                    date_specific_df = df_output[df_output['Date'] == date_str].copy()
                    
                    # --- REVERTED TO ST.DATAFRAME ---
                    display_df = date_specific_df.rename(columns={'Power_Consumed_kWh': 'Power (kWh)'})
                    for bp in PREDEF_BP_COUNTS:
                        for bucket in SOC_BUCKETS:
                            # Rename from backend name (e.g., "1BP_Less than 10%")
                            # to display name (e.g., "1BP: Less than 10%")
                            display_df.rename(columns={f"{bp}BP_{bucket}": f"{bp}BP: {bucket}"}, inplace=True)
                    
                    # Re-order columns to group by BUCKET first, then BP count
                    swap_cols_ordered = [
                        f"{bp}BP: {bucket}" 
                        for bucket in SOC_BUCKETS  # <-- Outer loop is BUCKET
                        for bp in PREDEF_BP_COUNTS # <-- Inner loop is BP
                    ]
                    
                    # Create the final list of columns in the new order
                    final_display_cols = ['Time Slot', 'Power (kWh)'] + swap_cols_ordered
                    
                    # Filter to only columns that actually exist in the data
                    final_display_cols_exist = [col for col in final_display_cols if col in display_df.columns]
                    
                    # Display the dataframe with the new column order
                    st.dataframe(display_df[final_display_cols_exist], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
# ==============================================================
# Streamlit app: Power vs SOC vs Swap Analysis (Final, Polished)
# - Fulfills requirements encoded by Srijaun / "Sri"
# - Each Transaction = 1 swap; BP category (1/2/3) assigned
# - SOC bucket uses min SOC unless max-min > threshold -> use avg
# - Multiple upload files (csv/xlsx/zip), multi-station merge
# - Hourly power computed from cumulative Station Energy (diff)
# - Clean UI with tabs; XLSX downloads
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import re
import plotly.express as px
from datetime import date

# --------------------------
# Page config and header
# --------------------------
st.set_page_config(page_title="Power vs SOC vs Swap — Advanced", layout="wide")
st.markdown("<h1 style='text-align:center'>🔋 Power vs SOC vs Swap — Station Level</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:gray'>Transaction-level swaps (1 per vehicle). BP category assigned by No. of BPs. SOC bucket uses min SOC or average when packs differ widely.</div>", unsafe_allow_html=True)
st.write(" ")

# --------------------------
# Constants
# --------------------------
HOUR_MIN, HOUR_MAX = 7, 23
SOC_BUCKETS_ORDER = ['Less than 10%', 'Between 10-30%', 'Between 30.1-50%', 'Between 50.1-70%', 'Between 70.1-79%', 'Less than 99.9%']
BP_COUNTS = [1, 2, 3]

# --------------------------
# Helper functions for file loading
# --------------------------
def extract_csvs_from_zip(uploaded_file):
    """Return concatenated DataFrame of all csv/xlsx files inside the zip."""
    dfs = []
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            inner = [f for f in z.namelist() if not f.endswith('/') and not f.startswith('__MACOSX')]
            for f in inner:
                if f.lower().endswith('.csv'):
                    with z.open(f) as fh:
                        dfs.append(pd.read_csv(fh))
                elif f.lower().endswith('.xlsx') or f.lower().endswith('.xls'):
                    with z.open(f) as fh:
                        dfs.append(pd.read_excel(fh))
    except Exception as e:
        st.warning(f"Could not read zip {getattr(uploaded_file,'name','uploaded')}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

def load_single_file(f_obj):
    """Load single file-like object. Returns DataFrame or None."""
    if f_obj is None:
        return None
    name = getattr(f_obj, 'name', '')
    try:
        if name.lower().endswith('.zip'):
            return extract_csvs_from_zip(f_obj)
        if name.lower().endswith('.csv'):
            return pd.read_csv(f_obj)
        if name.lower().endswith('.xlsx') or name.lower().endswith('.xls'):
            return pd.read_excel(f_obj)
        # fallback try CSV
        try:
            return pd.read_csv(f_obj)
        except:
            return pd.read_excel(f_obj)
    except Exception as e:
        st.warning(f"Failed to load {name}: {e}")
        return None

def load_multiple_files(list_of_files):
    """Accepts list returned by st.file_uploader(..., accept_multiple_files=True)"""
    if not list_of_files:
        return None
    dfs = []
    for f in list_of_files:
        df = load_single_file(f)
        if df is not None and not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

# --------------------------
# SOC parsing & bucketing helpers
# --------------------------
def parse_soc_list(s):
    """Return list of floats from incoming SOC field (handles commas, pipes, semicolons)."""
    if pd.isna(s):
        return []
    s_str = str(s).replace('%','').strip()
    parts = re.split(r'[,\|;]', s_str)
    vals = []
    for p in parts:
        p = p.strip()
        if p == '':
            continue
        try:
            vals.append(float(p))
        except:
            # try to clean stray chars
            p_clean = re.sub(r'[^\d\.\-]', '', p)
            try:
                vals.append(float(p_clean))
            except:
                continue
    return vals

def choose_bucket_for_transaction(socs, threshold_pct):
    """
    socs: list of floats (incoming pack SOCs)
    threshold_pct: if max-min > threshold -> use average, else use min
    returns bucket label string (as per SOC_BUCKETS_ORDER) or None
    """
    if not socs:
        return None
    s_arr = [float(x) for x in socs]
    mn = min(s_arr)
    mx = max(s_arr)
    if (mx - mn) > threshold_pct:
        val = sum(s_arr) / len(s_arr)
    else:
        val = mn
    # map val to bucket
    if val <= 10:
        return 'Less than 10%'
    if val <= 30:
        return 'Between 10-30%'
    if val <= 50:
        return 'Between 30.1-50%'
    if val <= 70:
        return 'Between 50.1-70%'
    if val <= 79:
        return 'Between 70.1-79%'
    if val <= 99.9:
        return 'Less than 99.9%'
    return None

# --------------------------
# UX: Sidebar controls
# --------------------------
st.sidebar.header("Data & Options")

# Master session option
use_master = st.sidebar.checkbox("Use master merged (session) instead of uploads", value=False)
master_upload = None
if use_master:
    master_upload = st.sidebar.file_uploader("Upload master merged dataframe (optional)", type=['csv','xlsx'])

# File uploaders (only used if not using master)
swap_uploads = None
power_uploads = None
if not use_master:
    st.sidebar.markdown("**Upload Swap SOC Files (multiple)**")
    swap_uploads = st.sidebar.file_uploader("Swap SOC files (csv/xlsx/zip)", accept_multiple_files=True, type=['csv','xlsx','zip'])
    st.sidebar.markdown("**Upload Station Power Files (multiple)**")
    power_uploads = st.sidebar.file_uploader("Power files (csv/xlsx/zip)", accept_multiple_files=True, type=['csv','xlsx','zip'])

# SOC threshold slider
threshold = st.sidebar.slider("SOC difference threshold (%) to switch to average (default = 8%)", min_value=0, max_value=20, value=8)

# Fallback for power
power_fallback = st.sidebar.checkbox("Fallback to swap 'Total kWh Consumed' when power data missing", value=True)

# Small help
st.sidebar.markdown("---")
st.sidebar.markdown("Hours considered: **07 → 23** (based on Swap Start Time).")
st.sidebar.markdown("Each Transaction = 1 swap. BP category uses `No. of BPs`.")

# --------------------------
# Load data: either master or uploads
# --------------------------
swap_df = None
power_df = None
if use_master and master_upload is not None:
    master_df = load_single_file(master_upload)
    if master_df is None:
        st.error("Master file could not be read.")
        st.stop()
    # We expect master to be transaction-level with columns including Station Id, Date, Hour, No. of BPs, Incoming BP SOC, etc.
    swap_df = master_df.copy()
    power_df = None  # assume not required; fallback behavior later
    st.success("Master loaded into app (will be used as swap dataset).")
elif not use_master:
    # Load and merge all uploaded swap files and power files
    with st.spinner("Loading files..."):
        swap_df = load_multiple_files(swap_uploads)
        power_df = load_multiple_files(power_uploads)

    if swap_df is None:
        st.warning("No swap files uploaded yet.")
    else:
        st.success(f"Loaded swap data: {len(swap_df):,} rows from uploads." )
    if power_df is None:
        st.warning("No power files uploaded yet.")
    else:
        st.success(f"Loaded power data: {len(power_df):,} rows from uploads." )

# If still no swap_df, stop
if swap_df is None or swap_df.empty:
    st.info("Upload swap SOC files (or load master). Waiting for input.")
    st.stop()

# --------------------------
# CLEAN & PREPARE swap_df
# --------------------------
# trim column names
swap_df.columns = swap_df.columns.str.strip()
# Normalize station id column name variations
if 'Station Id' not in swap_df.columns and 'Station ID' in swap_df.columns:
    swap_df.rename(columns={'Station ID':'Station Id'}, inplace=True)

# Station Id normalize
if 'Station Id' in swap_df.columns:
    swap_df['Station Id'] = swap_df['Station Id'].astype(str).str.strip().str.upper()
else:
    # create placeholder if missing (will break station selection later)
    swap_df['Station Id'] = 'UNKNOWN'

# Datetime parsing: Swap Start Time
if 'Swap Start Time' not in swap_df.columns:
    st.error("Swap file(s) must contain 'Swap Start Time' column. Rename accordingly and re-upload.")
    st.stop()
swap_df['Swap Start Time'] = pd.to_datetime(swap_df['Swap Start Time'], errors='coerce')

# Date & Hour
swap_df['Date'] = swap_df['Swap Start Time'].dt.date
swap_df['Hour'] = swap_df['Swap Start Time'].dt.hour

# Filter hours 7..23 and drop invalid datetimes
invalid_dt = swap_df['Swap Start Time'].isna().sum()
if invalid_dt > 0:
    st.warning(f"{invalid_dt} swap-row(s) had invalid 'Swap Start Time' and were dropped.")
swap_df = swap_df[swap_df['Hour'].between(HOUR_MIN, HOUR_MAX)].copy()

# Parse incoming BP SOC lists
swap_df['Incoming_BP_SOC_list'] = swap_df.get('Incoming BP SOC', '').apply(parse_soc_list)

# Parse No. of BPs
if 'No. of BPs' in swap_df.columns:
    swap_df['No. of BPs'] = pd.to_numeric(swap_df['No. of BPs'], errors='coerce').fillna(1).astype(int)
else:
    # assume 1 if column missing
    swap_df['No. of BPs'] = 1

# For transactions, we need one record per Transaction Id (transaction-level)
if 'Transaction Id' not in swap_df.columns:
    st.error("Swap dataset must have 'Transaction Id' column.")
    st.stop()

# Prepare per-transaction summary: choose station, date, hour, no. of BPs, best soc for bucketing
trans_agg = []
for tx, g in swap_df.groupby('Transaction Id', sort=False):
    # choose first non-null Station Id, Date, Hour, No. of BPs
    station = g['Station Id'].iloc[0]
    date_val = g['Date'].iloc[0]
    hour_val = g['Hour'].iloc[0]
    # combine all incoming soc lists from rows in this transaction
    socs = []
    for lst in g['Incoming_BP_SOC_list']:
        socs.extend(lst)
    # if still empty, try to read from 'Incoming BP SOC' raw column
    if not socs and 'Incoming BP SOC' in g.columns:
        # try parse single string
        for raw in g['Incoming BP SOC'].astype(str).tolist():
            socs.extend(parse_soc_list(raw))
    # pick No. of BPs from first non-null or sum if inconsistent (prefer first)
    try:
        bp_count = int(g['No. of BPs'].dropna().iloc[0])
    except:
        bp_count = 1
    # bucket choice using threshold slider
    bucket = choose_bucket_for_transaction(socs, threshold)
    trans_agg.append({
        'Transaction Id': tx,
        'Station Id': station,
        'Date': date_val,
        'Hour': hour_val,
        'No. of BPs': bp_count,
        'Incoming_SOC_list': socs,
        'SOC_Bucket': bucket
    })

tx_df = pd.DataFrame(trans_agg)

# Ensure station id normalized
tx_df['Station Id'] = tx_df['Station Id'].astype(str).str.strip().str.upper()

# Basic validations
total_transactions_loaded = len(tx_df)
if total_transactions_loaded == 0:
    st.error("No valid transactions remained after preprocessing. Check the Swap Start Time and Transaction Id fields.")
    st.stop()

# --------------------------
# POWER PREPARATION
# --------------------------
if power_df is not None and not power_df.empty:
    power_df.columns = power_df.columns.str.strip()
    # normalize station id col name
    if 'Station ID' not in power_df.columns and 'Station Id' in power_df.columns:
        power_df.rename(columns={'Station Id':'Station ID'}, inplace=True)
    if 'Station ID' not in power_df.columns:
        st.warning("Power file missing 'Station ID' column. Power data may not be merged.")
    else:
        power_df['Station ID'] = power_df['Station ID'].astype(str).str.strip().str.upper()

    # parse Date
    if 'Date' not in power_df.columns:
        st.warning("Power file missing 'Date' column. Power will not be used unless present.")
    else:
        power_df['Date'] = pd.to_datetime(power_df['Date'], errors='coerce').dt.date

    # convert Hour safely
    if 'Hour' in power_df.columns:
        power_df['Hour'] = pd.to_numeric(power_df['Hour'], errors='coerce')
        nh = int(power_df['Hour'].isna().sum())
        if nh > 0:
            st.warning(f"{nh} power-row(s) have invalid Hour and will be dropped.")
        power_df = power_df[power_df['Hour'].notna()].copy()
        power_df['Hour'] = power_df['Hour'].astype(int)
        power_df = power_df[power_df['Hour'].between(HOUR_MIN, HOUR_MAX)]
    else:
        st.warning("Power file missing 'Hour' column. Power will not be used unless Hour present.")

    # ensure Station Energy numeric
    if 'Station Energy' in power_df.columns:
        power_df['Station Energy'] = pd.to_numeric(power_df['Station Energy'], errors='coerce').fillna(0)
    else:
        st.warning("Power file missing 'Station Energy' column. Power will not be used unless present.")

    # compute hourly diff per station-date
    if set(['Station ID','Date','Hour','Station Energy']).issubset(power_df.columns):
        power_df = power_df.sort_values(['Station ID','Date','Hour']).reset_index(drop=True)
        power_df['Prev_Station_Energy'] = power_df.groupby(['Station ID','Date'])['Station Energy'].shift(1)
        # compute delta
        power_df['Power_Consumed_kWh'] = power_df['Station Energy'] - power_df['Prev_Station_Energy']
        # for first entry of day (prev NaN), set delta = Station Energy (or 0)
        power_df.loc[power_df['Prev_Station_Energy'].isna(), 'Power_Consumed_kWh'] = power_df.loc[power_df['Prev_Station_Energy'].isna(), 'Station Energy']
        power_df['Power_Consumed_kWh'] = power_df['Power_Consumed_kWh'].clip(lower=0.0)
    else:
        st.warning("Power data incomplete (Station ID / Date / Hour / Station Energy). Power merge may not work.")

# --------------------------
# Sidebar filters (station & dates)
# --------------------------
st.sidebar.header("Select scope")
stations = sorted(tx_df['Station Id'].unique())
sel_station = st.sidebar.selectbox("Station (station-level output)", options=stations)

dates_all = sorted(tx_df['Date'].dropna().unique())
dates_str = [str(d) for d in dates_all]
sel_dates_str = st.sidebar.multiselect("Date(s)", options=dates_str, default=dates_str)
sel_dates = [pd.to_datetime(d).date() for d in sel_dates_str]

# --------------------------
# Build filtered frames
# --------------------------
f_tx = tx_df[(tx_df['Station Id']==sel_station) & (tx_df['Date'].isin(sel_dates))].copy()
if power_df is not None and set(['Station ID','Date','Hour','Power_Consumed_kWh']).issubset(power_df.columns):
    f_power = power_df[(power_df['Station ID']==sel_station) & (power_df['Date'].isin(sel_dates))].copy()
else:
    f_power = None

# If fallback is selected and power missing, we'll use swap-level 'Total kWh Consumed' if present
if f_power is None and power_fallback:
    # attempt to use original swap_df's Total kWh Consumed aggregated per hour
    if 'Total kWh Consumed' in swap_df.columns:
        tmp = swap_df.copy()
        # ensure Station Id normalize same
        tmp['Station Id'] = tmp.get('Station Id', '').astype(str).str.strip().str.upper()
        tmp = tmp[(tmp['Station Id'] == sel_station) & (tmp['Date'].isin(sel_dates))]
        # Some swap rows may repeat per bp - but we aggregate per transaction earlier - so we must sum per hour from original swap rows
        # We'll sum 'Total kWh Consumed' in tmp per hour
        tmp['Total kWh Consumed'] = pd.to_numeric(tmp.get('Total kWh Consumed', 0), errors='coerce').fillna(0)
        power_fallback_hourly = tmp.groupby('Hour', as_index=False)['Total kWh Consumed'].sum().rename(columns={'Total kWh Consumed':'Power_Consumed_kWh'})
        f_power = power_fallback_hourly.copy()
    else:
        f_power = None

# --------------------------
# Build final hourly table
# --------------------------
hours_range = list(range(HOUR_MIN, HOUR_MAX+1))
hours_df = pd.DataFrame({'Hour': hours_range})

# Power hourly summary (ensure we have Hour column)
if f_power is None or f_power.empty:
    power_hour = hours_df.copy()
    power_hour['Power_Consumed_kWh'] = 0.0
else:
    # if f_power contains Power_Consumed_kWh per Hour or Station, aggregate grouping by Hour
    if 'Power_Consumed_kWh' in f_power.columns:
        power_hour = f_power.groupby('Hour', as_index=False)['Power_Consumed_kWh'].sum().reset_index(drop=True)
    elif 'Power_Consumed kWh' in f_power.columns:
        power_hour = f_power.groupby('Hour', as_index=False)['Power_Consumed kWh'].sum().reset_index(drop=True).rename(columns={'Power_Consumed kWh':'Power_Consumed_kWh'})
    else:
        # unexpected structure
        power_hour = hours_df.copy()
        power_hour['Power_Consumed_kWh'] = 0.0
    # ensure all hours present
    power_hour = hours_df.merge(power_hour, on='Hour', how='left').fillna(0)

# Total swaps (transactions) and split by BP type
# Each transaction counts as 1 swap; categorize by No. of BPs
swap_counts = f_tx.groupby(['Hour', 'No. of BPs'], as_index=False)['Transaction Id'].count().rename(columns={'Transaction Id':'Count'})
# pivot
swap_counts_pivot = swap_counts.pivot_table(index='Hour', columns='No. of BPs', values='Count', fill_value=0)
# ensure columns for 1,2,3
for c in BP_COUNTS:
    if c not in swap_counts_pivot.columns:
        swap_counts_pivot[c] = 0
swap_counts_pivot = swap_counts_pivot[[1,2,3]].rename(columns={1:'Total_Swaps_1BP', 2:'Total_Swaps_2BP', 3:'Total_Swaps_3BP'})
swap_counts_pivot = hours_df.merge(swap_counts_pivot.reset_index(), on='Hour', how='left').fillna(0).set_index('Hour')

# Total transactions per hour (sum across BP types)
swap_counts_pivot['Total_Swaps'] = swap_counts_pivot[['Total_Swaps_1BP','Total_Swaps_2BP','Total_Swaps_3BP']].sum(axis=1).astype(int)

# Bucketed counts by (transaction-level) SOC bucket and BP type
bucketed = f_tx.groupby(['Hour','SOC_Bucket','No. of BPs'], as_index=False)['Transaction Id'].count().rename(columns={'Transaction Id':'Count'})
if bucketed.empty:
    # create zero frame with expected columns
    cols = [f"{b} ({bp}BP)" for b in SOC_BUCKETS_ORDER for bp in BP_COUNTS]
    pivot_buckets = pd.DataFrame(0, index=hours_range, columns=cols)
else:
    pivot_buckets = bucketed.pivot_table(index='Hour', columns=['SOC_Bucket','No. of BPs'], values='Count', fill_value=0)
    # flatten
    new_cols = []
    for col in pivot_buckets.columns:
        bucket_label = str(col[0])
        bp_label = int(col[1])
        new_cols.append(f"{bucket_label} ({bp_label}BP)")
    pivot_buckets.columns = new_cols
    # ensure all expected columns exist
    for b in SOC_BUCKETS_ORDER:
        for bp in BP_COUNTS:
            cname = f"{b} ({bp}BP)"
            if cname not in pivot_buckets.columns:
                pivot_buckets[cname] = 0
    pivot_buckets = pivot_buckets[[f"{b} ({bp}BP)" for b in SOC_BUCKETS_ORDER for bp in BP_COUNTS]]
    pivot_buckets = pivot_buckets.reindex(hours_range, fill_value=0)

# Merge everything to final table
power_hour = power_hour.set_index('Hour')
final_df = power_hour.join(swap_counts_pivot, how='left').fillna(0)
final_df = final_df.join(pivot_buckets, how='left').fillna(0)

# Ensure types and formatting
final_df['Power_Consumed_kWh'] = final_df['Power_Consumed_kWh'].astype(float).round(3)
for c in ['Total_Swaps_1BP','Total_Swaps_2BP','Total_Swaps_3BP','Total_Swaps']:
    if c in final_df.columns:
        final_df[c] = final_df[c].astype(int)

# Grand total
totals = final_df.sum(numeric_only=True)
totals.name = 'Grand Total'
final_with_total = pd.concat([final_df, totals.to_frame().T])

# --------------------------
# UI: Tabs (Table, Charts, Validation)
# --------------------------
tab1, tab2, tab3 = st.tabs(["Table", "Charts", "Validation"])

with tab1:
    st.subheader(f"Station: {sel_station} — Dates: {', '.join(sel_dates_str) if sel_dates_str else 'ALL'}")
    st.dataframe(final_with_total, use_container_width=True)
    # download final table as xlsx
    def to_xlsx_bytes(df):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='station_hourly', index=True)
            writer.save()
        out.seek(0)
        return out.read()
    st.download_button("Download hourly table (XLSX)", data=to_xlsx_bytes(final_with_total.reset_index()), file_name=f"{sel_station}_hourly_{date.today()}.xlsx")

with tab2:
    st.subheader("Charts: Power vs Swaps (per hour)")
    # prepare a tidy dataframe for plotting
    plot_df = final_df.reset_index().melt(id_vars=['Hour','Power_Consumed_kWh','Total_Swaps'], value_vars=[c for c in final_df.columns if c.endswith('BP') and 'Less' in c or 'Between' in c], var_name='Bucket_BP', value_name='Count')
    # Simpler plots: Power line, total swaps line
    fig = px.line(final_df.reset_index(), x='Hour', y='Power_Consumed_kWh', markers=True, title='Hourly Power Consumption')
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(final_df.reset_index(), x='Hour', y='Total_Swaps', title='Hourly Total Swaps (transactions)')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Tip:** Use the Table tab for full bucket breakdown (columns).")

with tab3:
    st.subheader("Validation & Diagnostics")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Counts & rows**")
        st.write(f"- Transactions loaded (selected scope): {len(f_tx)}")
        st.write(f"- Total transactions (all data): {total_transactions_loaded}")
        st.write(f"- Distinct stations in transactions: {tx_df['Station Id'].nunique()}")
        st.write(f"- Bucket distribution (selected scope):")
        st.dataframe(f_tx['SOC_Bucket'].value_counts().reindex(SOC_BUCKETS_ORDER).fillna(0).astype(int))

    with colB:
        st.write("**Power & files**")
        if f_power is None:
            st.write("- No station power available for selected scope.")
            if power_fallback:
                st.write("- Fallback used from swaps (if available).")
        else:
            st.write(f"- Power rows used: {len(f_power)}")
            st.dataframe(f_power.head(5))
        st.write("**Files loaded summary**")
        st.write(f"- Swap input rows (raw): {len(swap_df)}")
        st.write(f"- Unique Transactions (grouped): {total_transactions_loaded}")
        st.write(f"- Power raw rows (if any): {0 if power_df is None else len(power_df)}")

# --------------------------
# Master save/download options
# --------------------------
st.markdown("---")
st.header("Save / Master / Download")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Save current merged transactions as session master"):
        st.session_state['saved_master'] = tx_df
        st.success("Saved transaction-level master in session.")

with col2:
    if st.button("Clear session master"):
        st.session_state.pop('saved_master', None)
        st.info("Session master cleared.")

with col3:
    if 'saved_master' in st.session_state:
        df_master = st.session_state['saved_master']
        def to_xlsx_master(df):
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='transactions', index=False)
                writer.save()
            out.seek(0)
            return out.read()
        st.download_button("Download saved master (XLSX)", data=to_xlsx_master(df_master), file_name=f"master_transactions_{date.today()}.xlsx")

st.success("Analysis complete. If anything is off, point to the exact mismatch (example: station X, date Y, hour Z) and I will patch immediately.")

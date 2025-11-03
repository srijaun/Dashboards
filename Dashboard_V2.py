# app.py
# Swaps vs Power - production-ready Streamlit dashboard (single-file)
# pip install streamlit pandas numpy plotly openpyxl pyarrow kaleido
# run: streamlit run app.py

import io
import os
import zipfile
from typing import Dict, List, Tuple, Optional, Iterable
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Swaps vs Power Dashboard", layout="wide")
st.title("Swaps vs Power Dashboard")
st.caption("Upload Station Power & SoC/Swap data (CSV/XLSX or ZIP). Supports multi-file uploads, multi-site filters, weekly insights & downloads.")

MASTER_PARQUET = "master_combined.parquet"

# Provide your mapping here
SITE_TO_STATIONS: Dict[str, List[str]] = {
    "BLR_SMP_SUNM office, Doddanekkundi": ["WMQISXM1V1-00192", "WMQISXM1V1-00193", "WMQISXM1V1-00605"],
    "BLR_PVT_Hub_5thcross,HSR Layout": ["WMQISXM1V1-00705", "WMQISXM1V1-00713", "WMQISXM1V1-00725", "WMQISXM1V1-00739"],
    "BLR_PVT_Hub_Choodasandra,Hosa Road": ["WMQISXM1V1-00614", "WMQISXM1V1-00669", "WMQISXM1V1-00682", "WMQISXM1V1-00732"],
    "Delhi-Metro-Shubhash-Nagar": ["WMQISXM1V1-00137", "WMQISXM1V1-00170", "WMQISXM1V1-00267"],
    "Delhi-BRPL-Motibagh": ["WMQISXM1V1-00346", "WMQISXM1V1-00388", "WMQISXM1V1-00437", "WMQISXM1V1-01036"], 
    "Delhi-Mayapuri-Depot": ["WMQISXM1V1-00012", "WMQISXM1V1-02043", "WMQISXM1V1-02048"], 
    "Delhi-STATIQ-Govindpuri-Metro-Station": ["WMQISXM1V1-00150", "WMQISXM1V1-00226", "WMQISXM1V1-00421"], 
}

PALETTE = ["#5B9BD5", "#ED7D31", "#A6A6A6", "#70AD47"]

SOC_BUCKETS_ORDER = [
    "Less than 10%",
    "Between 10 - 30%",
    "Between 30.1- 50%",
    "Between 50.1 - 70%",
    "Between 70.1 - 85%",
    "Between 85.1 - 100%",
]

# -------------------------------
# UTILITIES
# -------------------------------
def make_unique_columns(cols: Iterable[str]) -> List[str]:
    """
    Return list where duplicate column names are made unique by appending suffixes.
    Example: ['a','a','b'] -> ['a','a__1','b']
    """
    out = []
    seen = {}
    for c in cols:
        base = str(c)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    return out

def safe_read_csv_or_excel(buf_or_path) -> pd.DataFrame:
    """
    Try CSV then Excel; accepts Path-like, file-like, or streamlit UploadedFile
    """
    try:
        name = getattr(buf_or_path, "name", "") or ""
        lname = name.lower()
        if lname.endswith(".csv") or (isinstance(buf_or_path, (bytes, bytearray)) is False and isinstance(buf_or_path, str) and buf_or_path.lower().endswith(".csv")):
            return pd.read_csv(buf_or_path)
        if lname.endswith(".xlsx") or (isinstance(buf_or_path, (bytes, bytearray)) is False and isinstance(buf_or_path, str) and buf_or_path.lower().endswith(".xlsx")):
            return pd.read_excel(buf_or_path)
        # fallback
        try:
            return pd.read_csv(buf_or_path)
        except Exception:
            return pd.read_excel(buf_or_path)
    except Exception:
        return pd.DataFrame()

def standardize_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # make columns lowercase, trimmed, non-word replaced by underscore
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # ensure uniqueness
    df.columns = make_unique_columns(df.columns)
    return df

def ensure_int_hour(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "iu":
        return series.astype(int)
    parsed = pd.to_datetime(series.astype(str), errors="coerce").dt.hour
    if parsed.notna().any():
        return parsed.fillna(0).astype(int)
    # fallback try numeric then fill 0
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def soc_bucket(v: float) -> str:
    if pd.isna(v):
        return "Unknown"
    x = float(v)
    if x >= 100:
        x = 99.8
    if x < 10:
        return "Less than 10%"
    if 10 <= x <= 30:
        return "Between 10 - 30%"
    if 30 < x <= 50:
        return "Between 30.1- 50%"
    if 50 < x <= 70:
        return "Between 50.1 - 70%"
    if 70 < x <= 85:
        return "Between 70.1 - 85%"
    if 85 < x <= 100:
        return "Between 85.1 - 100%"
    return "Unknown"

def build_station_color_map(stations: List[str]) -> Dict[str, str]:
    colors = {}
    for i, s in enumerate(sorted(stations)):
        colors[s] = PALETTE[i % len(PALETTE)]
    return colors

# -------------------------------
# ZIP helpers (robust)
# -------------------------------
def extract_frames_from_zip_buffer(zip_buffer: io.BytesIO) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Reads zip bytes (or file-like) and returns two lists: power_frames, swaps_frames.
    Classification uses filename keywords and column heuristics.
    """
    p_list, s_list = [], []
    try:
        with zipfile.ZipFile(zip_buffer) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                lname = name.lower()
                if not (lname.endswith(".csv") or lname.endswith(".xlsx")):
                    continue
                try:
                    raw = zf.read(name)
                    bio = io.BytesIO(raw)
                    df = None
                    if lname.endswith(".csv"):
                        df = pd.read_csv(bio)
                    else:
                        df = pd.read_excel(bio)
                    if df is None or df.empty:
                        continue
                    df = standardize_columns_lower(df)
                except Exception:
                    # skip problematic files but continue processing others
                    continue

                # classification by filename
                if any(x in lname for x in ["power", "station_energy", "station_energy_total", "energy"]):
                    p_list.append(df)
                    continue
                if any(x in lname for x in ["soc", "swap", "received_soc", "swap_start", "received_postswap"]):
                    s_list.append(df)
                    continue

                # fallback classification by columns
                cols = [c.lower() for c in df.columns]
                if any("station_energy" in c or "station_energy_total" in c or "station_energy" in c or "energy" == c for c in cols):
                    p_list.append(df)
                    continue
                if any("received_soc" in c or "incoming_soc" in c or "swap_start" in c or "swap_time" in c for c in cols):
                    s_list.append(df)
                    continue

                # ambiguous: ignore to avoid polluting classification
    except Exception:
        # return whatever collected so far
        pass
    return p_list, s_list

# -------------------------------
# CORE: build master (robust + cached)
# -------------------------------
@st.cache_data(show_spinner=True)
def build_master_from_uploads(power_files: List[io.BytesIO], swaps_files: List[io.BytesIO], zip_files: List[io.BytesIO]) -> pd.DataFrame:
    """
    Accepts lists of UploadedFile objects (or file-like). Returns merged master DataFrame
    with columns: station_id, date_only (date), hour (int), power_consumed, SOC buckets..., total_swaps,
    mean_received_soc, min_received_soc, max_received_soc, time_str.
    """
    power_frames, swap_frames = [], []

    # read direct power files
    for f in power_files or []:
        try:
            df = safe_read_csv_or_excel(f)
            if df is None or df.empty:
                continue
            df = standardize_columns_lower(df)
            power_frames.append(df)
        except Exception:
            continue

    # read direct swap files
    for f in swaps_files or []:
        try:
            df = safe_read_csv_or_excel(f)
            if df is None or df.empty:
                continue
            df = standardize_columns_lower(df)
            swap_frames.append(df)
        except Exception:
            continue

    # read zips
    for z in zip_files or []:
        try:
            # z is UploadedFile -> get bytes
            data = z.read()
            buf = io.BytesIO(data)
            p_list, s_list = extract_frames_from_zip_buffer(buf)
            power_frames.extend([df for df in p_list if not df.empty])
            swap_frames.extend([df for df in s_list if not df.empty])
        except Exception:
            continue

    # if none present return empty
    if not power_frames and not swap_frames:
        return pd.DataFrame()

    # Before concatenation: ensure every DataFrame has unique columns (prevent InvalidIndexError)
    def _fix_df_cols(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        out = []
        for df in dfs:
            df = df.copy()
            # standardize columns lowercase + unique
            df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
            df.columns = make_unique_columns(df.columns)
            out.append(df)
        return out

    power_frames = _fix_df_cols(power_frames)
    swap_frames = _fix_df_cols(swap_frames)

    # concat safely (if no frames, empty df)
    try:
        df_power_raw = pd.concat(power_frames, ignore_index=True, sort=False) if power_frames else pd.DataFrame()
    except Exception as e:
        # fallback: merge row by row to avoid index issues
        temp = []
        for df in power_frames:
            temp.append(df.reset_index(drop=True))
        df_power_raw = pd.concat(temp, ignore_index=True, sort=False) if temp else pd.DataFrame()

    try:
        df_swaps_raw = pd.concat(swap_frames, ignore_index=True, sort=False) if swap_frames else pd.DataFrame()
    except Exception as e:
        temp = []
        for df in swap_frames:
            temp.append(df.reset_index(drop=True))
        df_swaps_raw = pd.concat(temp, ignore_index=True, sort=False) if temp else pd.DataFrame()

    # If either empty, try to proceed but warn upstream
    # PROCESS POWER
    power_hourly = pd.DataFrame()
    if not df_power_raw.empty:
        # find candidate columns robustly (scan names that contain substrings)
        cols = df_power_raw.columns.tolist()
        def find_candidate(substrs: List[str]) -> Optional[str]:
            for s in substrs:
                for c in cols:
                    if s in c:
                        return c
            return None

        station_col = find_candidate(["station_id", "stationid", "station"])
        date_col = find_candidate(["date_time", "datetime", "timestamp", "date"])
        hour_col = find_candidate(["hour", "_hour", "hr"])
        energy_col = find_candidate(["station_energy_total", "station_energy", "station_energy_kwh", "station_total_energy", "energy"])

        if station_col and date_col and energy_col:
            dfp = df_power_raw.copy()
            # normalize
            dfp[station_col] = dfp[station_col].astype(str).str.strip()
            dfp[date_col] = pd.to_datetime(dfp[date_col], errors="coerce")
            dfp = dfp.dropna(subset=[date_col]).copy()

            if not hour_col:
                dfp["hour"] = dfp[date_col].dt.hour
                hour_col = "hour"
            else:
                dfp[hour_col] = ensure_int_hour(dfp[hour_col])

            dfp["date_only"] = dfp[date_col].dt.date

            # sort and compute diff per station-date
            try:
                dfp = dfp.sort_values([station_col, "date_only", hour_col])
            except Exception:
                # if sorting fails due to weird columns, fallback to index order
                dfp = dfp.copy()

            dfp["power_consumed"] = dfp.groupby([station_col, "date_only"])[energy_col].diff().fillna(0.0)
            dfp.loc[dfp["power_consumed"] < 0, "power_consumed"] = 0.0
            dfp["power_consumed"] = dfp["power_consumed"].replace([np.nan, np.inf, -np.inf], 0.0)

            # aggregate hourly (in case multiple rows per hour)
            power_hourly = dfp.groupby([station_col, "date_only", hour_col], as_index=False)["power_consumed"].sum()
            power_hourly = power_hourly.rename(columns={station_col: "station_id", hour_col: "hour"})
            power_hourly["station_id"] = power_hourly["station_id"].astype(str).str.strip()
            # ensure correct types
            power_hourly["hour"] = power_hourly["hour"].astype(int)
        else:
            # unable to detect required cols – leave empty (upstream will warn)
            power_hourly = pd.DataFrame()

    # PROCESS SWAPS
    swaps_pivot = pd.DataFrame()
    swaps_stats = pd.DataFrame()
    if not df_swaps_raw.empty:
        cols = df_swaps_raw.columns.tolist()
        def find_candidate_swap(substrs: List[str]) -> Optional[str]:
            for s in substrs:
                for c in cols:
                    if s in c:
                        return c
            return None

        station_col = find_candidate_swap(["station_id", "stationid", "station"])
        time_col = find_candidate_swap(["swap_start_time", "swap_start", "swap_time", "start_time", "timestamp", "date"])
        soc_col = find_candidate_swap(["received_soc", "incoming_soc", "received_postswap_soc", "issued_soc", "soc"])

        dfs = df_swaps_raw.copy()
        # If time column present -> parse to derive date_only & hour
        if time_col and time_col in dfs.columns:
            dfs[time_col] = pd.to_datetime(dfs[time_col], errors="coerce")
            dfs = dfs.dropna(subset=[time_col]).copy()
            dfs["date_only"] = dfs[time_col].dt.date
            dfs["hour"] = dfs[time_col].dt.hour
        else:
            # fallback: try to create date_only if 'date' exists
            if "date" in dfs.columns:
                try:
                    dfs["date_only"] = pd.to_datetime(dfs["date"], errors="coerce").dt.date
                    dfs["hour"] = pd.to_datetime(dfs["date"], errors="coerce").dt.hour.fillna(0).astype(int)
                except Exception:
                    dfs["date_only"] = pd.NaT
                    dfs["hour"] = 0
            else:
                dfs["date_only"] = pd.NaT
                dfs["hour"] = 0

        # soc numeric
        if soc_col and soc_col in dfs.columns:
            # ensure it's a Series or list-like
            try:
                dfs[soc_col] = pd.to_numeric(dfs[soc_col], errors="coerce").clip(lower=0, upper=100)
                dfs.loc[dfs[soc_col] >= 100, soc_col] = 99.8
                dfs["soc_category"] = dfs[soc_col].apply(soc_bucket)
            except TypeError:
                # if type error, set NaNs
                dfs["soc_category"] = "Unknown"
        else:
            dfs["soc_category"] = "Unknown"

        # normalize station id column if present
        if station_col and station_col in dfs.columns:
            dfs[station_col] = dfs[station_col].astype(str).str.strip()
        else:
            # try common names
            if "station_id" in dfs.columns:
                dfs["station_id"] = dfs["station_id"].astype(str).str.strip()
                station_col = "station_id"
            else:
                dfs["station_id"] = ""
                station_col = "station_id"

        # counts
        try:
            swaps_soc = dfs.groupby([station_col, "date_only", "hour", "soc_category"]).size().reset_index(name="count")
        except Exception:
            swaps_soc = pd.DataFrame()

        if not swaps_soc.empty:
            swaps_pivot = swaps_soc.pivot_table(index=[station_col, "date_only", "hour"], columns="soc_category", values="count", fill_value=0).reset_index()
            # normalize station column name to station_id
            if station_col != "station_id":
                swaps_pivot = swaps_pivot.rename(columns={station_col: "station_id"})
            swaps_pivot["station_id"] = swaps_pivot["station_id"].astype(str).str.strip()
            # ensure all SOC buckets present
            for b in SOC_BUCKETS_ORDER:
                if b not in swaps_pivot.columns:
                    swaps_pivot[b] = 0
            # compute total swaps
            swaps_pivot["total_swaps"] = swaps_pivot[[c for c in SOC_BUCKETS_ORDER if c in swaps_pivot.columns]].sum(axis=1).astype(int)
        else:
            swaps_pivot = pd.DataFrame()

        # per-hour stats (mean/min/max) if soc_col exists
        if soc_col and soc_col in dfs.columns:
            try:
                stats = dfs.groupby([station_col, "date_only", "hour"])[soc_col].agg(["mean", "min", "max"]).reset_index()
                stats = stats.rename(columns={station_col: "station_id", "mean": "mean_received_soc", "min": "min_received_soc", "max": "max_received_soc"})
                stats["station_id"] = stats["station_id"].astype(str).str.strip()
                swaps_stats = stats
            except Exception:
                swaps_stats = pd.DataFrame()

    # If neither has content after processing -> return empty
    if (power_hourly is None or power_hourly.empty) and (swaps_pivot is None or swaps_pivot.empty):
        return pd.DataFrame()

    # Build union grid of stations/dates/hours and merge
    try:
        stations = pd.Index(sorted(set(power_hourly["station_id"]) | set(swaps_pivot["station_id"])))
    except Exception:
        # handle cases where one is empty
        sset = set()
        if power_hourly is not None and not power_hourly.empty:
            sset |= set(power_hourly["station_id"].astype(str).unique())
        if swaps_pivot is not None and not swaps_pivot.empty:
            sset |= set(swaps_pivot["station_id"].astype(str).unique())
        stations = pd.Index(sorted(sset))

    # dates
    dset = set()
    if power_hourly is not None and not power_hourly.empty:
        dset |= set(power_hourly["date_only"].astype("O").unique())
    if swaps_pivot is not None and not swaps_pivot.empty:
        dset |= set(swaps_pivot["date_only"].astype("O").unique())
    dates = pd.Index(sorted(dset))

    # if still no stations/dates return concatenation or empty
    if len(stations) == 0 or len(dates) == 0:
        pieces = []
        if power_hourly is not None and not power_hourly.empty:
            pieces.append(power_hourly)
        if swaps_pivot is not None and not swaps_pivot.empty:
            pieces.append(swaps_pivot)
        if pieces:
            merged = pd.concat(pieces, sort=False, ignore_index=True).fillna(0)
            return merged
        return pd.DataFrame()

    hours = pd.Index(range(24))
    grid = pd.MultiIndex.from_product([stations, dates, hours], names=["station_id", "date_only", "hour"]).to_frame(index=False)

    merged = grid.merge(power_hourly, on=["station_id", "date_only", "hour"], how="left")
    if swaps_pivot is not None and not swaps_pivot.empty:
        merged = merged.merge(swaps_pivot, on=["station_id", "date_only", "hour"], how="left")
    if swaps_stats is not None and not swaps_stats.empty:
        merged = merged.merge(swaps_stats, on=["station_id", "date_only", "hour"], how="left")

    # sanitize
    merged["power_consumed"] = merged.get("power_consumed", 0.0).replace([np.nan, np.inf, -np.inf], 0.0).fillna(0.0)
    for b in SOC_BUCKETS_ORDER:
        merged[b] = merged.get(b, 0).fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    merged["total_swaps"] = merged.get("total_swaps", merged[SOC_BUCKETS_ORDER].sum(axis=1)).replace([np.nan, np.inf, -np.inf], 0).fillna(0).astype(int)
    merged["mean_received_soc"] = merged.get("mean_received_soc", np.nan)
    merged["min_received_soc"] = merged.get("min_received_soc", np.nan)
    merged["max_received_soc"] = merged.get("max_received_soc", np.nan)

    merged["station_id"] = merged["station_id"].astype(str).str.strip()
    merged["time_str"] = merged["hour"].apply(lambda h: f"{int(h):02d}:00")

    # persist cache
    try:
        merged.to_parquet(MASTER_PARQUET, index=False)
    except Exception:
        pass

    return merged

# -------------------------------
# SIDEBAR: Uploads & Cache controls
# -------------------------------
st.sidebar.header("1) Upload Data (CSV/XLSX/ZIP)")
st.sidebar.caption("Upload Station Power files and SoC/Swap files (can upload many). ZIPs with nested folders supported.")

power_files = st.sidebar.file_uploader("Station Power files (multiple)", type=["csv", "xlsx"], accept_multiple_files=True)
swaps_files = st.sidebar.file_uploader("SoC & Swap files (multiple)", type=["csv", "xlsx"], accept_multiple_files=True)
zip_files = st.sidebar.file_uploader("Optional ZIP archives (multiple)", type=["zip"], accept_multiple_files=True)

st.sidebar.markdown("---")
use_existing = st.sidebar.checkbox("Use existing master_parquet if present", value=True)
rebuild_now = st.sidebar.button("Rebuild master (merge+process) now")

# build or load master
master_df = None
if use_existing and os.path.exists(MASTER_PARQUET) and not rebuild_now and not (power_files or swaps_files or zip_files):
    try:
        master_df = pd.read_parquet(MASTER_PARQUET)
        st.sidebar.success(f"Loaded cached master → {MASTER_PARQUET}")
    except Exception:
        master_df = None

if master_df is None:
    with st.spinner("Merging & processing uploads..."):
        master_df = build_master_from_uploads(power_files or [], swaps_files or [], zip_files or [])
    if master_df is None or master_df.empty:
        st.warning("No valid merged data yet. Upload Station Power & SoC files (or ZIP), or enable existing master.")
        st.stop()

# Ensure date_only exists
if "date_only" not in master_df.columns:
    for c in ["date", "timestamp", "datetime"]:
        if c in master_df.columns:
            master_df["date_only"] = pd.to_datetime(master_df[c], errors="coerce").dt.date
            break
    if "date_only" not in master_df.columns:
        # fallback: try to coerce index to datetime
        try:
            master_df["date_only"] = pd.to_datetime(master_df.index, errors="coerce").date
        except Exception:
            master_df["date_only"] = pd.NaT

# replace inf / nan in numeric columns
num_cols = master_df.select_dtypes(include=[np.number]).columns
master_df[num_cols] = master_df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# -------------------------------
# SIDEBAR: Filters
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.header("2) Filters")

site_names = list(SITE_TO_STATIONS.keys())
site_sel = st.sidebar.multiselect("Select Site(s)", options=site_names, default= site_names if site_names else [])
# collect stations from chosen sites
selected_stations_from_sites = []
for s in site_sel:
    selected_stations_from_sites.extend(SITE_TO_STATIONS.get(s, []))
selected_stations_from_sites = sorted(list(set(selected_stations_from_sites)))

station_options = ["ALL"] + selected_stations_from_sites
stations_sel = st.sidebar.multiselect("Select Station(s) (or ALL)", options=station_options, default=["ALL"] if station_options else [])

hour_range = st.sidebar.slider("Operational hour range", 0, 23, (7, 23), step=1)

# resolve stations_to_show (intersection with master)
if "ALL" in stations_sel:
    stations_to_show = [s for s in selected_stations_from_sites if s in master_df["station_id"].unique()]
else:
    stations_to_show = [s for s in stations_sel if s in master_df["station_id"].unique()]

if not stations_to_show:
    st.warning("No stations selected or not present in data. Update SITE_TO_STATIONS or upload data.")
    st.stop()

if st.sidebar.checkbox("Show dataset columns (debug)"):
    st.sidebar.write(list(master_df.columns))
    st.sidebar.write("Unique stations (sample):", sorted(master_df["station_id"].unique())[:200])

# -------------------------------
# MAIN: Hourly View (stacked)
# -------------------------------
st.header("Hourly View — Selected Stations")
avail_dates = sorted(master_df.loc[master_df["station_id"].isin(stations_to_show), "date_only"].dropna().unique())
if not avail_dates:
    st.warning("No dates available for the selected stations.")
    st.stop()

global_date = st.date_input("Select date", value=avail_dates[0], min_value=min(avail_dates), max_value=max(avail_dates))
global_date = pd.to_datetime(global_date).date()
h_start, h_end = hour_range

station_colors = build_station_color_map(stations_to_show)

per_station_tables = {}
figs_power_swaps = []
figs_soc_power = []
labels = []

for station in stations_to_show:
    df_station = master_df[master_df["station_id"] == station].copy()
    mask = (df_station["date_only"] == global_date) & (df_station["hour"] >= h_start) & (df_station["hour"] <= h_end)
    df_view = df_station.loc[mask].copy()

    if df_view.empty:
        hrs = list(range(h_start, h_end + 1))
        df_view = pd.DataFrame({"hour": hrs, "power_consumed": [0.0]*len(hrs), "total_swaps": [0]*len(hrs)})
        for b in SOC_BUCKETS_ORDER:
            df_view[b] = 0
        df_view["Time"] = [f"{int(h):02d}:00" for h in hrs]
    else:
        df_view["Time"] = df_view["hour"].apply(lambda h: f"{int(h):02d}:00")
        df_view["power_consumed"] = df_view["power_consumed"].replace([np.nan, np.inf, -np.inf], 0.0).fillna(0.0)
        for b in SOC_BUCKETS_ORDER:
            if b in df_view.columns:
                df_view[b] = df_view[b].fillna(0).astype(int)
            else:
                df_view[b] = 0
        df_view["total_swaps"] = df_view.get("total_swaps", 0).fillna(0).astype(int)

    # charts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_view["hour"], y=df_view["power_consumed"], mode="lines+markers", name="Power Consumed (kWh)", marker=dict(color=station_colors.get(station))))
    fig1.add_trace(go.Scatter(x=df_view["hour"], y=df_view["total_swaps"], mode="lines+markers", name="Total Swaps"))
    y_max = max(1, int(max(df_view["power_consumed"].max(), df_view["total_swaps"].max())))
    fig1.update_layout(title=f"{station} — Power vs Swaps ({global_date})", xaxis=dict(tickmode="linear", dtick=1), yaxis=dict(range=[0, y_max]), height=380)
    figs_power_swaps.append((station, fig1))

    fig2 = go.Figure()
    for b in SOC_BUCKETS_ORDER:
        fig2.add_trace(go.Bar(x=df_view["hour"], y=df_view[b], name=b))
    fig2.add_trace(go.Scatter(x=df_view["hour"], y=df_view["power_consumed"], mode="lines+markers", name="Power Consumed", marker=dict(color=station_colors.get(station))))
    cat_max = int(df_view[SOC_BUCKETS_ORDER].max().max()) if len(df_view) else 0
    fig2.update_layout(barmode="group", title=f"{station} — SoC Categories + Power ({global_date})", xaxis=dict(tickmode="linear", dtick=1), yaxis=dict(range=[0, max(1, cat_max, int(df_view['power_consumed'].max() if len(df_view) else 0))]), height=380)
    figs_soc_power.append((station, fig2))

    # table
    table_cols = ["Time", "power_consumed", "total_swaps"] + SOC_BUCKETS_ORDER
    df_table = df_view[["Time", "power_consumed", "total_swaps"] + SOC_BUCKETS_ORDER].rename(columns={"power_consumed": "Power Consumed", "total_swaps": "Total Swaps"})
    totals = {"Time": "TOTAL", "Power Consumed": float(df_table["Power Consumed"].sum()), "Total Swaps": int(df_table["Total Swaps"].sum())}
    for b in SOC_BUCKETS_ORDER:
        totals[b] = int(df_table[b].sum())
    df_table = pd.concat([df_table, pd.DataFrame([totals])], ignore_index=True)
    per_station_tables[station] = df_table
    labels.append(station)

# render stacked: each station's two charts then table
for (station, fig1), (_, fig2) in zip(figs_power_swaps, figs_soc_power):
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"**{station} — Hourly Table ({global_date})**")
    st.dataframe(per_station_tables.get(station), use_container_width=True)
    csv_bytes = per_station_tables.get(station).to_csv(index=False).encode("utf-8")
    st.download_button(label=f"Download CSV — {station}", data=csv_bytes, file_name=f"swaps_vs_power_{station}_{global_date}.csv", mime="text/csv", key=f"dl_{station}")

# -------------------------------
# WEEKLY / MULTI-DAY SUMMARY & INSIGHTS
# -------------------------------
st.markdown("---")
st.header("Multi-Day / Weekly Summary & Insights")

all_dates = sorted(master_df["date_only"].dropna().unique())
if len(all_dates) == 0:
    st.info("No date data available for weekly summary.")
else:
    min_d, max_d = min(all_dates), max(all_dates)
    wk_range = st.date_input("Select date range for weekly summary (multi-day allowed)", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(wk_range, tuple) and len(wk_range) == 2:
        wd_start, wd_end = wk_range
    else:
        wd_start = wk_range
        wd_end = wk_range
    wd_start = pd.to_datetime(wd_start).date()
    wd_end = pd.to_datetime(wd_end).date()

    wk_mask = (master_df["station_id"].isin(stations_to_show)) & (master_df["date_only"] >= wd_start) & (master_df["date_only"] <= wd_end) & (master_df["hour"] >= h_start) & (master_df["hour"] <= h_end)
    wk = master_df.loc[wk_mask].copy()

    # sanitize
    wk["power_consumed"] = wk["power_consumed"].replace([np.nan, np.inf, -np.inf], 0.0).fillna(0.0)
    for b in SOC_BUCKETS_ORDER:
        if b not in wk.columns:
            wk[b] = 0
    wk[SOC_BUCKETS_ORDER] = wk[SOC_BUCKETS_ORDER].fillna(0).astype(int)
    wk["total_swaps"] = wk.get("total_swaps", 0).fillna(0).astype(int)

    if wk.empty:
        st.info("No data for the selected date range & stations.")
    else:
        # counts per day-station
        cat_counts = wk.groupby(["date_only", "station_id"], as_index=False)[SOC_BUCKETS_ORDER].sum()
        cat_counts["total_swaps_day"] = cat_counts[SOC_BUCKETS_ORDER].sum(axis=1).astype(int)

        # dominant category
        def dominant_row(r):
            vals = r[SOC_BUCKETS_ORDER].to_numpy()
            idx = int(np.nanargmax(vals))
            return pd.Series({"dominant_category": SOC_BUCKETS_ORDER[idx], "dominant_count": int(vals[idx])})
        dom = pd.concat([cat_counts, cat_counts.apply(dominant_row, axis=1)], axis=1)

        station_colors = build_station_color_map(stations_to_show)
        x_days = [d.strftime("%Y-%m-%d") for d in pd.date_range(wd_start, wd_end)]
        fig_week = go.Figure()
        for stn in stations_to_show:
            sub = dom[dom["station_id"] == stn].copy()
            lookup = {pd.to_datetime(r["date_only"]).strftime("%Y-%m-%d"):(int(r["dominant_count"]), r["dominant_category"], int(r["total_swaps_day"])) for _, r in sub.iterrows()}
            y_vals, hover = [], []
            for xd in x_days:
                if xd in lookup:
                    cnt, cat, tot = lookup[xd]
                    y_vals.append(cnt)
                    hover.append(f"SoC Category: {cat} | Swaps in Category: {cnt} | Total Swaps: {tot}")
                else:
                    y_vals.append(0)
                    hover.append("SoC Category: - | Swaps in Category: 0 | Total Swaps: 0")
            fig_week.add_trace(go.Bar(x=x_days, y=y_vals, name=stn, marker_color=station_colors.get(stn, "#999999"), hovertext=hover, hoverinfo="text"))

        fig_week.update_layout(title=f"Weekly Summary — {', '.join(site_sel) if site_sel else 'Selected Sites'}", barmode="group", xaxis=dict(title="Date"), yaxis=dict(title="Dominant SoC Category — swaps (count)"), legend=dict(orientation="h", y=1.1), height=480)
        st.plotly_chart(fig_week, use_container_width=True)

        # KPI cards
        total_power = wk["power_consumed"].sum()
        total_swaps = wk["total_swaps"].sum()
        power_per_swap = total_power / total_swaps if total_swaps > 0 else 0.0

        # choose soc numeric column if present, else approximate using bucket mids
        soc_candidates = [c for c in wk.columns if "mean_received_soc" in c or "received_soc" in c or "incoming_soc" in c or "soc_val" in c]
        soc_col = soc_candidates[0] if soc_candidates else None

        if soc_col and soc_col in wk.columns:
            mean_soc = wk[soc_col].dropna().mean()
            min_soc = wk[soc_col].dropna().min()
            max_soc = wk[soc_col].dropna().max()
        else:
            # approximate from buckets
            mids = {
                "Less than 10%": 5,
                "Between 10 - 30%": 20,
                "Between 30.1- 50%": 40,
                "Between 50.1 - 70%": 60,
                "Between 70.1 - 85%": 77.5,
                "Between 85.1 - 100%": 92.5,
            }
            bucket_sums = wk[SOC_BUCKETS_ORDER].sum()
            total_counts = bucket_sums.sum()
            if total_counts > 0:
                mean_soc = sum(bucket_sums[b] * mids[b] for b in SOC_BUCKETS_ORDER) / total_counts
                occupied = [b for b in SOC_BUCKETS_ORDER if bucket_sums[b] > 0]
                if occupied:
                    min_soc = mids[occupied[0]]
                    max_soc = mids[occupied[-1]]
                else:
                    min_soc = max_soc = np.nan
            else:
                mean_soc = min_soc = max_soc = np.nan

        if "mean_received_soc" in wk.columns and wk["total_swaps"].sum() > 0:
            weighted_mean = (wk["mean_received_soc"].fillna(0) * wk["total_swaps"]).sum() / wk["total_swaps"].sum()
        else:
            weighted_mean = mean_soc if not pd.isna(mean_soc) else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Power (kWh)", f"{total_power:.1f}")
        c2.metric("Total Swaps", f"{int(total_swaps)}")
        c3.metric("Power per Swap (kWh)", f"{power_per_swap:.3f}")
        c4.metric("Weighted Avg Received SoC (%)", f"{weighted_mean:.2f}" if not pd.isna(weighted_mean) else "N/A")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Average SoC (est)", f"{mean_soc:.2f} %" if not pd.isna(mean_soc) else "N/A")
        s2.metric("Minimum SoC (est)", f"{min_soc:.2f} %" if not pd.isna(min_soc) else "N/A")
        s3.metric("Maximum SoC (est)", f"{max_soc:.2f} %" if not pd.isna(max_soc) else "N/A")
        s4.metric("Stations considered", f"{len(stations_to_show)}")

        with st.expander("Weekly Insights & Details", expanded=True):
            st.markdown(f"- **Majority SoC category:** {wk[SOC_BUCKETS_ORDER].sum().idxmax() if wk[SOC_BUCKETS_ORDER].sum().sum()>0 else 'N/A'}")
            daily_tot = wk.groupby("date_only")["total_swaps"].sum() if "total_swaps" in wk.columns else pd.Series(dtype=int)
            if not daily_tot.empty:
                peak_day = pd.to_datetime(daily_tot.idxmax()).strftime("%Y-%m-%d")
                low_day = pd.to_datetime(daily_tot.idxmin()).strftime("%Y-%m-%d")
                st.markdown(f"- **Peak swap day:** {peak_day} ({int(daily_tot.max())} swaps)")
                st.markdown(f"- **Lowest swap day:** {low_day} ({int(daily_tot.min())} swaps)")
            st.markdown(f"- **Stations considered:** {', '.join(stations_to_show)}")
            st.markdown(f"- **Hours considered:** {h_start}:00 - {h_end}:00")
            st.markdown(f"- **Date range:** {wd_start} to {wd_end}")

        # Downloads
        weekly_agg = dom.copy() if 'dom' in locals() else wk.copy()
        weekly_csv = weekly_agg.to_csv(index=False).encode("utf-8")
        st.download_button("Download Weekly Aggregated CSV", data=weekly_csv, file_name=f"weekly_agg_{wd_start}_to_{wd_end}.csv", mime="text/csv")

        master_filtered = master_df.loc[wk_mask].copy().reset_index(drop=True)
        master_csv = master_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Master (filtered) CSV", data=master_csv, file_name=f"master_filtered_{wd_start}_to_{wd_end}.csv", mime="text/csv")

        # export weekly chart option
        cols = st.columns([1,1,1])
        png_btn = cols[0].button("Export Weekly Chart PNG")
        jpeg_btn = cols[1].button("Export Weekly Chart JPEG")
        pdf_btn = cols[2].button("Export Weekly Chart PDF")
        if (png_btn or jpeg_btn or pdf_btn) and 'fig_week' in locals():
            fmt = "png" if png_btn else ("jpeg" if jpeg_btn else "pdf")
            try:
                img_bytes = fig_week.to_image(format=fmt, engine="kaleido", scale=2)
                st.download_button(f"Download {fmt.upper()}", data=img_bytes, file_name=f"weekly_chart_{wd_start}_to_{wd_end}.{fmt}", mime=f"image/{fmt if fmt!='pdf' else 'pdf'}")
            except Exception as e:
                st.error(f"Failed to export image: {e}")

# Footer notes
with st.expander("Processing assumptions & tips", expanded=False):
    st.markdown(f"""
- **Power Consumed** is computed as hourly **diff** of cumulative Station Energy per day; negatives clipped to 0.
- **SoC buckets** used: {', '.join(SOC_BUCKETS_ORDER)}.
- ZIP reader scans filenames and file contents to classify files (be forgiving with file names).
- Cached master: `{MASTER_PARQUET}` stored in app folder; uncheck 'Use existing master_parquet' to force rebuild.
""")

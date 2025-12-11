"""
ASOS 1-Minute Data Functions
============================

Fetch TRUE per-minute precipitation and weather data from IEM.

Source: https://mesonet.agron.iastate.edu/request/asos/1min.phtml

Key feature:
- 1-min `precip` = TRUE precipitation per minute (not hourly running totals)

Note: Data is delayed 18-36 hours (not real-time) due to NCEI collection method.
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path


# =============================================================================
# STATION CONFIG
# =============================================================================

STATIONS = {
    'JFK': {'name': 'JFK Airport', 'color': '#1f77b4', 'ls': '-'},
    'LGA': {'name': 'LaGuardia Airport', 'color': '#ff7f0e', 'ls': '--'},
    'NYC': {'name': 'Central Park', 'color': '#2ca02c', 'ls': ':'}
}


# =============================================================================
# FETCH 1-MINUTE DATA
# =============================================================================

def fetch_1min_chunk(station_id, start_date, end_date, verbose=True):
    """
    Fetch 1-minute ASOS data for a single time chunk.
    """
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
    
    params = {
        'station': station_id,
        'tz': 'UTC',
        'year1': start_date.year,
        'month1': start_date.month,
        'day1': start_date.day,
        'year2': end_date.year,
        'month2': end_date.month,
        'day2': end_date.day,
        'vars': 'tmpf,dwpf,sknt,drct,gust_sknt,gust_drct,ptype,precip',
        'sample': '1min',
        'what': 'download',
        'delim': 'comma',
    }
    
    try:
        response = requests.get(url, params=params, timeout=300)
        
        if response.status_code == 200 and len(response.text) > 100:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            return None
    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
        return None


def fetch_1min_station(station_id, start_date, end_date, verbose=True):
    """
    Fetch 1-minute data for a station in monthly chunks, then merge.
    """
    if verbose:
        print(f"\n{station_id} ({STATIONS.get(station_id, {}).get('name', '')}):")
    
    chunks = []
    current = start_date.replace(day=1)
    
    while current < end_date:
        # End of this month
        next_month = current + relativedelta(months=1)
        chunk_end = min(next_month - relativedelta(days=1), end_date)
        
        if verbose:
            print(f"  {current.strftime('%Y-%m')}... ", end='', flush=True)
        
        df = fetch_1min_chunk(station_id, current, chunk_end, verbose=False)
        
        if df is not None and len(df) > 0:
            chunks.append(df)
            if verbose:
                print(f"✓ {len(df):,} rows")
        else:
            if verbose:
                print("✗ no data")
        
        current = next_month
    
    if len(chunks) == 0:
        if verbose:
            print(f"  ✗ No data retrieved for {station_id}")
        return None
    
    # Merge all chunks
    df_combined = pd.concat(chunks, ignore_index=True)
    df_combined['valid'] = pd.to_datetime(df_combined['valid(UTC)'])
    df_combined = df_combined.drop_duplicates(subset=['valid']).sort_values('valid')
    df_combined = df_combined.reset_index(drop=True)
    
    if verbose:
        print(f"  ✓ Total: {len(df_combined):,} rows")
    
    return df_combined


def fetch_all_stations_1min(station_ids, start_date, end_date, verbose=True):
    """
    Fetch 1-minute data for all stations.
    """
    if verbose:
        print("=" * 60)
        print("FETCHING 1-MINUTE ASOS DATA")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print("=" * 60)
    
    raw_data = {}
    for station_id in station_ids:
        df = fetch_1min_station(station_id, start_date, end_date, verbose)
        if df is not None:
            raw_data[station_id] = df
    
    if verbose:
        print(f"\n✓ Fetched {len(raw_data)}/{len(station_ids)} stations")
    
    return raw_data


# =============================================================================
# PRECIP TYPE MAPPING
# =============================================================================

# Raw ptype codes mapping
# Note: NCEI/IEM format is mostly undocumented. Codes inferred from METAR standards.
# Source: https://mesonet.agron.iastate.edu/request/asos/1min.phtml
PTYPE_MAP = {
    # Dry - No precipitation
    'NP': 'dry',
    # Rain (any intensity: light-, moderate, heavy+)
    'R': 'rain', 'R+': 'rain', 'R-': 'rain',
    # Snow (any intensity: light-, moderate, heavy+)
    'S': 'snow', 'S+': 'snow', 'S-': 'snow',
    # Precipitation detected but type uncertain
    'P': 'precip', 'P?': 'precip',
    # Missing or sensor error
    'M': 'missing', 'M ': 'missing',
    '?0': 'missing', '?1': 'missing', '?2': 'missing', '?3': 'missing',
}

# Category descriptions for reference
PRECIP_CATEGORIES = {
    'dry': 'No precipitation (NP)',
    'rain': 'Rain - any intensity (R, R+, R-)',
    'snow': 'Snow - any intensity (S, S+, S-)',
    'precip': 'Precipitation detected, type uncertain (P, P?)',
    'missing': 'Missing data or sensor error (M, ?0-?3)',
}


def map_precip_category(ptype):
    """
    Map raw ptype code to simplified category.
    
    Categories
    ----------
    - dry: No precipitation (NP)
    - rain: Rain of any intensity (R, R+, R-)
    - snow: Snow of any intensity (S, S+, S-)
    - precip: Precipitation detected, type uncertain (P, P?)
    - missing: Missing data or sensor error (M, ?0, ?1, ?2, ?3)
    
    Note: NCEI/IEM 1-minute data format is mostly undocumented.
    These mappings are inferred from standard METAR conventions.
    
    Returns
    -------
    str
        Category: 'dry', 'rain', 'snow', 'precip', or 'missing'
    """
    if pd.isna(ptype) or ptype in ['', 'nan', 'None']:
        return 'missing'
    
    ptype_str = str(ptype).strip()
    
    # Direct lookup
    if ptype_str in PTYPE_MAP:
        return PTYPE_MAP[ptype_str]
    
    # Pattern matching for codes not in map
    ptype_upper = ptype_str.upper()
    
    if ptype_upper == 'NP':
        return 'dry'
    if ptype_upper.startswith('R'):
        return 'rain'
    if ptype_upper.startswith('S'):
        return 'snow'
    if ptype_upper.startswith('P'):
        return 'precip'
    if ptype_upper.startswith('M') or ptype_upper.startswith('?'):
        return 'missing'
    
    return 'missing'


# =============================================================================
# CONVERT TO METRIC
# =============================================================================

def convert_to_metric(df, station_id):
    """
    Convert 1-min data to metric units.
    
    Conversions:
    - tmpf/dwpf (°F) → temp_c/dewpoint_c (°C)
    - sknt/gust_sknt (knots) → wind_speed_ms/wind_gust_ms (m/s)
    - precip (inches) → precip_mm (mm)
    - ptype → precip_type (raw code) + precip_category (simplified)
    """
    out = pd.DataFrame()
    out['datetime'] = pd.to_datetime(df['valid(UTC)'])
    out['station_id'] = station_id
    
    # Temperature: °F → °C
    if 'tmpf' in df.columns:
        out['temp_c'] = (pd.to_numeric(df['tmpf'], errors='coerce') - 32) * 5/9
    
    if 'dwpf' in df.columns:
        out['dewpoint_c'] = (pd.to_numeric(df['dwpf'], errors='coerce') - 32) * 5/9
    
    # Wind: knots → m/s
    if 'sknt' in df.columns:
        out['wind_speed_ms'] = pd.to_numeric(df['sknt'], errors='coerce') * 0.51444
    
    if 'drct' in df.columns:
        out['wind_dir_deg'] = pd.to_numeric(df['drct'], errors='coerce')
    
    # Wind gust: knots → m/s
    if 'gust_sknt' in df.columns:
        out['wind_gust_ms'] = pd.to_numeric(df['gust_sknt'], errors='coerce') * 0.51444
    
    if 'gust_drct' in df.columns:
        out['wind_gust_dir_deg'] = pd.to_numeric(df['gust_drct'], errors='coerce')
    
    # Precipitation type (raw code + simplified category)
    if 'ptype' in df.columns:
        out['precip_type'] = df['ptype'].astype(str).replace('nan', None)
        out['precip_category'] = out['precip_type'].apply(map_precip_category)
    
    # Precipitation: inches → mm
    if 'precip' in df.columns:
        out['precip_mm'] = pd.to_numeric(df['precip'], errors='coerce') * 25.4
    
    return out.sort_values('datetime').reset_index(drop=True)


def process_all_stations(raw_data, verbose=True):
    """Convert all stations to metric."""
    if verbose:
        print("\nConverting to metric...")
    
    processed = {}
    for station_id, df in raw_data.items():
        processed[station_id] = convert_to_metric(df, station_id)
        if verbose:
            n = len(processed[station_id])
            precip_sum = processed[station_id]['precip_mm'].sum()
            print(f"  {station_id}: {n:,} rows, total precip = {precip_sum:.1f} mm")
    
    if verbose:
        print("✓ Conversion complete\n")
    
    return processed


# =============================================================================
# GENERAL RESAMPLE FUNCTION
# =============================================================================

def resample_data(df, interval='5min'):
    """
    Resample data to specified interval.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'datetime' column
    interval : str
        Resampling interval (e.g., '5min', '10min', '15min', '30min', '1H')
    
    Returns
    -------
    pd.DataFrame
        Resampled data
    
    Aggregation rules:
    - precip_mm: SUM
    - temp_c, dewpoint_c: MEAN
    - wind_speed_ms, wind_dir_deg: MEAN
    - wind_gust_ms: MAX
    - wind_gust_dir_deg: MEAN
    - precip_type, precip_category: MODE (most frequent)
    """
    df = df.copy()
    df = df.set_index('datetime')
    
    agg_dict = {
        'precip_mm': 'sum',
        'temp_c': 'mean',
        'dewpoint_c': 'mean',
        'wind_speed_ms': 'mean',
        'wind_dir_deg': 'mean',
        'wind_gust_ms': 'max',
        'wind_gust_dir_deg': 'mean',
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    df_resampled = df.resample(interval).agg(agg_dict)
    
    # Handle categorical columns separately (mode)
    def get_mode(x):
        x = x.dropna()
        if len(x) == 0:
            return None
        mode = x.mode()
        return mode.iloc[0] if len(mode) > 0 else None
    
    if 'precip_type' in df.columns:
        ptype_resampled = df['precip_type'].resample(interval).apply(get_mode)
        df_resampled['precip_type'] = ptype_resampled
    
    if 'precip_category' in df.columns:
        pcat_resampled = df['precip_category'].resample(interval).apply(get_mode)
        df_resampled['precip_category'] = pcat_resampled
    
    df_resampled = df_resampled.reset_index()
    
    return df_resampled


def resample_all_stations(processed_data, interval='5min', verbose=True):
    """
    Resample all stations to specified interval.
    
    Parameters
    ----------
    processed_data : dict
        {station_id: DataFrame}
    interval : str
        Resampling interval (e.g., '5min', '10min', '15min', '30min', '1H')
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        {station_id: resampled DataFrame}
    """
    if verbose:
        print(f"Resampling to {interval} intervals...")
    
    resampled = {}
    for station_id, df in processed_data.items():
        df_res = resample_data(df, interval)
        df_res['station_id'] = station_id
        resampled[station_id] = df_res
        if verbose:
            print(f"  {station_id}: {len(df_res):,} rows")
    
    if verbose:
        print(f"✓ Resampling to {interval} complete\n")
    
    return resampled


# =============================================================================
# ACCUMULATED PRECIPITATION
# =============================================================================

def compute_accumulated(df, start_date=None, end_date=None):
    """
    Compute cumulative precipitation.
    
    Filters to date range first, then computes cumsum (starts at 0).
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    if start_date:
        df = df[df['datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['datetime'] <= pd.to_datetime(end_date)]
    
    df = df.copy()
    df['precip_mm'] = df['precip_mm'].fillna(0)
    df['accumulated_mm'] = df['precip_mm'].cumsum()
    
    return df.reset_index(drop=True)


def compute_accumulated_all(processed_data, start_date=None, end_date=None):
    """Compute accumulated for all stations."""
    accumulated = {}
    for station_id, df in processed_data.items():
        accumulated[station_id] = compute_accumulated(df, start_date, end_date)
    return accumulated


# =============================================================================
# PRECIP TYPE ANALYSIS
# =============================================================================

def get_precip_type_summary(data_dict, use_category=True, verbose=True):
    """
    Summarize precipitation types across all stations.
    
    Parameters
    ----------
    data_dict : dict
        {station_id: DataFrame}
    use_category : bool
        If True, use simplified categories (dry, rain, snow, precip, missing)
        If False, use raw ptype codes
    verbose : bool
        Print summary table
    
    Returns
    -------
    pd.DataFrame
        Summary table with counts per station and type
    """
    summaries = []
    
    # Determine which column to use
    col = 'precip_category' if use_category else 'precip_type'
    
    for station_id, df in data_dict.items():
        # If precip_category requested but doesn't exist, create it on the fly
        if col == 'precip_category' and col not in df.columns:
            if 'precip_type' in df.columns:
                df = df.copy()
                df['precip_category'] = df['precip_type'].apply(map_precip_category)
            else:
                continue
        
        if col not in df.columns:
            continue
        
        # Count types
        type_counts = df[col].value_counts().to_dict()
        
        for ptype, count in type_counts.items():
            if ptype and ptype != 'None' and ptype != 'nan':
                summaries.append({
                    'station_id': station_id,
                    'type': ptype,
                    'count': count
                })
    
    if len(summaries) == 0:
        if verbose:
            print("No precipitation type data found.")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(summaries)
    
    if verbose:
        print("\nPrecipitation Type Summary:")
        print("-" * 50)
        pivot = summary_df.pivot_table(
            index='type', 
            columns='station_id', 
            values='count', 
            fill_value=0
        )
        # Reorder rows for better display
        if use_category:
            order = ['dry', 'rain', 'snow', 'precip', 'missing']
            order = [o for o in order if o in pivot.index]
            if order:
                pivot = pivot.reindex(order)
        print(pivot.to_string())
        
        # Print percentages
        print("\nPercentages:")
        print("-" * 50)
        pct = pivot.div(pivot.sum()) * 100
        print(pct.round(1).to_string())
        print()
    
    return summary_df


def filter_by_precip_type(df, precip_types):
    """
    Filter data to specific precipitation types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'precip_type' column
    precip_types : str or list
        Type(s) to filter for (e.g., 'rain', ['rain', 'snow'])
    
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    if isinstance(precip_types, str):
        precip_types = [precip_types]
    
    return df[df['precip_type'].isin(precip_types)].copy()


# =============================================================================
# FILE I/O
# =============================================================================

def save_data(data_dict, output_dir, prefix='', 
              save_individual=True, save_combined=True, verbose=True):
    """
    Save station data to CSV files.
    
    Parameters
    ----------
    data_dict : dict
        {station_id: DataFrame}
    output_dir : str or Path
        Output directory path
    prefix : str
        Prefix for filenames (e.g., '1min', '5min')
    save_individual : bool
        Save individual station files
    save_combined : bool
        Save combined all-stations file
    verbose : bool
        Print progress and output paths
    
    Returns
    -------
    dict
        {'individual': [paths], 'combined': path or None}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {'individual': [], 'combined': None}
    
    if verbose:
        print(f"\nSaving to: {output_dir.resolve()}")
        print("-" * 40)
    
    # Individual station files
    if save_individual:
        for station_id, df in data_dict.items():
            fname = f"{station_id}_{prefix}.csv" if prefix else f"{station_id}.csv"
            fpath = output_dir / fname
            df.to_csv(fpath, index=False)
            saved_files['individual'].append(fpath)
            if verbose:
                print(f"  ✓ {fname} ({len(df):,} rows)")
    
    # Combined file
    if save_combined:
        combined = pd.concat(data_dict.values(), ignore_index=True)
        combined = combined.sort_values(['station_id', 'datetime']).reset_index(drop=True)
        fname = f"ALL_STATIONS_{prefix}.csv" if prefix else "ALL_STATIONS.csv"
        fpath = output_dir / fname
        combined.to_csv(fpath, index=False)
        saved_files['combined'] = fpath
        if verbose:
            print(f"  ✓ {fname} ({len(combined):,} rows)")
    
    if verbose:
        print("-" * 40)
        print(f"✓ Save complete: {output_dir.resolve()}\n")
    
    return saved_files


# =============================================================================
# PLOTTING
# =============================================================================

def plot_precipitation_subplots(data_dict, start_date=None, end_date=None, 
                                 figsize=(14, 10), title_prefix='',
                                 ylim=None, tick_labelsize=None, title_fontsize=12):
    """
    Plot precipitation (mm) for each station in subplots.
    
    Parameters
    ----------
    data_dict : dict
        {station_id: DataFrame}
    start_date, end_date : datetime, optional
        Filter date range
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for title
    ylim : tuple, optional
        Y-axis limits (min, max). Default: (0, auto)
    tick_labelsize : int, optional
        Font size for tick labels
    title_fontsize : int
        Font size for titles
    """
    import matplotlib.pyplot as plt
    
    n_stations = len(data_dict)
    fig, axes = plt.subplots(n_stations, 1, figsize=figsize, sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    for ax, (station_id, df) in zip(axes, data_dict.items()):
        df = df.copy()
        
        if start_date:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]
        
        cfg = STATIONS.get(station_id, {'color': 'blue', 'name': station_id})
        
        ax.plot(df['datetime'], df['precip_mm'], 
                color=cfg['color'], lw=0.5, alpha=0.7)
        
        ax.set_ylabel('Precip (mm)')
        ax.set_title(f"{station_id} ({cfg.get('name', '')})", fontsize=title_fontsize)
        ax.grid(alpha=0.3)
        
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)
        
        if tick_labelsize:
            ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    axes[-1].set_xlabel('Date')
    
    if start_date and end_date:
        date_str = f"{pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}"
    else:
        date_str = ""
    
    title = f"{title_prefix} Precipitation" if title_prefix else "Precipitation"
    if date_str:
        title += f"\n{date_str}"
    
    fig.suptitle(title, fontsize=title_fontsize)
    plt.tight_layout()
    
    return fig, axes


def plot_weather_subplots(data_dict, params=None, start_date=None, end_date=None,
                          figsize=(14, 12), title_prefix='',
                          ylims=None, tick_labelsize=None, title_fontsize=12):
    """
    Plot multiple weather parameters for all stations.
    
    Parameters
    ----------
    data_dict : dict
        {station_id: DataFrame}
    params : list, optional
        Parameters to plot. Default: ['precip_mm', 'temp_c', 'wind_speed_ms']
    start_date, end_date : datetime, optional
        Filter date range
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for title
    ylims : dict, optional
        Y-axis limits per parameter. E.g., {'precip_mm': (0, 10), 'temp_c': (-10, 30)}
    tick_labelsize : int, optional
        Font size for tick labels
    title_fontsize : int
        Font size for titles
    """
    import matplotlib.pyplot as plt
    
    if params is None:
        params = ['precip_mm', 'temp_c', 'wind_speed_ms']
    
    # Handle ylims: can be a list [ymin, ymax] (apply to all) or dict {param: (ymin, ymax)}
    if ylims is None:
        ylims_dict = {}
    elif isinstance(ylims, (list, tuple)) and len(ylims) == 2:
        # List format: apply same limits to all parameters
        ylims_dict = {param: tuple(ylims) for param in params}
    elif isinstance(ylims, dict):
        # Dict format: different limits per parameter
        ylims_dict = ylims
    else:
        ylims_dict = {}
    
    param_labels = {
        'precip_mm': 'Precipitation (mm)',
        'temp_c': 'Temperature (°C)',
        'dewpoint_c': 'Dewpoint (°C)',
        'wind_speed_ms': 'Wind Speed (m/s)',
        'wind_gust_ms': 'Wind Gust (m/s)',
        'wind_dir_deg': 'Wind Direction (°)',
        'visibility_km': 'Visibility (km)',
        'accumulated_mm': 'Accumulated Precip (mm)',
    }
    
    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, params):
        for station_id, df in data_dict.items():
            df = df.copy()
            
            if start_date:
                df = df[df['datetime'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.to_datetime(end_date)]
            
            if param not in df.columns:
                continue
            
            cfg = STATIONS.get(station_id, {'color': 'blue', 'ls': '-', 'name': station_id})
            
            ax.plot(df['datetime'], df[param], 
                    label=f"{station_id}",
                    color=cfg['color'], ls=cfg.get('ls', '-'), 
                    lw=0.8, alpha=0.7)
        
        ax.set_ylabel(param_labels.get(param, param))
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Apply ylims if specified for this parameter
        if param in ylims_dict:
            ax.set_ylim(ylims_dict[param])
        
        if tick_labelsize:
            ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    axes[-1].set_xlabel('Date')
    
    if start_date and end_date:
        date_str = f"{pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}"
    else:
        date_str = ""
    
    title = f"{title_prefix} Weather Data" if title_prefix else "Weather Data"
    if date_str:
        title += f"\n{date_str}"
    
    fig.suptitle(title, fontsize=title_fontsize)
    plt.tight_layout()
    
    return fig, axes


def plot_accumulated(accumulated_dict, start_date=None, end_date=None,
                     figsize=(14, 6), ylim=None, tick_labelsize=None, title_fontsize=12):
    """
    Plot accumulated precipitation for all stations.
    
    Parameters
    ----------
    accumulated_dict : dict
        {station_id: DataFrame} with 'accumulated_mm' column
    start_date, end_date : datetime, optional
        For title only
    figsize : tuple
        Figure size
    ylim : tuple, optional
        Y-axis limits (min, max). Default: (0, auto)
    tick_labelsize : int, optional
        Font size for tick labels
    title_fontsize : int
        Font size for titles
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for station_id, df in accumulated_dict.items():
        cfg = STATIONS.get(station_id, {'color': 'blue', 'ls': '-', 'name': station_id})
        
        ax.plot(df['datetime'], df['accumulated_mm'], 
                label=f"{station_id} ({cfg.get('name', '')})",
                color=cfg['color'], ls=cfg.get('ls', '-'), lw=2)
        
        final = df['accumulated_mm'].iloc[-1]
        ax.annotate(f'{final:.0f} mm', 
                    xy=(df['datetime'].iloc[-1], final),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=cfg['color'])
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Accumulated Precipitation (mm)')
    
    if start_date and end_date:
        date_str = f"{pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}"
        ax.set_title(f'Cumulative Precipitation\n{date_str}', fontsize=title_fontsize)
    else:
        ax.set_title('Cumulative Precipitation', fontsize=title_fontsize)
    
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)
    
    if tick_labelsize:
        ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    plt.tight_layout()
    
    return fig, ax


def plot_precip_by_type(data_dict, start_date=None, end_date=None, figsize=(14, 10),
                        ylim=None, tick_labelsize=None, title_fontsize=12,
                        use_category=True):
    """
    Plot precipitation with background colored by precip type.
    
    Parameters
    ----------
    data_dict : dict
        {station_id: DataFrame}
    start_date, end_date : datetime, optional
        Filter date range
    figsize : tuple
        Figure size
    ylim : tuple, optional
        Y-axis limits (min, max). Default: (0, auto)
    tick_labelsize : int, optional
        Font size for tick labels
    title_fontsize : int
        Font size for titles
    use_category : bool
        If True, use simplified categories. If False, use raw ptype codes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Color map for precip categories
    category_colors = {
        'dry': '#f0f0f0',       # very light gray (almost invisible)
        'rain': '#1f77b4',      # blue
        'snow': '#00FFFF',      # cyan
        'precip': '#ff7f0e',    # orange
        'missing': '#d62728',   # red
    }
    
    # Color map for raw ptype codes (fallback)
    ptype_colors = {
        'NP': '#f0f0f0',
        'R': '#1f77b4', 'R+': '#0d4a6b', 'R-': '#5aa3d0',
        'S': '#00FFFF', 'S+': '#00CCCC', 'S-': '#66FFFF',
        'P': '#ff7f0e', 'P?': '#ffb366',
        'M': '#d62728',
    }
    
    n_stations = len(data_dict)
    fig, axes = plt.subplots(n_stations, 1, figsize=figsize, sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    col = 'precip_category' if use_category else 'precip_type'
    colors = category_colors if use_category else ptype_colors
    
    for ax, (station_id, df) in zip(axes, data_dict.items()):
        df = df.copy()
        
        if start_date:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]
        
        # If precip_category doesn't exist, create it
        if col == 'precip_category' and col not in df.columns:
            if 'precip_type' in df.columns:
                df['precip_category'] = df['precip_type'].apply(map_precip_category)
            else:
                ax.plot(df['datetime'], df['precip_mm'], lw=0.5, alpha=0.7, color='black')
                continue
        
        if col not in df.columns:
            ax.plot(df['datetime'], df['precip_mm'], lw=0.5, alpha=0.7, color='black')
        else:
            # Get y max for axvspan
            if ylim:
                ymax = ylim[1]
            else:
                ymax = df['precip_mm'].max() * 1.1 if df['precip_mm'].max() > 0 else 1
            
            # Paint intervals by type (axvspan for each contiguous block)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Find contiguous intervals of each type
            df['type_change'] = (df[col] != df[col].shift()).cumsum()
            
            legend_handles = {}
            for _, group in df.groupby('type_change'):
                ptype = group[col].iloc[0]
                if ptype and ptype != 'None' and ptype != 'nan' and ptype != 'dry':
                    color = colors.get(ptype, '#333333')
                    t_start = group['datetime'].iloc[0]
                    t_end = group['datetime'].iloc[-1]
                    ax.axvspan(t_start, t_end, alpha=0.4, color=color, linewidth=0)
                    
                    if ptype not in legend_handles:
                        legend_handles[ptype] = Patch(facecolor=color, alpha=0.4, label=ptype)
            
            # Plot precip line on top
            ax.plot(df['datetime'], df['precip_mm'], lw=0.5, alpha=0.9, color='black')
            
            # Add legend
            if legend_handles:
                # Order legend: rain, snow, precip, missing
                order = ['rain', 'snow', 'precip', 'missing']
                handles = [legend_handles[k] for k in order if k in legend_handles]
                ax.legend(handles=handles, loc='upper right', fontsize=7)
        
        cfg = STATIONS.get(station_id, {'name': station_id})
        ax.set_ylabel('Precip (mm)')
        ax.set_title(f"{station_id} ({cfg.get('name', '')})", fontsize=title_fontsize)
        ax.grid(alpha=0.3)
        
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)
        
        if tick_labelsize:
            ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    axes[-1].set_xlabel('Date')
    
    if start_date and end_date:
        date_str = f"{pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}"
    else:
        date_str = ""
    
    title = "Precipitation by Type"
    if date_str:
        title += f"\n{date_str}"
    
    fig.suptitle(title, fontsize=title_fontsize)
    plt.tight_layout()
    
    return fig, axes
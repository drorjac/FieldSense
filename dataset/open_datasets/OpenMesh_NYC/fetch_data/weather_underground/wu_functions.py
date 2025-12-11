"""
Weather Underground PWS API Functions - UPDATED VERSION
Added automatic chunking for date ranges > 31 days
"""

import requests
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple

# Base URL for Weather Underground PWS API
BASE_URL = "https://api.weather.com/v2/pws"

# All available parameters in WU PWS API responses
CURRENT_PARAMS = {
    'stationID': 'Station ID',
    'obsTimeUtc': 'Observation Time (UTC)',
    'obsTimeLocal': 'Observation Time (Local)',
    'neighborhood': 'Neighborhood',
    'softwareType': 'Software Type',
    'country': 'Country',
    'solarRadiation': 'Solar Radiation (W/m²)',
    'lon': 'Longitude',
    'realtimeFrequency': 'Realtime Frequency',
    'epoch': 'Epoch Time',
    'lat': 'Latitude',
    'uv': 'UV Index',
    'winddir': 'Wind Direction (degrees)',
    'humidity': 'Humidity (%)',
    'qcStatus': 'QC Status'
}

IMPERIAL_PARAMS = {
    'temp': 'Temperature (°F)',
    'heatIndex': 'Heat Index (°F)',
    'dewpt': 'Dew Point (°F)',
    'windChill': 'Wind Chill (°F)',
    'windSpeed': 'Wind Speed (mph)',
    'windGust': 'Wind Gust (mph)',
    'pressure': 'Pressure (in)',
    'precipRate': 'Precipitation Rate (in/hr)',
    'precipTotal': 'Precipitation Total (in)',
    'elev': 'Elevation (ft)'
}

METRIC_PARAMS = {
    'temp': 'Temperature (°C)',
    'heatIndex': 'Heat Index (°C)',
    'dewpt': 'Dew Point (°C)',
    'windChill': 'Wind Chill (°C)',
    'windSpeed': 'Wind Speed (km/h)',
    'windGust': 'Wind Gust (km/h)',
    'pressure': 'Pressure (mb)',
    'precipRate': 'Precipitation Rate (mm/hr)',
    'precipTotal': 'Precipitation Total (mm)',
    'elev': 'Elevation (m)'
}

# ============================================================================
# COLUMN MAPPING CONFIGURATION
# ============================================================================

# Column mapping: original_name -> new_simple_name
# Organized by category with precipitation FIRST
WU_COLUMN_MAPPING = {
    # ========== PRECIPITATION (FIRST!) ==========
    'metric.precipRate': 'precip_rate',  # mm/h or in/h
    'metric.precipTotal': 'precip_total',  # mm or in

    # ========== TIME & LOCATION ==========
    'obsTimeLocal': 'time_local',
    'obsTimeUtc': 'time_utc',
    'epoch': 'timestamp_unix',
    'stationID': 'station_id',
    'tz': 'timezone',
    'lat': 'latitude',
    'lon': 'longitude',

    # ========== TEMPERATURE ==========
    'metric.tempHigh': 'temp_high',
    'metric.tempLow': 'temp_low',
    'metric.tempAvg': 'temp_avg',
    'metric.heatindexHigh': 'heat_index_high',
    'metric.heatindexLow': 'heat_index_low',
    'metric.heatindexAvg': 'heat_index_avg',
    'metric.windchillHigh': 'wind_chill_high',
    'metric.windchillLow': 'wind_chill_low',
    'metric.windchillAvg': 'wind_chill_avg',
    'metric.dewptHigh': 'dewpoint_high',
    'metric.dewptLow': 'dewpoint_low',
    'metric.dewptAvg': 'dewpoint_avg',

    # ========== HUMIDITY ==========
    'humidityHigh': 'humidity_high',
    'humidityLow': 'humidity_low',
    'humidityAvg': 'humidity_avg',

    # ========== WIND ==========
    'metric.windspeedHigh': 'wind_speed_high',
    'metric.windspeedLow': 'wind_speed_low',
    'metric.windspeedAvg': 'wind_speed_avg',
    'metric.windgustHigh': 'wind_gust_high',
    'metric.windgustLow': 'wind_gust_low',
    'metric.windgustAvg': 'wind_gust_avg',
    'winddirAvg': 'wind_direction_avg',

    # ========== PRESSURE ==========
    'metric.pressureMax': 'pressure_max',
    'metric.pressureMin': 'pressure_min',
    'metric.pressureTrend': 'pressure_trend',

    # ========== SOLAR/UV ==========
    'solarRadiationHigh': 'solar_radiation_high',
    'uvHigh': 'uv_index_high',

    # ========== QUALITY ==========
    'qcStatus': 'qc_status',
}

# Reversed mapping for converting back if needed
SIMPLE_TO_WU_MAPPING = {v: k for k, v in WU_COLUMN_MAPPING.items()}

# ============================================================================
# COLUMN ORDERING - PRECIPITATION FIRST!
# ============================================================================

# Define the order of columns in the clean DataFrame
COLUMN_ORDER = [
    # PRECIPITATION FIRST!
    'precip_rate',
    'precip_total',

    # Time
    'time_local',
    'time_utc',
    'timestamp_unix',

    # Location
    'station_id',
    'latitude',
    'longitude',
    'timezone',

    # Temperature
    'temp_avg',
    'temp_high',
    'temp_low',
    'dewpoint_avg',
    'dewpoint_high',
    'dewpoint_low',
    'heat_index_avg',
    'heat_index_high',
    'heat_index_low',
    'wind_chill_avg',
    'wind_chill_high',
    'wind_chill_low',

    # Humidity
    'humidity_avg',
    'humidity_high',
    'humidity_low',

    # Wind
    'wind_speed_avg',
    'wind_speed_high',
    'wind_speed_low',
    'wind_gust_avg',
    'wind_gust_high',
    'wind_gust_low',
    'wind_direction_avg',

    # Pressure
    'pressure_max',
    'pressure_min',
    'pressure_trend',

    # Solar/UV
    'solar_radiation_high',
    'uv_index_high',

    # Quality
    'qc_status',
]


# ============================================================================
# DATA CONVERSION AND METADATA FUNCTIONS
# ============================================================================

def find_project_root():
    """Find project root by looking for dataset folder"""
    from pathlib import Path
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'dataset' / 'weather stations').exists():
            return parent
    return None


def read_pws_metadata(custom_path=None):
    """
    Read PWS metadata CSV file

    Args:
        custom_path: Optional path to pws_metadata.csv

    Returns:
        DataFrame with PWS metadata
    """
    import pandas as pd
    from pathlib import Path
    
    if custom_path:
        df = pd.read_csv(custom_path)
    else:
        root = find_project_root()
        if root is None:
            raise FileNotFoundError("Could not find dataset folder. Please provide custom_path.")
        df = pd.read_csv(root / 'dataset' / 'weather stations' / 'pws_metadata.csv')

    # Clean up
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df


def get_station_list(pws_meta):
    """
    Get list of station IDs

    Args:
        pws_meta: DataFrame with PWS metadata

    Returns:
        List of station IDs
    """
    return pws_meta['Station ID'].tolist()


def convert_wu_columns(df, keep_original=False):
    """
    Convert Weather Underground DataFrame columns to simple names
    Precipitation columns come first!

    Args:
        df: pandas DataFrame with WU data
        keep_original: If True, keep original columns with '_orig' suffix

    Returns:
        DataFrame with renamed columns
    """
    import pandas as pd

    # Make a copy
    df_clean = df.copy()

    # Keep original columns if requested
    if keep_original:
        for old_col, new_col in WU_COLUMN_MAPPING.items():
            if old_col in df_clean.columns:
                df_clean[f'{new_col}_orig'] = df_clean[old_col]

    # Rename columns
    df_clean = df_clean.rename(columns=WU_COLUMN_MAPPING)

    # Reorder columns (precipitation first!)
    available_cols = [col for col in COLUMN_ORDER if col in df_clean.columns]
    other_cols = [col for col in df_clean.columns if col not in available_cols]

    # Precipitation columns first, then rest in order, then any extras
    df_clean = df_clean[available_cols + other_cols]

    return df_clean


def create_metadata_df(df):
    """
    Create a metadata DataFrame with column descriptions

    Args:
        df: pandas DataFrame with clean column names

    Returns:
        DataFrame with metadata (column name, description, units, data type)
    """
    import pandas as pd

    metadata = {
        # Precipitation
        'precip_rate': ('Precipitation Rate', 'mm/h or in/h', 'float'),
        'precip_total': ('Precipitation Total', 'mm or in', 'float'),

        # Time
        'time_local': ('Observation Time (Local)', 'datetime', 'datetime'),
        'time_utc': ('Observation Time (UTC)', 'datetime', 'datetime'),
        'timestamp_unix': ('Unix Timestamp', 'seconds since epoch', 'int'),

        # Location
        'station_id': ('Weather Station ID', 'string', 'object'),
        'latitude': ('Latitude', 'degrees', 'float'),
        'longitude': ('Longitude', 'degrees', 'float'),
        'timezone': ('Time Zone', 'string', 'object'),

        # Temperature
        'temp_avg': ('Temperature Average', '°C or °F', 'float'),
        'temp_high': ('Temperature High', '°C or °F', 'float'),
        'temp_low': ('Temperature Low', '°C or °F', 'float'),
        'dewpoint_avg': ('Dew Point Average', '°C or °F', 'float'),
        'dewpoint_high': ('Dew Point High', '°C or °F', 'float'),
        'dewpoint_low': ('Dew Point Low', '°C or °F', 'float'),
        'heat_index_avg': ('Heat Index Average', '°C or °F', 'float'),
        'heat_index_high': ('Heat Index High', '°C or °F', 'float'),
        'heat_index_low': ('Heat Index Low', '°C or °F', 'float'),
        'wind_chill_avg': ('Wind Chill Average', '°C or °F', 'float'),
        'wind_chill_high': ('Wind Chill High', '°C or °F', 'float'),
        'wind_chill_low': ('Wind Chill Low', '°C or °F', 'float'),

        # Humidity
        'humidity_avg': ('Humidity Average', '%', 'float'),
        'humidity_high': ('Humidity High', '%', 'float'),
        'humidity_low': ('Humidity Low', '%', 'float'),

        # Wind
        'wind_speed_avg': ('Wind Speed Average', 'km/h or mph', 'float'),
        'wind_speed_high': ('Wind Speed High', 'km/h or mph', 'float'),
        'wind_speed_low': ('Wind Speed Low', 'km/h or mph', 'float'),
        'wind_gust_avg': ('Wind Gust Average', 'km/h or mph', 'float'),
        'wind_gust_high': ('Wind Gust High', 'km/h or mph', 'float'),
        'wind_gust_low': ('Wind Gust Low', 'km/h or mph', 'float'),
        'wind_direction_avg': ('Wind Direction Average', 'degrees (0-360)', 'float'),

        # Pressure
        'pressure_max': ('Pressure Maximum', 'mb or inHg', 'float'),
        'pressure_min': ('Pressure Minimum', 'mb or inHg', 'float'),
        'pressure_trend': ('Pressure Trend', 'mb or inHg', 'float'),

        # Solar/UV
        'solar_radiation_high': ('Solar Radiation High', 'W/m²', 'float'),
        'uv_index_high': ('UV Index High', 'index', 'float'),

        # Quality
        'qc_status': ('Quality Control Status', 'status code', 'int'),
    }

    # Build metadata DataFrame
    meta_data = []
    for col in df.columns:
        if col in metadata:
            desc, units, dtype = metadata[col]
            meta_data.append({
                'column': col,
                'description': desc,
                'units': units,
                'data_type': dtype,
                'has_data': df[col].notna().any(),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
            })
        else:
            # Unknown column
            meta_data.append({
                'column': col,
                'description': 'Unknown',
                'units': 'Unknown',
                'data_type': str(df[col].dtype),
                'has_data': df[col].notna().any(),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
            })

    return pd.DataFrame(meta_data)


def print_column_comparison(df_original, df_clean):
    """
    Print a comparison of original vs clean column names

    Args:
        df_original: Original DataFrame
        df_clean: Cleaned DataFrame
    """
    print("=" * 80)
    print("COLUMN NAME MAPPING")
    print("=" * 80)
    print(f"\n{'Original Column':<40} → {'Clean Column':<30}")
    print("-" * 80)

    for old_col, new_col in WU_COLUMN_MAPPING.items():
        if old_col in df_original.columns:
            marker = "✅" if new_col in df_clean.columns else "❌"
            print(f"{marker} {old_col:<38} → {new_col:<30}")

    print("\n" + "=" * 80)
    print(f"Total columns: {len(df_original.columns)} → {len(df_clean.columns)}")
    print("=" * 80)


def get_current_conditions(api_key: str, station_id: str, units: str = 'e') -> Optional[Dict]:
    """
    Fetch current weather conditions for a station

    Args:
        api_key: Weather Underground API key
        station_id: Station ID (e.g., 'KNYNEWYO1127')
        units: 'e' for English, 'm' for metric

    Returns:
        Dictionary with current conditions or None if error
    """
    url = f"{BASE_URL}/observations/current"
    params = {
        'stationId': station_id,
        'format': 'json',
        'units': units,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'observations' in data and len(data['observations']) > 0:
            return data
        else:
            print(f"No observations found for station {station_id}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching current conditions for {station_id}: {e}")
        return None


def get_historical_data_chunk(api_key: str, station_id: str, start_date: datetime,
                               end_date: datetime, units: str = 'e',
                               data_type: str = 'hourly') -> Optional[Dict]:
    """
    Fetch historical data for a date range (max 31 days per chunk)

    Args:
        api_key: Weather Underground API key
        station_id: Station ID
        start_date: Start datetime
        end_date: End datetime
        units: 'e' for English, 'm' for metric
        data_type: 'hourly', 'all', or 'daily'

    Returns:
        Dictionary with historical observations or None if error
    """
    url = f"{BASE_URL}/history/{data_type}"

    # Format dates as YYYYMMDD
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    params = {
        'stationId': station_id,
        'format': 'json',
        'units': units,
        'startDate': start_str,
        'endDate': end_str,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'observations' in data:
            return data
        else:
            print(f"No historical data found for {station_id} ({start_str} to {end_str})")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data for {station_id} ({start_str} to {end_str}): {e}")
        return None


def get_historical_data_multi_chunk(api_key: str, station_id: str, start_date: datetime,
                                    end_date: datetime, units: str = 'e',
                                    data_type: str = 'hourly') -> Optional[Dict]:
    """
    Fetch historical data for any date range by automatically chunking into 31-day segments

    Args:
        api_key: Weather Underground API key
        station_id: Station ID
        start_date: Start datetime
        end_date: End datetime
        units: 'e' for English, 'm' for metric
        data_type: 'hourly', 'all', or 'daily'

    Returns:
        Dictionary with combined historical observations or None if error
    """
    days_diff = (end_date - start_date).days
    
    # If <= 31 days, just make one call
    if days_diff <= 31:
        return get_historical_data_chunk(api_key, station_id, start_date, end_date, units, data_type)
    
    # Otherwise, split into chunks
    print(f"  📦 Date range spans {days_diff} days - splitting into chunks of max 31 days...")
    
    all_observations = []
    current_start = start_date
    chunk_num = 1
    
    while current_start < end_date:
        # Calculate end of this chunk (max 31 days from start)
        chunk_end = min(current_start + timedelta(days=31), end_date)
        
        print(f"  → Chunk {chunk_num}: {current_start.date()} to {chunk_end.date()}")
        
        # Fetch this chunk
        chunk_data = get_historical_data_chunk(api_key, station_id, current_start, chunk_end, units, data_type)
        
        if chunk_data and 'observations' in chunk_data:
            chunk_obs = chunk_data['observations']
            all_observations.extend(chunk_obs)
            print(f"    ✓ Got {len(chunk_obs)} observations")
        else:
            print(f"    ✗ No data for this chunk")
        
        # Move to next chunk
        current_start = chunk_end + timedelta(days=1)
        chunk_num += 1
    
    if all_observations:
        # Return combined data in same format as single chunk
        combined_data = {
            'observations': all_observations
        }
        print(f"  ✅ Total observations combined: {len(all_observations)}")
        return combined_data
    else:
        print(f"  ❌ No observations retrieved for entire date range")
        return None


def fetch_all_data(api_key: str, station_ids: List[str], start_date: datetime,
                   end_date: datetime, units: str = 'e',
                   fetch_options: Dict = None) -> Dict[str, Dict]:
    """
    Fetch all available data for multiple stations
    NOW WITH AUTOMATIC CHUNKING FOR LONG DATE RANGES!

    Args:
        api_key: Weather Underground API key
        station_ids: List of station IDs
        start_date: Start datetime for historical data
        end_date: End datetime for historical data
        units: 'e' for English, 'm' for metric
        fetch_options: Dict specifying what to fetch (current, rapid, hourly, daily)

    Returns:
        Dictionary with all data organized by station
    """
    if fetch_options is None:
        fetch_options = {'current': True, 'rapid': True, 'hourly': True, 'daily': True}

    all_data = {}

    for station_id in station_ids:
        print(f"\n{'=' * 60}")
        print(f"Fetching data for station: {station_id}")
        print(f"{'=' * 60}")

        station_data = {
            'station_id': station_id,
            'current': None,
            'rapid': None,
            'hourly': None,
            'daily': None,
            'fetch_time': datetime.now().isoformat()
        }

        # Fetch current conditions
        if fetch_options.get('current', False):
            print(f"  → Fetching current conditions...")
            current = get_current_conditions(api_key, station_id, units)
            if current:
                station_data['current'] = current
                print(f"    ✓ Current conditions fetched")

        # Fetch rapid history (high resolution - last 24 hours only)
        if fetch_options.get('rapid', False):
            print(f"  → Fetching rapid history (high resolution, last 24h only)...")
            rapid = get_rapid_history(api_key, station_id, units)
            if rapid:
                station_data['rapid'] = rapid
                obs_count = len(rapid.get('observations', []))
                print(f"    ✓ Rapid history fetched ({obs_count} observations)")

        # Fetch historical data with automatic chunking
        if fetch_options.get('historical', False):
            print(f"  → Fetching historical data ({start_date.date()} to {end_date.date()})...")
            historical = get_historical_data_multi_chunk(api_key, station_id, start_date, end_date, units, 'hourly')
            if historical:
                station_data['rapid'] = historical  # Store in 'rapid' key for backwards compatibility
                obs_count = len(historical.get('observations', []))
                print(f"    ✓ Historical data fetched ({obs_count} total observations)")

        # Fetch hourly history (limited to 7 days by API)
        if fetch_options.get('hourly', False):
            days_diff = (end_date - start_date).days
            if days_diff > 7:
                print(f"  ⚠ Hourly data limited to last 7 days (requested {days_diff} days)")
                hourly_start = end_date - timedelta(days=7)
            else:
                hourly_start = start_date

            print(f"  → Fetching hourly history ({hourly_start.date()} to {end_date.date()})...")
            hourly = get_hourly_history(api_key, station_id, hourly_start, end_date, units)
            if hourly:
                station_data['hourly'] = hourly
                obs_count = len(hourly.get('observations', []))
                print(f"    ✓ Hourly history fetched ({obs_count} observations)")

        # Fetch daily summary
        if fetch_options.get('daily', False):
            days_diff = (end_date - start_date).days
            summary_days = min(days_diff, 7)
            print(f"  → Fetching daily summary ({summary_days} days)...")
            daily = get_daily_summary(api_key, station_id, summary_days, units)
            if daily:
                station_data['daily'] = daily
                summary_count = len(daily.get('summaries', []))
                print(f"    ✓ Daily summary fetched ({summary_count} days)")

        all_data[station_id] = station_data

    return all_data


def get_hourly_history(api_key: str, station_id: str, start_date: datetime,
                       end_date: datetime, units: str = 'e') -> Optional[Dict]:
    """
    Fetch hourly historical observations (max 7 days)

    Args:
        api_key: Weather Underground API key
        station_id: Station ID
        start_date: Start datetime
        end_date: End datetime
        units: 'e' for English, 'm' for metric

    Returns:
        Dictionary with hourly observations or None if error
    """
    url = f"{BASE_URL}/observations/hourly/7day"
    params = {
        'stationId': station_id,
        'format': 'json',
        'units': units,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'observations' in data:
            # Filter by date range
            observations = data['observations']
            filtered_obs = []

            for obs in observations:
                obs_time = datetime.strptime(obs['obsTimeLocal'][:19], '%Y-%m-%d %H:%M:%S')
                if start_date <= obs_time <= end_date:
                    filtered_obs.append(obs)

            data['observations'] = filtered_obs
            return data
        else:
            print(f"No observations found for station {station_id}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching hourly history for {station_id}: {e}")
        return None


def get_daily_summary(api_key: str, station_id: str, num_days: int = 7,
                      units: str = 'e') -> Optional[Dict]:
    """
    Fetch daily summary (max 7 days)

    Args:
        api_key: Weather Underground API key
        station_id: Station ID
        num_days: Number of days (1-7)
        units: 'e' for English, 'm' for metric

    Returns:
        Dictionary with daily summaries or None if error
    """
    # API endpoint supports 7day, 30day
    endpoint = '7day' if num_days <= 7 else '30day'
    url = f"{BASE_URL}/dailysummary/{endpoint}"

    params = {
        'stationId': station_id,
        'format': 'json',
        'units': units,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'summaries' in data:
            # Limit to requested number of days
            data['summaries'] = data['summaries'][:num_days]
            return data
        else:
            print(f"No summary data found for station {station_id}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching daily summary for {station_id}: {e}")
        return None


def get_rapid_history(api_key: str, station_id: str, units: str = 'e') -> Optional[Dict]:
    """
    Fetch rapid history - high resolution data for last 24 hours (5-10 minute intervals)

    Args:
        api_key: Weather Underground API key
        station_id: Station ID
        units: 'e' for English, 'm' for metric

    Returns:
        Dictionary with rapid history observations or None if error
    """
    url = f"{BASE_URL}/observations/all/1day"
    params = {
        'stationId': station_id,
        'format': 'json',
        'units': units,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'observations' in data:
            return data
        else:
            print(f"No rapid history found for station {station_id}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching rapid history for {station_id}: {e}")
        return None


def save_to_json(data: Dict, filename: str, output_dir: str = 'output') -> bool:
    """
    Save data to a JSON file

    Args:
        data: Data to save
        filename: Output filename
        output_dir: Output directory (default: 'output')

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Data saved to {filepath}")
        return True

    except Exception as e:
        print(f"✗ Error saving to file: {e}")
        return False


def print_current_summary(data: Dict, units: str = 'e'):
    """Print a summary of current conditions"""
    if not data or 'observations' not in data:
        return

    obs = data['observations'][0]
    unit_type = 'imperial' if units == 'e' else 'metric'
    measurements = obs.get(unit_type, {})

    print(f"\n  Station: {obs.get('stationID', 'N/A')}")
    print(f"  Time: {obs.get('obsTimeLocal', 'N/A')}")
    print(f"  Temperature: {measurements.get('temp', 'N/A')}°{'F' if units == 'e' else 'C'}")
    print(f"  Humidity: {obs.get('humidity', 'N/A')}%")
    print(f"  Wind Speed: {measurements.get('windSpeed', 'N/A')} {'mph' if units == 'e' else 'km/h'}")
    print(f"  Pressure: {measurements.get('pressure', 'N/A')} {'in' if units == 'e' else 'mb'}")


def get_all_available_parameters() -> Dict[str, Dict[str, str]]:
    """
    Get all available parameters that can be read from the API

    Returns:
        Dictionary containing all parameter categories
    """
    return {
        'current_observation': CURRENT_PARAMS,
        'imperial_units': IMPERIAL_PARAMS,
        'metric_units': METRIC_PARAMS
    }


def print_available_parameters():
    """Print all available parameters"""
    params = get_all_available_parameters()

    print("\n" + "=" * 60)
    print("AVAILABLE WEATHER PARAMETERS")
    print("=" * 60)

    print("\n📊 CURRENT OBSERVATION PARAMETERS:")
    for key, label in params['current_observation'].items():
        print(f"  • {key:20} → {label}")

    print("\n🇺🇸 IMPERIAL UNIT PARAMETERS:")
    for key, label in params['imperial_units'].items():
        print(f"  • {key:20} → {label}")

    print("\n🌍 METRIC UNIT PARAMETERS:")
    for key, label in params['metric_units'].items():
        print(f"  • {key:20} → {label}")

    print("\n" + "=" * 60)


def validate_date_range(start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
    """
    Validate and adjust date range

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        Tuple of validated (start_date, end_date)
    """
    now = datetime.now()

    # Ensure end_date is not in the future
    if end_date > now:
        print(f"⚠ End date adjusted from {end_date} to {now} (cannot be in future)")
        end_date = now

    # Ensure start_date is before end_date
    if start_date >= end_date:
        print(f"⚠ Start date must be before end date")
        start_date = end_date - timedelta(days=7)

    # Info about date range
    days_diff = (end_date - start_date).days
    if days_diff > 31:
        print(f"ℹ️  Note: {days_diff} days requested. Will automatically split into chunks of 31 days.")

    return start_date, end_date


def export_wu_data(
    results: Dict,
    output_dir: str,
    start_date: datetime,
    end_date: datetime,
    export_all: bool = False
) -> None:
    """
    Export WU data to multiple formats.
    
    Args:
        results: Results dictionary from run_wu_pipeline
        output_dir: Output directory
        start_date: Start date for filename
        end_date: End date for filename
        export_all: If True, export all formats (CSV, JSON, metadata)
    """
    import os
    from pathlib import Path
    
    if results is None:
        print("❌ No data to export")
        return
    
    if not export_all:
        print("ℹ️ Export skipped (export_all=False)")
        return
    
    print("💾 EXPORTING DATA")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    all_dfs = results.get('dataframes', {})
    all_metadata = results.get('metadata', {})
    all_data = results.get('raw_data', {})
    
    # Export each station
    for station_id, dfs in all_dfs.items():
        df_clean = dfs.get('clean')
        if df_clean is None or len(df_clean) == 0:
            continue
        
        # CSV - Clean data
        csv_file = output_path / f'weather_clean_{station_id}_{date_str}.csv'
        df_clean.to_csv(csv_file, index=False)
        print(f"✅ CSV: {csv_file}")
        
        # CSV - Precipitation only
        precip_cols = ['time_local', 'precip_rate', 'precip_total']
        available = [c for c in precip_cols if c in df_clean.columns]
        if len(available) > 0:
            precip_file = output_path / f'precipitation_{station_id}_{date_str}.csv'
            df_clean[available].to_csv(precip_file, index=False)
            print(f"✅ Precipitation CSV: {precip_file}")
        
        # Metadata
        if station_id in all_metadata:
            meta_file = output_path / f'metadata_{station_id}_{date_str}.csv'
            all_metadata[station_id].to_csv(meta_file, index=False)
            print(f"✅ Metadata: {meta_file}")
    
    # JSON - All raw data
    if all_data:
        json_file = f'weather_all_{date_str}.json'
        save_to_json(all_data, json_file, str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)


def run_wu_pipeline(
    api_key: str,
    station_ids: List[str],
    start_date: datetime,
    end_date: datetime,
    units: str = 'm',
    fetch_options: Dict = None,
    output_dir: Optional[str] = None,
    save_data: bool = False
) -> Dict:
    """
    High-level wrapper function to run the complete Weather Underground pipeline:
    1. Fetch data
    2. Convert to DataFrames with clean names
    3. Return organized results
    
    Args:
        api_key: Weather Underground API key
        station_ids: List of station IDs to fetch
        start_date: Start datetime
        end_date: End datetime
        units: 'm' for metric, 'e' for imperial
        fetch_options: Dict specifying what to fetch (default: historical only)
        output_dir: Directory to save data (optional)
        save_data: Whether to save data to files
        
    Returns:
        Dictionary with:
            - 'raw_data': Original API responses
            - 'dataframes': Dict of DataFrames by station (raw and clean)
            - 'metadata': Metadata DataFrames
            - 'summary': Summary statistics
    """
    import pandas as pd
    
    if fetch_options is None:
        fetch_options = {'historical': True}
    
    # Step 1: Validate dates
    start, end = validate_date_range(start_date, end_date)
    
    # Step 2: Fetch data
    print("=" * 70)
    print("FETCHING DATA FROM WEATHER UNDERGROUND API")
    print("=" * 70)
    fetch_start_time = datetime.now()
    
    all_data = fetch_all_data(api_key, station_ids, start, end, units, fetch_options)
    
    fetch_duration = (datetime.now() - fetch_start_time).total_seconds()
    
    if all_data:
        total_obs = sum(len(d.get('rapid', {}).get('observations', [])) 
                       if 'rapid' in d and d['rapid'] 
                       else len(d.get('historical', {}).get('observations', [])) 
                       if 'historical' in d and d['historical']
                       else 0
                       for d in all_data.values())
        print(f"✓ Success: Fetched {total_obs:,} observations in {fetch_duration:.1f}s")
        print(f"✓ Stations processed: {len(all_data)}/{len(station_ids)}")
    else:
        print("✗ Error: No data fetched")
        return None
    
    print("=" * 70)
    
    # Step 3: Convert to DataFrames
    print("\n" + "=" * 70)
    print("CONVERTING TO DATAFRAME WITH CLEAN NAMES")
    print("=" * 70)
    
    all_dfs = {}
    all_metadata = {}
    
    for station_id, station_data in all_data.items():
        print(f"\nProcessing station: {station_id}")
        
        # Find which data type was fetched
        for data_type in ['current', 'rapid', 'historical', 'hourly', 'daily']:
            if data_type in station_data and station_data[data_type] is not None:
                data = station_data[data_type]
                
                # Extract observations
                if isinstance(data, dict):
                    observations = data.get('observations', [data])
                elif isinstance(data, list):
                    observations = data
                else:
                    continue
                
                if not observations:
                    continue
                
                # Create DataFrame with original names
                df_raw = pd.json_normalize(observations) if isinstance(observations, list) else pd.DataFrame([observations])
                
                if 'obsTimeLocal' in df_raw.columns:
                    df_raw['obsTimeLocal'] = pd.to_datetime(df_raw['obsTimeLocal'])
                if 'obsTimeUtc' in df_raw.columns:
                    df_raw['obsTimeUtc'] = pd.to_datetime(df_raw['obsTimeUtc'])
                
                df_raw = df_raw.sort_values('obsTimeLocal').reset_index(drop=True) if 'obsTimeLocal' in df_raw.columns else df_raw
                
                # Convert to clean names
                df_clean = convert_wu_columns(df_raw)
                metadata = create_metadata_df(df_clean)
                
                all_dfs[station_id] = {
                    'raw': df_raw,
                    'clean': df_clean
                }
                all_metadata[station_id] = metadata
                
                print(f"  ✓ Created DataFrame: {len(df_clean)} rows, {len(df_clean.columns)} columns")
                break
    
    print("\n" + "=" * 70)
    print(f"✓ Processed {len(all_dfs)} station(s)")
    print("=" * 70)
    
    # Step 4: Save data if requested
    if save_data and output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each station's clean data
        for station_id, dfs in all_dfs.items():
            csv_file = output_path / f"{station_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            dfs['clean'].to_csv(csv_file, index=False)
            print(f"  ✓ Saved: {csv_file}")
    
    # Return organized results
    return {
        'raw_data': all_data,
        'dataframes': all_dfs,
        'metadata': all_metadata,
        'summary': {
            'stations': list(all_dfs.keys()),
            'total_observations': total_obs if all_data else 0,
            'date_range': (start, end),
            'units': units
        },
        'validated_dates': (start, end)  # For quality checks
    }

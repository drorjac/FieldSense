"""
Weather Underground Plotting and Analysis Functions
===================================================

Functions for visualizing and analyzing WU PWS data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from typing import Optional, Dict, List
import numpy as np

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_precipitation_analysis_multi(station_data: Dict, units: str = 'm', show: bool = False, 
                                       max_stations: int = 4, verbose: bool = False):
    """
    Plot precipitation for multiple stations in one figure (sample of 3-4 stations).
    
    Args:
        station_data: Dictionary with station_id -> {'clean': df, ...}
        units: 'm' for metric, 'e' for imperial
        show: Whether to display the plot
        max_stations: Maximum number of stations to plot (default 4)
        verbose: If True, print detailed statistics
    """
    if not station_data or len(station_data) == 0:
        print("❌ No station data available")
        return
    
    # Select sample stations (first max_stations)
    station_ids = list(station_data.keys())[:max_stations]
    unit = 'mm/h' if units == 'm' else 'in/h'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(station_ids)))
    
    for i, station_id in enumerate(station_ids):
        df = station_data[station_id]['clean']
        precip_col = 'precip_rate' if 'precip_rate' in df.columns else 'precip_total'
        
        if precip_col in df.columns:
            total = df[precip_col].sum()
            ax.plot(df['time_local'], df[precip_col], linewidth=1.5, 
                   label=f'{station_id} ({total:.1f} {unit.split("/")[0]})',
                   color=colors[i], alpha=0.8)
    
    ax.set_ylabel(f'Precipitation Rate ({unit})', fontweight='bold', fontsize=11)
    ax.set_xlabel('Date', fontweight='bold', fontsize=11)
    ax.set_title(f'💧 Precipitation: Sample Stations ({len(station_ids)} of {len(station_data)})', 
                fontsize=13, fontweight='bold')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.4, color='#999999', linestyle='--')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    if not verbose:
        for station_id in station_ids:
            df = station_data[station_id]['clean']
            precip_col = 'precip_rate' if 'precip_rate' in df.columns else 'precip_total'
            if precip_col in df.columns:
                total = df[precip_col].sum()
                events = len(df[df[precip_col] > 0])
                print(f"💧 {station_id}: {total:.1f} {unit.split('/')[0]} total, {events} events")


def plot_cumulative_precipitation_multi(station_data: Dict, units: str = 'm', show: bool = False,
                                        max_stations: int = 4, verbose: bool = False):
    """
    Plot cumulative precipitation for multiple stations in one figure (sample of 3-4 stations).
    
    Args:
        station_data: Dictionary with station_id -> {'clean': df, ...}
        units: 'm' for metric, 'e' for imperial
        show: Whether to display the plot
        max_stations: Maximum number of stations to plot (default 4)
        verbose: If True, print detailed statistics
    """
    if not station_data or len(station_data) == 0:
        print("❌ No station data available")
        return
    
    # Select sample stations (first max_stations)
    station_ids = list(station_data.keys())[:max_stations]
    unit = 'mm' if units == 'm' else 'in'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(station_ids)))
    
    for i, station_id in enumerate(station_ids):
        df = station_data[station_id]['clean']
        precip_col = 'precip_rate' if 'precip_rate' in df.columns else 'precip_total'
        
        if precip_col in df.columns:
            df_plot = df.copy()
            df_plot['cumulative'] = df_plot[precip_col].cumsum()
            total = df_plot['cumulative'].iloc[-1]
            
            ax.plot(df_plot['time_local'], df_plot['cumulative'], linewidth=2,
                   label=f'{station_id} ({total:.1f} {unit})',
                   color=colors[i], alpha=0.8)
    
    ax.set_ylabel(f'Cumulative Precipitation ({unit})', fontweight='bold', fontsize=11)
    ax.set_xlabel('Date', fontweight='bold', fontsize=11)
    ax.set_title(f'💧 Cumulative Precipitation: Sample Stations ({len(station_ids)} of {len(station_data)})',
                fontsize=13, fontweight='bold')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.4, color='#999999', linestyle='--')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    if not verbose:
        for station_id in station_ids:
            df = station_data[station_id]['clean']
            precip_col = 'precip_rate' if 'precip_rate' in df.columns else 'precip_total'
            if precip_col in df.columns:
                total = df[precip_col].sum()
                events = len(df[df[precip_col] > 0])
                print(f"📈 {station_id}: {total:.1f} {unit} total, {events} events")


def plot_precipitation_analysis(df: pd.DataFrame, station_id: str, units: str = 'm', show: bool = False, verbose: bool = False, max_stations: int = 4):
    """
    High-contrast precipitation visualization and analysis.
    
    Args:
        df: DataFrame with clean column names
        station_id: Station ID for title
        units: 'm' for metric, 'e' for imperial
        show: Whether to display the plot
        verbose: If True, print detailed statistics
        max_stations: Not used in single-station mode
    """
    if verbose:
        print("🔍 PRECIPITATION DATA ANALYSIS")

    if df is None or len(df) == 0:
        print("❌ No DataFrame available")
        return

    # Check for precipitation columns
    precip_cols = [col for col in df.columns if 'precip' in col.lower()]

    if len(precip_cols) == 0:
        print("❌ No precipitation columns found")
        return

    if verbose:
        print(f"✅ Found {len(precip_cols)} precipitation column(s): {precip_cols}")
        # Analyze each column
        for col in precip_cols:
            values = df[col]
            non_zero = values[values > 0]
            print(f"\n📊 {col}:")
            print(f"  • Total values: {len(values)}")
            print(f"  • Non-zero: {len(non_zero)} ({len(non_zero)/len(values)*100:.1f}%)")
            print(f"  • Min: {values.min():.3f}, Max: {values.max():.3f}, Sum: {values.sum():.3f}")
            if len(non_zero) > 0:
                print(f"  • Mean (non-zero): {non_zero.mean():.3f}")

    # Create visualization
    precip_col = 'precip_rate' if 'precip_rate' in df.columns else precip_cols[0]
    unit = 'mm/h' if units == 'm' else 'in/h'

    if df[precip_col].max() > 0:
        # Simple single plot
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(df['time_local'], df[precip_col], width=0.05, color='#0066CC', 
               edgecolor='#003366', alpha=0.8, linewidth=0.5)
        ax.set_ylabel(f'Precipitation Rate ({unit})', fontweight='bold', fontsize=11)
        ax.set_xlabel('Date', fontweight='bold', fontsize=11)
        ax.set_title(f'💧 Precipitation: {station_id}', fontsize=13, fontweight='bold')
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.4, color='#999999', linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        if show:
            plt.show()

        # Summary statistics
        total_precip = df[precip_col].sum()
        max_rate = df[precip_col].max()
        rain_events = df[df[precip_col] > 0]
        if verbose:
            print(f"\n📊 Statistics: Total={total_precip:.2f} {unit}, Max={max_rate:.2f} {unit}, Events={len(rain_events)}")
            if len(rain_events) > 0:
                print(f"⏱️  Period: {rain_events['time_local'].min()} to {rain_events['time_local'].max()}")
                daily = df.groupby(df['time_local'].dt.date)[precip_col].sum().sort_values(ascending=False)
                print(f"📅 Top days: {', '.join([f'{d}: {t:.1f}' for d, t in list(daily.head(5).items())])}")
        else:
            print(f"💧 {station_id}: {total_precip:.1f} {unit} total, {len(rain_events)} events, max {max_rate:.1f} {unit}")

    else:
        # NO PRECIPITATION
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(df['time_local'], [0.5] * len(df), width=0.05, color='#DDDDDD',
               edgecolor='#999999', alpha=0.7, linewidth=0.8)
        ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8, zorder=5)
        ax.set_ylabel(f'Precipitation Rate ({unit})', fontweight='bold', fontsize=11)
        ax.set_xlabel('Date', fontweight='bold', fontsize=11)
        ax.set_title(f'💧 Precipitation: {station_id} - DRY PERIOD', fontsize=13, fontweight='bold', color='red')
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.4, color='#999999', linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylim(-0.5, 5)
        plt.tight_layout()
        if show:
            plt.show()

        if verbose:
            print("⚠️  DRY PERIOD - NO PRECIPITATION DETECTED")
            print(f"📅 Period: {df['time_local'].min().date()} to {df['time_local'].max().date()}")
        else:
            print(f"💧 {station_id}: No precipitation (0.00 {unit})")


def plot_cumulative_precipitation(df: pd.DataFrame, station_id: str, units: str = 'm', show: bool = False, verbose: bool = False):
    """
    Simple cumulative precipitation plot (single station).
    
    Args:
        df: DataFrame with clean column names
        station_id: Station ID for title
        units: 'm' for metric, 'e' for imperial
        show: Whether to display the plot
        verbose: If True, print detailed statistics
    """
    if verbose:
        print("📈 CUMULATIVE PRECIPITATION ANALYSIS")

    if df is None or len(df) == 0:
        print("❌ No DataFrame available")
        return

    precip_cols = [col for col in df.columns if 'precip' in col.lower()]
    if len(precip_cols) == 0:
        print("❌ No precipitation columns")
        return

    precip_col = 'precip_rate' if 'precip_rate' in df.columns else 'precip_total'
    df_plot = df.copy()
    df_plot['cumulative_precip'] = df_plot[precip_col].cumsum()

    unit = 'mm' if units == 'm' else 'in'
    unit_h = 'mm/h' if units == 'm' else 'in/h'

    rain_events = df_plot[df_plot[precip_col] > 0]
    total_accum = df_plot['cumulative_precip'].iloc[-1]
    days = (df_plot['time_local'].max() - df_plot['time_local'].min()).days
    
    if verbose:
        print(f"💧 Total: {total_accum:.2f} {unit}, Events: {len(rain_events)}, Avg: {total_accum/days:.2f} {unit}/day")
    else:
        print(f"📈 {station_id}: {total_accum:.1f} {unit} total, {len(rain_events)} events")

    # CREATE SIMPLE PLOT - Just cumulative accumulation
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot['time_local'], df_plot['cumulative_precip'],
            linewidth=2, color='#0066CC', label=f'{station_id} ({total_accum:.1f} {unit})')
    ax.fill_between(df_plot['time_local'], 0, df_plot['cumulative_precip'],
                    alpha=0.3, color='#3399FF')
    ax.set_ylabel(f'Cumulative Precipitation ({unit})', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_title(f'💧 Cumulative Precipitation: {station_id}', fontsize=14, fontweight='bold')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.6, color='#666666', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    if show:
        plt.show()

    if verbose:
        daily_precip = df_plot.groupby(df_plot['time_local'].dt.date)[precip_col].sum()
        print(f"\n📊 Details: Period={df_plot['time_local'].min().date()} to {df_plot['time_local'].max().date()}")
        print(f"📈 Max rate: {df_plot[precip_col].max():.2f} {unit_h}")
        if len(daily_precip[daily_precip > 0]) > 0:
            print(f"📅 Wettest: {daily_precip.max():.2f} {unit} on {daily_precip.idxmax()}")


def plot_weather_dashboard_multi(station_data: Dict, units: str = 'm', show: bool = False, 
                                  max_stations: int = 4, verbose: bool = False):
    """
    Plot weather dashboard for multiple stations (sample of 3-4 stations).
    
    Args:
        station_data: Dictionary with station_id -> {'clean': df, ...}
        units: 'm' for metric, 'e' for imperial
        show: Whether to display the plot
        max_stations: Maximum number of stations to plot (default 4)
        verbose: If True, print detailed statistics
    """
    if not station_data or len(station_data) == 0:
        if verbose:
            print("❌ No station data available")
        return
    
    # Select sample stations (first max_stations)
    station_ids = list(station_data.keys())[:max_stations]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(station_ids)))
    
    for i, station_id in enumerate(station_ids):
        df = station_data[station_id]['clean']
        
        # Temperature
        if 'temp_avg' in df.columns:
            axes[0].plot(df['time_local'], df['temp_avg'], linewidth=1, 
                        label=station_id, color=colors[i], alpha=0.7)
        
        # Humidity
        if 'humidity_avg' in df.columns:
            axes[1].plot(df['time_local'], df['humidity_avg'], linewidth=1,
                        label=station_id, color=colors[i], alpha=0.7)
        
        # Wind
        if 'wind_speed_avg' in df.columns:
            axes[2].plot(df['time_local'], df['wind_speed_avg'], linewidth=1,
                        label=station_id, color=colors[i], alpha=0.7)
        
        # Precipitation
        if 'precip_rate' in df.columns:
            axes[3].bar(df['time_local'], df['precip_rate'], width=0.04, 
                       color=colors[i], alpha=0.5, label=station_id)
    
    # Set labels
    axes[0].set_ylabel('Temp (°C)' if units == 'm' else 'Temp (°F)', fontweight='bold')
    axes[0].set_title(f'Weather Dashboard: Sample Stations ({len(station_ids)} of {len(station_data)})', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8)
    
    axes[1].set_ylabel('Humidity (%)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8)
    
    axes[2].set_ylabel('Wind (km/h)' if units == 'm' else 'Wind (mph)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=8)
    
    axes[3].set_ylabel('Precip (mm/h)' if units == 'm' else 'Precip (in/h)', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='best', fontsize=8)
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    if show:
        plt.show()


def plot_weather_dashboard(df: pd.DataFrame, station_id: str, units: str = 'm', show: bool = False, verbose: bool = False):
    """
    Multi-variable weather dashboard.
    
    Args:
        df: DataFrame with clean column names
        station_id: Station ID for title
        units: 'm' for metric, 'e' for imperial
        show: Whether to display the plot
        verbose: If True, print detailed statistics
    """
    if df is None or len(df) == 0:
        if verbose:
            print("❌ No data for dashboard")
        return

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    # Temperature
    if 'temp_avg' in df.columns:
        axes[0].plot(df['time_local'], df['temp_avg'], linewidth=1.5, color='red')
        if 'temp_low' in df.columns and 'temp_high' in df.columns:
            axes[0].fill_between(df['time_local'], df['temp_low'], df['temp_high'], alpha=0.2, color='red')
        axes[0].set_ylabel('Temp (°C)' if units == 'm' else 'Temp (°F)', fontweight='bold')
        axes[0].set_title(f'Weather Dashboard: {station_id}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

    # Humidity
    if 'humidity_avg' in df.columns:
        axes[1].plot(df['time_local'], df['humidity_avg'], linewidth=1.5, color='blue')
        if 'humidity_low' in df.columns and 'humidity_high' in df.columns:
            axes[1].fill_between(df['time_local'], df['humidity_low'], df['humidity_high'], alpha=0.2, color='blue')
        axes[1].set_ylabel('Humidity (%)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    # Wind
    if 'wind_speed_avg' in df.columns:
        axes[2].plot(df['time_local'], df['wind_speed_avg'], linewidth=1.5, color='green')
        if 'wind_gust_high' in df.columns:
            axes[2].plot(df['time_local'], df['wind_gust_high'], linewidth=1, color='darkgreen', alpha=0.5, label='Gusts')
        axes[2].set_ylabel('Wind (km/h)' if units == 'm' else 'Wind (mph)', fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

    # Precipitation
    if 'precip_rate' in df.columns:
        axes[3].bar(df['time_local'], df['precip_rate'], width=0.04, color='dodgerblue', alpha=0.7)
        axes[3].set_ylabel('Precip (mm/h)' if units == 'm' else 'Precip (in/h)', fontweight='bold')
        axes[3].set_xlabel('Date', fontweight='bold')
        axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if show:
        plt.show()


def check_data_quality(df: pd.DataFrame, start_date, end_date, verbose: bool = False):
    """
    Perform data quality checks.
    
    Args:
        df: DataFrame with clean column names
        start_date: Requested start date
        end_date: Requested end date
        verbose: If True, print detailed statistics
    """
    if df is None or len(df) == 0:
        if verbose:
            print("❌ No data for quality checks")
        return

    first_date = df['time_local'].min()
    last_date = df['time_local'].max()
    expected_obs = (end_date - start_date).days * 24
    completeness = (len(df) / expected_obs) * 100
    time_diffs = df['time_local'].diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]

    if verbose:
        print("🔍 DATA QUALITY CHECKS")
        print(f"Date: {first_date.date()} to {last_date.date()}")
        print(f"Completeness: {completeness:.1f}% ({len(df)}/{expected_obs} obs)")
        print(f"Gaps > 2h: {len(gaps)}")
    else:
        print(f"✓ Quality: {completeness:.0f}% complete, {len(gaps)} gaps")

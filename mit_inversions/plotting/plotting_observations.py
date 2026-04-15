# plotting.py 
# Created: 15 April 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description:
#  Functions for plotting time series of observations data, including a standard 
#  format for publications and an Economist-style plot.

import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def align_time_series(data_list, time_name="time", target_time=None, method="interp", tolerance=None):
    """
    Align multiple xarray objects to a common time coordinate.

    Parameters
    ----------
    data_list : list of xr.DataArray or xr.Dataset
        Input time series objects.
    time_name : str
        Name of the time coordinate.
    target_time : array-like or None
        Target time coordinate. If None, uses the union of all times.
    method : str
        "interp" for interpolation, "nearest", "ffill", "bfill", or "exact".

    Returns
    -------
    list of xr objects aligned to the same time coordinate
    """

    # Determine target time axis
    if target_time is None:
        all_times = xr.concat(
            [d[time_name] for d in data_list], dim=time_name
        )
        target_time = all_times.to_index().unique().sort_values()

    aligned = []
    for d in data_list:
        if method == "exact":
            aligned.append(d.reindex({time_name: target_time}))
        elif method in ["nearest", "ffill", "bfill"]:
            aligned.append(d.reindex({time_name: target_time}, method=method, tolerance=tolerance))
        elif method == "interp":
            aligned.append(d.interp({time_name: target_time}))
        else:
            raise ValueError(f"Unknown method: {method}")

    return aligned


def plot_observations_timeseries_standard(myobs: dict, start_date: str, end_date: str, species: str, unit: str="ppt"):
    """
    Plot the observations in a standard format for publications. 
    Parameters:
    - myobs (dict): 
        Dictionary of xarray.Dataset objects containing observations, keyed by site.
    - start_date (str): 
        Start date for the plot (e.g., "2000-01-01").
    - end_date (str): 
        End date for the plot (e.g., "2020-12-31").
    - species (str):
        Name of the species being plotted (e.g., "CFC-11").
    - unit (str): 
        Unit for the y-axis (e.g., "ppt", "ppb", "ppm"). Default is "ppt".
    """

    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],

        # Figure background
        "figure.facecolor": "1.0", #"0.92",
        "axes.facecolor": "1.0",

        # Axis frame
        "axes.edgecolor": "0.3",
        "axes.linewidth": 1.0,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # Tick sizes
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,

        # Minor ticks
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,})
    
    if unit == "ppt":
        unit_sf = 1e12
    elif unit == "ppb":
        unit_sf = 1e9
    elif unit == "ppm":
        unit_sf = 1e6

    time_difference = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    for site in myobs.keys():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(myobs[site].time, myobs[site].mf * unit_sf, '.', lw=2, color='#A0B1bA')#, color='#009ADE')
        ax.set_title(f"{site} - {species}", fontsize=14)
        
        ax.set_xlim((pd.to_datetime(start_date), pd.to_datetime(end_date)))
        ax.set_xlabel("Time", fontsize=12)
        
        if time_difference <= 90:
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        else:
            ax.xaxis.set_minor_locator(mdates.MonthLocator())

        ax.set_ylabel(f"Atmospheric mole fraction ({unit})", fontsize=12)
        ax.set_title(f"{site} - {species}", fontsize=14)

        fig.tight_layout()
        plt.show()


def economist_timeseries_plot(
    x,
    y1,
    y2=None,
    y3=None,
    labels=None,
    colors=None,
    title="",
    subtitle="",
    ylabel="",
    source=None,
    note=None,
    highlight_last=True,
    direct_labels=True
):
    """
    Economist-style time series plot (up to 3 lines)

    Features:
    - Direct labeling (optional)
    - Clean editorial style
    """

    # Economist palette
    default_colors = ["#e3120b", "#006ba2", "#3e9651"]
    default_colors = ["#b22222", "#006ba2", "#3e9651"]
    if colors is None:
        colors = default_colors

    if labels is None:
        labels = ["Series 1", "Series 2", "Series 3"]

    # Style
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "xtick.top": False,
        "ytick.right": False,
        "ytick.left": False,
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build series list
    series = [(y1, colors[0], labels[0])]
    if y2 is not None:
        series.append((y2, colors[1], labels[1]))
    if y3 is not None:
        series.append((y3, colors[2], labels[2]))

    # Plot lines
    for y, c, lab in series:
        ax.plot(x, y, color=c, linewidth=2.8)

        if highlight_last:
            ax.scatter(x[-1], y[-1], color=c, s=35, zorder=3)

        # Direct labels at line end (Economist style)
        if direct_labels:
            ax.text(
                x[-1],
                y[-1],
                f" {lab}",
                color=c,
                va="center",
                fontsize=10
            )

    # Grid (very subtle)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.xaxis.grid(False)

    # Spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    

    # Titles
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    
    if subtitle:
        ax.text(0, 1.02, subtitle,
                transform=ax.transAxes,
                fontsize=11,
                color="#555555")

    # Axis labels
    ax.set_ylabel(ylabel)

    # Remove legend if using direct labels
    if not direct_labels and len(series) > 1:
        ax.legend(frameon=False)

    # Source and notes
    y_text = 0.01
    if source:
        fig.text(0.01, y_text, f"Source: {source}",
                 ha="left", fontsize=9, color="#555555")
        y_text += 0.03

    if note:
        fig.text(0.01, y_text, note,
                 ha="left", fontsize=9, color="#555555")

    plt.tight_layout()
    return fig, ax
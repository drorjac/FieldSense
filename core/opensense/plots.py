"""
Standard figures for OpenSense CML data, built on poligrain.

One function per figure the notebooks and scripts need, so a notebook cell
is a call and not forty lines of matplotlib. Each returns the figure and
never calls ``plt.show()``.

``network``          radar frame, link paths and gauges on one map
``metadata``         what the network can measure: length vs frequency
``retrieval_steps``  loss and baseline, attenuation and wet flag, rain rate
``hexbins``          estimate against references, poligrain hexbins
``series``           several (time,) series on one axis
``map_panels``       rainfall fields side by side, links on top
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from core import viz_style as vs

C1, C2, C3 = (vs.SERIES[k] for k in vs.SERIES_ORDER)
_GEOMETRY = ("cml_id", "site_0_lon", "site_0_lat", "site_1_lon", "site_1_lat")


def _links(cml) -> xr.Dataset:
    """Geometry only: plot_lines colours a DataArray by value, a Dataset not."""
    geo = xr.Dataset(coords={c: cml[c] for c in _GEOMETRY})
    return geo


def network(radar_frame: xr.DataArray, cml, gauges=None, vmax: float = 40.0,
            title: str = ""):
    """A radar frame with the link paths and gauge locations on top (lon/lat)."""
    import poligrain as plg

    fig, ax = plt.subplots(figsize=(7, 6))
    plg.plot_map.plot_plg(da_grid=radar_frame, use_lon_lat=True, ax=ax, vmin=0,
                          vmax=vmax, cmap=vs.CMAP_RAIN,
                          colorbar_label="radar rain rate (mm/h)")
    plg.plot_map.plot_lines(_links(cml), use_lon_lat=True, ax=ax,
                            line_color=vs.INK_PRIMARY, line_width=1.0)
    if gauges is not None:
        ax.scatter(gauges.lon, gauges.lat, s=40, color=C2, edgecolor=vs.SURFACE,
                   lw=1.5, zorder=5, label="gauge")
        ax.legend(loc="lower left")
    ax.set(xlabel="longitude", ylabel="latitude", title=title)
    return fig


def metadata(cml):
    """Length against frequency, and the frequency distribution."""
    import poligrain as plg

    freq = cml.frequency_ghz
    if "sublink_id" in freq.dims:
        freq = freq.isel(sublink_id=0)
    # poligrain's metadata plots take the OpenSense file units, metres and MHz,
    # and divide by 1000 themselves; given km and GHz every axis comes out
    # 1000x too small, without a warning.
    length_m, freq_mhz = cml.length_km * 1000.0, freq * 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    plg.plot_metadata.plot_len_vs_freq(length_m, freq_mhz, ax=axes[0], marker_color=C1)
    axes[0].set(xlabel="length (km)", ylabel="frequency (GHz)")
    plg.plot_metadata.plot_distribution(length_m, freq_mhz, variable="frequency",
                                        bins=15, ax=axes[1])
    axes[1].set_xlabel("frequency (GHz)")
    fig.tight_layout()
    return fig


def retrieval_steps(cml: xr.Dataset, retrieved: xr.Dataset, cml_id, sublink: int = 0):
    """The chain on one sublink: loss vs baseline, attenuation vs wet flag, rain.

    ``retrieved`` is the output of ``retrieval.retrieve_dataset`` on ``cml``.
    """
    one = retrieved.sel(cml_id=cml_id).isel(sublink_id=sublink)
    raw = cml.sel(cml_id=cml_id).isel(sublink_id=sublink)
    loss = raw.tsl - raw.rsl if "tsl" in raw else -raw.rsl

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(loss.time, loss, lw=1, color=vs.INK_SECONDARY, label="loss")
    ax[0].plot(one.time, one.baseline, lw=2, color=C1, label="dry baseline")
    ax[0].set_ylabel("loss (dB)")
    ax[1].fill_between(one.time, 0, 1, where=one.wet.values, color=vs.GRIDLINE,
                       transform=ax[1].get_xaxis_transform(), label="classified wet")
    ax[1].plot(one.time, one.A_obs, lw=1.2, color=C2, label="A_obs")
    ax[1].plot(one.time, one.waa, lw=1.2, color=vs.INK_PRIMARY, label="wet-antenna part")
    ax[1].set_ylabel("attenuation (dB)")
    ax[2].plot(one.time, one.R, lw=1, color=C1)
    ax[2].set_ylabel("rain rate (mm/h)")
    for a in ax[:2]:
        a.legend(loc="upper left")
    length = float(np.ravel(retrieved.sel(cml_id=cml_id).length_km)[0])
    freq = float(np.ravel(one.frequency_ghz)[0])
    ax[0].set_title(f"link {cml_id}: {length:.1f} km, {freq:.1f} GHz")
    fig.tight_layout()
    return fig


def hexbins(pairs: dict, threshold: float = 0.1):
    """One poligrain hexbin per ``{title: (reference, estimate)}``.

    Both sides are aligned by coordinate first, so dimension order does not
    matter.
    """
    import poligrain as plg

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2), squeeze=False)
    for ax, (title, (ref, est)) in zip(axes[0], pairs.items()):
        ref, est = xr.align(ref, est.transpose(*ref.dims))
        plg.validation.plot_hexbin(ref.values, est.values, ref_thresh=threshold,
                                   est_thresh=threshold, ax=ax, cmap="Blues", gridsize=30)
        ax.set(title=title, xlabel="reference (mm/h)", ylabel="CML (mm/h)")
    fig.tight_layout()
    return fig


def series(lines: dict, title: str = "", ylabel: str = "rain rate (mm/h)"):
    """Several (time,) DataArrays on one axis, in the palette's fixed order."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = [vs.INK_MUTED, C1, C2, C3]
    for (label, da), c in zip(lines.items(), colors):
        ax.plot(da.time, da, lw=1.6, color=c, label=label)
    ax.set(ylabel=ylabel, title=title)
    ax.legend(loc="upper right")
    return fig


def map_panels(fields: dict, lon: xr.DataArray, lat: xr.DataArray, links=None,
               vmax: float = 40.0, title: str = ""):
    """Rainfall fields on one lon/lat grid, side by side, one colour scale.

    ``links`` (a per-link DataArray with site lon/lat) is drawn on each
    panel, coloured by its own values on the same scale.
    """
    import poligrain as plg

    fig, axes = plt.subplots(1, len(fields), figsize=(4.7 * len(fields), 4.8),
                             sharey=True, squeeze=False)
    for ax, (name, f) in zip(axes[0], fields.items()):
        plg.plot_map.plot_plg(da_grid=f.assign_coords(lon=lon, lat=lat), da_cmls=links,
                              use_lon_lat=True, ax=ax, vmin=0, vmax=vmax,
                              cmap=vs.CMAP_RAIN, add_colorbar=ax is axes[0][-1],
                              colorbar_label="mm/h", kwargs_cmls_plot=dict(line_width=1.5))
        ax.set_title(name)
    if title:
        fig.suptitle(title, x=0.06, ha="left")
    fig.tight_layout()
    return fig

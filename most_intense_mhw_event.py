# ==============================================================
# MOST INTENSE REGIONAL MARINE HEATWAVE EVENT
# TROPICAL NORTH EAST ATLANTIC
#
# Region:
#   0-30°N, 60-10°W
#
# Period:
#   1982-2024
#
# Baseline:
#   1981-2010
#
# Event definition:
#   Regional SST > regional daily P90
#   for at least 5 consecutive days
#
# "Most intense event":
#   Event with the highest MAXIMUM intensity
#   relative to daily climatological mean
#
# Outputs:
#   (a) SST / climatology / P90 time series
#   (b) Mean SST anomaly map during event
# ==============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.filterwarnings("ignore")


# ==============================================================
# 1. PATHS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

THRESHOLD_FILE = (
    r"C:\Users\Aina Ajibola\Desktop\P90_1981-2010\threshold.nc"
)


# ==============================================================
# 2. SETTINGS
# ==============================================================

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"

ANALYSIS_START = "1982-01-01"
ANALYSIS_END = "2024-12-31"

MIN_DURATION = 5

HALF_WINDOW = 5
SMOOTH_WINDOW = 31

PADDING_DAYS = 15


# ==============================================================
# 3. HELPER FUNCTIONS
# ==============================================================

def preprocess_oisst(ds):

    rename_dict = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename_dict[old] = new

    if rename_dict:
        ds = ds.rename(rename_dict)

    for dim in ["zlev", "depth", "lev", "level"]:

        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)

    if "sst" not in ds.data_vars:
        raise KeyError("Variable 'sst' not found.")

    ds = ds[["sst"]]

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    ds = ds.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    return ds


def get_clim_day(dates):

    dates = pd.DatetimeIndex(dates)

    ref_dates = pd.to_datetime({
        "year": np.full(len(dates), 2000),
        "month": dates.month,
        "day": dates.day
    })

    return (
        pd.DatetimeIndex(ref_dates)
        .dayofyear
        .to_numpy(dtype=np.int16)
    )


def circular_smooth(values, window=31):

    values = np.asarray(values, dtype=float)

    half = window // 2

    extended = np.concatenate([
        values[-half:],
        values,
        values[:half]
    ])

    smoothed = np.full_like(values, np.nan)

    for i in range(len(values)):

        smoothed[i] = np.nanmean(
            extended[i:i + window]
        )

    return smoothed


def find_events(condition, min_duration=5):

    condition = np.asarray(condition, dtype=bool)

    events = []

    start = None

    for i, value in enumerate(condition):

        if value and start is None:
            start = i

        if start is not None:

            run_finished = (
                (not value)
                or
                (i == len(condition) - 1)
            )

            if run_finished:

                if value and i == len(condition) - 1:
                    end = i
                else:
                    end = i - 1

                duration = end - start + 1

                if duration >= min_duration:
                    events.append(
                        (start, end, duration)
                    )

                start = None

    return events


# ==============================================================
# 4. FIND FILES
# ==============================================================

files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*_oisst.nc")
    )
)

if not files:

    files = sorted(
        glob.glob(
            os.path.join(DATA_DIR, "*.nc")
        )
    )

if not files:
    raise FileNotFoundError(
        f"No NetCDF files found in:\n{DATA_DIR}"
    )

print(f"Files found: {len(files):,}")


# ==============================================================
# 5. OPEN OISST
# ==============================================================

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    preprocess=preprocess_oisst,
    parallel=False,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="outer",
    engine="netcdf4"
)

sst = ds["sst"].sortby("time")


# ==============================================================
# 6. NORMALIZE TIME
# ==============================================================

dates = pd.DatetimeIndex(
    sst.time.values
).normalize()

sst = sst.assign_coords(
    time=dates
)

keep = np.where(
    ~dates.duplicated(keep="first")
)[0]

sst = sst.isel(
    time=keep
)

sst = sst.sortby("time")

dates = pd.DatetimeIndex(
    sst.time.values
)


# ==============================================================
# 7. CHECK SST UNITS
# ==============================================================

sample = float(
    sst.isel(
        time=slice(0, 10)
    )
    .mean(skipna=True)
    .compute()
)

if sample > 100:

    print("Converting SST from Kelvin to °C.")

    sst = sst - 273.15

else:

    print("SST already appears to be °C.")


# ==============================================================
# 8. LATITUDE WEIGHTS
# ==============================================================

lat_weights = xr.DataArray(
    np.cos(
        np.deg2rad(
            sst.lat.values
        )
    ),
    coords={"lat": sst.lat},
    dims=["lat"]
)


# ==============================================================
# 9. REGIONAL DAILY SST
# ==============================================================

regional_sst = (
    sst
    .weighted(lat_weights)
    .mean(
        dim=["lat", "lon"],
        skipna=True
    )
    .compute()
)


# ==============================================================
# 10. REGIONAL DAILY CLIMATOLOGY
# ==============================================================

baseline_regional = regional_sst.sel(
    time=slice(
        BASE_START,
        BASE_END
    )
)

baseline_dates = pd.DatetimeIndex(
    baseline_regional.time.values
)

baseline_values = np.asarray(
    baseline_regional.values,
    dtype=float
)

baseline_days = get_clim_day(
    baseline_dates
)

regional_climatology = np.full(
    366,
    np.nan
)

for day in range(1, 367):

    distance = np.abs(
        baseline_days - day
    )

    distance = np.minimum(
        distance,
        366 - distance
    )

    mask = (
        distance <= HALF_WINDOW
    )

    regional_climatology[
        day - 1
    ] = np.nanmean(
        baseline_values[mask]
    )

regional_climatology = circular_smooth(
    regional_climatology,
    window=SMOOTH_WINDOW
)


# ==============================================================
# 11. OPEN P90 THRESHOLD
# ==============================================================

thr_ds = xr.open_dataset(
    THRESHOLD_FILE
)

rename_thr = {}

for old, new in {
    "latitude": "lat",
    "longitude": "lon",
    "Latitude": "lat",
    "Longitude": "lon"
}.items():

    if old in thr_ds.coords or old in thr_ds.dims:
        rename_thr[old] = new

if rename_thr:
    thr_ds = thr_ds.rename(rename_thr)


# ==============================================================
# 12. FIND THRESHOLD VARIABLE
# ==============================================================

candidate_vars = []

for var in thr_ds.data_vars:

    lower = var.lower()

    if any(
        key in lower
        for key in [
            "threshold",
            "thresh",
            "p90",
            "percentile"
        ]
    ):
        candidate_vars.append(var)

if candidate_vars:

    threshold_var = candidate_vars[0]

elif len(thr_ds.data_vars) == 1:

    threshold_var = list(
        thr_ds.data_vars
    )[0]

else:

    raise ValueError(
        "Could not identify threshold variable."
    )

print(
    f"P90 variable: {threshold_var}"
)

threshold = thr_ds[
    threshold_var
].squeeze(drop=True)


# ==============================================================
# 13. STANDARDIZE P90 LONGITUDE
# ==============================================================

if float(threshold.lon.max()) > 180:

    threshold = threshold.assign_coords(
        lon=((threshold.lon + 180.0) % 360.0) - 180.0
    )

threshold = threshold.sortby("lat")
threshold = threshold.sortby("lon")

threshold = threshold.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)


# ==============================================================
# 14. FIND DAY DIMENSION
# ==============================================================

day_dim = None

for dim in [
    "clim_day",
    "dayofyear",
    "day_of_year",
    "doy",
    "day",
    "time"
]:

    if (
        dim in threshold.dims
        and
        threshold.sizes[dim] in [365, 366]
    ):

        day_dim = dim
        break

if day_dim is None:

    for dim in threshold.dims:

        if threshold.sizes[dim] in [365, 366]:

            day_dim = dim
            break

if day_dim is None:

    raise ValueError(
        "Could not identify P90 daily dimension."
    )

if threshold.sizes[day_dim] != 366:

    raise ValueError(
        "This code expects a 366-day P90 threshold."
    )


# ==============================================================
# 15. ALIGN P90 TO SST GRID
# ==============================================================

threshold = threshold.interp(
    lat=sst.lat,
    lon=sst.lon,
    method="nearest"
)

threshold = threshold.transpose(
    day_dim,
    "lat",
    "lon"
)


# ==============================================================
# 16. BUILD STANDARD 366-DAY P90 ARRAY
# ==============================================================

threshold_raw = np.asarray(
    threshold.values,
    dtype=np.float32
)

day_coord = np.asarray(
    threshold[day_dim].values
)

threshold_366 = np.full(
    (
        366,
        sst.sizes["lat"],
        sst.sizes["lon"]
    ),
    np.nan,
    dtype=np.float32
)


if np.issubdtype(
    day_coord.dtype,
    np.datetime64
):

    thr_dates = pd.DatetimeIndex(
        day_coord
    )

    thr_days = get_clim_day(
        thr_dates
    )

    for i, d in enumerate(thr_days):

        threshold_366[
            d - 1
        ] = threshold_raw[i]

elif (
    np.issubdtype(
        day_coord.dtype,
        np.number
    )
    and
    np.nanmin(day_coord) >= 1
    and
    np.nanmax(day_coord) <= 366
):

    for i, d in enumerate(
        day_coord.astype(int)
    ):

        threshold_366[
            d - 1
        ] = threshold_raw[i]

else:

    threshold_366[:] = threshold_raw


# ==============================================================
# 17. REGIONAL AREA-WEIGHTED P90
# ==============================================================

weights_2d = np.cos(
    np.deg2rad(
        sst.lat.values
    )
)[:, None]

weights_2d = np.broadcast_to(
    weights_2d,
    (
        sst.sizes["lat"],
        sst.sizes["lon"]
    )
)

regional_p90 = np.full(
    366,
    np.nan
)

for d in range(366):

    field = threshold_366[d]

    valid = np.isfinite(field)

    regional_p90[d] = (
        np.nansum(
            field * weights_2d
        )
        /
        np.nansum(
            np.where(
                valid,
                weights_2d,
                np.nan
            )
        )
    )


# ==============================================================
# 18. ANALYSIS PERIOD
# ==============================================================

analysis = regional_sst.sel(
    time=slice(
        ANALYSIS_START,
        ANALYSIS_END
    )
)

analysis_dates = pd.DatetimeIndex(
    analysis.time.values
)

analysis_sst = np.asarray(
    analysis.values,
    dtype=float
)

analysis_days = get_clim_day(
    analysis_dates
)

analysis_clim = regional_climatology[
    analysis_days - 1
]

analysis_p90 = regional_p90[
    analysis_days - 1
]


# ==============================================================
# 19. FIND ALL REGIONAL MHW EVENTS
# ==============================================================

above = (
    analysis_sst
    >
    analysis_p90
)

events = find_events(
    above,
    min_duration=MIN_DURATION
)

if not events:

    raise RuntimeError(
        "No MHW events found."
    )

print(
    f"\nTotal confirmed regional MHW events: "
    f"{len(events)}"
)


# ==============================================================
# 20. CALCULATE METRICS FOR EVERY EVENT
# ==============================================================

event_results = []

for start_idx, end_idx, duration in events:

    event_sst_series = analysis_sst[
        start_idx:end_idx + 1
    ]

    event_clim_series = analysis_clim[
        start_idx:end_idx + 1
    ]

    intensity = (
        event_sst_series
        -
        event_clim_series
    )

    mean_intensity = np.nanmean(
        intensity
    )

    max_intensity = np.nanmax(
        intensity
    )

    cumulative_intensity = np.nansum(
        intensity
    )

    event_results.append({

        "start_idx": start_idx,

        "end_idx": end_idx,

        "duration": duration,

        "start_date":
            analysis_dates[start_idx],

        "end_date":
            analysis_dates[end_idx],

        "mean_intensity":
            mean_intensity,

        "max_intensity":
            max_intensity,

        "cumulative_intensity":
            cumulative_intensity
    })


# ==============================================================
# 21. FIND MOST INTENSE EVENT
#
# Highest MAXIMUM intensity
# ==============================================================

most_intense = max(
    event_results,
    key=lambda x: x["max_intensity"]
)

start_idx = most_intense[
    "start_idx"
]

end_idx = most_intense[
    "end_idx"
]

duration = most_intense[
    "duration"
]

event_start = most_intense[
    "start_date"
]

event_end = most_intense[
    "end_date"
]

mean_intensity = most_intense[
    "mean_intensity"
]

max_intensity = most_intense[
    "max_intensity"
]

cumulative_intensity = most_intense[
    "cumulative_intensity"
]


print("\n" + "=" * 80)

print("MOST INTENSE REGIONAL MHW EVENT")

print("=" * 80)

print(
    f"Start date           : "
    f"{event_start.strftime('%d %B %Y')}"
)

print(
    f"End date             : "
    f"{event_end.strftime('%d %B %Y')}"
)

print(
    f"Duration             : "
    f"{duration} days"
)

print(
    f"Maximum intensity    : "
    f"{max_intensity:.2f} °C"
)

print(
    f"Mean intensity       : "
    f"{mean_intensity:.2f} °C"
)

print(
    f"Cumulative intensity : "
    f"{cumulative_intensity:.2f} °C days"
)


# ==============================================================
# 22. TIME WINDOW FOR PANEL A
# ==============================================================

plot_start_idx = max(
    0,
    start_idx - PADDING_DAYS
)

plot_end_idx = min(
    len(analysis_dates) - 1,
    end_idx + PADDING_DAYS
)

ts_dates = analysis_dates[
    plot_start_idx:
    plot_end_idx + 1
]

ts_sst = analysis_sst[
    plot_start_idx:
    plot_end_idx + 1
]

ts_clim = analysis_clim[
    plot_start_idx:
    plot_end_idx + 1
]

ts_p90 = analysis_p90[
    plot_start_idx:
    plot_end_idx + 1
]


# ==============================================================
# 23. GRIDDED CLIMATOLOGY FOR EVENT DAYS
# ==============================================================

print(
    "\nCalculating event SST anomaly map..."
)

baseline_grid = sst.sel(
    time=slice(
        BASE_START,
        BASE_END
    )
)

baseline_grid_dates = pd.DatetimeIndex(
    baseline_grid.time.values
)

baseline_grid_days = get_clim_day(
    baseline_grid_dates
)

event_dates = analysis_dates[
    start_idx:
    end_idx + 1
]

event_days = get_clim_day(
    event_dates
)

unique_event_days = np.unique(
    event_days
)

clim_maps = {}


for day in unique_event_days:

    distance = np.abs(
        baseline_grid_days - day
    )

    distance = np.minimum(
        distance,
        366 - distance
    )

    indices = np.where(
        distance <= HALF_WINDOW
    )[0]

    clim_map = (
        baseline_grid
        .isel(time=indices)
        .mean(
            dim="time",
            skipna=True
        )
        .compute()
    )

    clim_maps[int(day)] = clim_map


# ==============================================================
# 24. DAILY SST ANOMALY MAPS
# ==============================================================

event_grid = sst.sel(
    time=slice(
        event_start,
        event_end
    )
)

daily_anomaly_maps = []

for i, date in enumerate(event_dates):

    clim_day = int(
        event_days[i]
    )

    daily_sst_map = event_grid.sel(
        time=date
    )

    anomaly = (
        daily_sst_map
        -
        clim_maps[clim_day]
    )

    daily_anomaly_maps.append(
        anomaly
    )


event_anomaly = xr.concat(
    daily_anomaly_maps,
    dim="event_time"
).mean(
    dim="event_time",
    skipna=True
).compute()


# ==============================================================
# 25. FIGURE
# ==============================================================

fig = plt.figure(
    figsize=(14, 12)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1, 1.25],
    hspace=0.30
)


# ==============================================================
# PANEL A
# ==============================================================

ax1 = fig.add_subplot(
    gs[0, 0]
)


ax1.plot(
    ts_dates,
    ts_clim,
    linewidth=2,
    label="Climatology"
)

ax1.plot(
    ts_dates,
    ts_p90,
    linewidth=2,
    label="P90 threshold"
)

ax1.plot(
    ts_dates,
    ts_sst,
    linewidth=2,
    label="SST"
)


event_mask = (
    (ts_dates >= event_start)
    &
    (ts_dates <= event_end)
)


ax1.fill_between(
    ts_dates,
    ts_p90,
    ts_sst,
    where=(
        event_mask
        &
        (ts_sst > ts_p90)
    ),
    alpha=0.45,
    label="MHW event"
)


ax1.axvline(
    event_start,
    linestyle="--",
    linewidth=1
)

ax1.axvline(
    event_end,
    linestyle="--",
    linewidth=1
)


ax1.set_ylabel(
    "SST (°C)",
    fontsize=11,
    fontweight="bold"
)


ax1.set_title(
    f"(a) Most Intense Regional MHW Event: "
    f"{duration} Days\n"
    f"{event_start.strftime('%d %b %Y')} – "
    f"{event_end.strftime('%d %b %Y')}\n"
    f"Maximum Intensity = {max_intensity:.2f} °C, "
    f"Mean Intensity = {mean_intensity:.2f} °C, "
    f"Cumulative Intensity = {cumulative_intensity:.1f} °C days",
    fontsize=12,
    fontweight="bold"
)


ax1.legend(
    loc="best"
)


ax1.grid(
    alpha=0.25
)


ax1.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%d/%m/%Y"
    )
)

plt.setp(
    ax1.get_xticklabels(),
    rotation=30,
    ha="right"
)


# ==============================================================
# PANEL B
# ==============================================================

ax2 = fig.add_subplot(
    gs[1, 0],
    projection=ccrs.PlateCarree()
)


# --------------------------------------------------------------
# Use symmetric anomaly scale
# --------------------------------------------------------------

absmax = float(
    np.nanmax(
        np.abs(
            event_anomaly.values
        )
    )
)

limit = np.ceil(
    absmax * 10
) / 10

levels = np.linspace(
    -limit,
    limit,
    17
)


cf = ax2.contourf(
    event_anomaly.lon,
    event_anomaly.lat,
    event_anomaly.values,
    levels=levels,
    cmap="RdBu_r",
    extend="both",
    transform=ccrs.PlateCarree()
)


# --------------------------------------------------------------
# Contours
# --------------------------------------------------------------

contour_step = max(
    0.2,
    round(
        limit / 5,
        1
    )
)

positive_levels = np.arange(
    contour_step,
    limit + contour_step,
    contour_step
)

if len(positive_levels) > 0:

    cs = ax2.contour(
        event_anomaly.lon,
        event_anomaly.lat,
        event_anomaly.values,
        levels=positive_levels,
        linewidths=0.7,
        transform=ccrs.PlateCarree()
    )

    ax2.clabel(
        cs,
        inline=True,
        fontsize=7,
        fmt="%.1f"
    )


# ==============================================================
# 26. MAP FEATURES
# ==============================================================

ax2.add_feature(
    cfeature.LAND,
    facecolor="0.75",
    zorder=3
)

ax2.add_feature(
    cfeature.COASTLINE,
    linewidth=0.8,
    zorder=4
)

ax2.add_feature(
    cfeature.BORDERS,
    linewidth=0.45,
    zorder=4
)


# ==============================================================
# 27. EXACT DOMAIN
# ==============================================================

ax2.set_extent(
    [
        -60,
        -10,
        0,
        30
    ],
    crs=ccrs.PlateCarree()
)


# ==============================================================
# 28. SHOW ALL IMPORTANT LATITUDE/LONGITUDE LABELS
# ==============================================================

lon_ticks = [
    -60,
    -50,
    -40,
    -30,
    -20,
    -10
]

lat_ticks = [
    0,
    5,
    10,
    15,
    20,
    25,
    30
]


ax2.set_xticks(
    lon_ticks,
    crs=ccrs.PlateCarree()
)

ax2.set_yticks(
    lat_ticks,
    crs=ccrs.PlateCarree()
)


ax2.xaxis.set_major_formatter(
    LongitudeFormatter(
        degree_symbol="°"
    )
)

ax2.yaxis.set_major_formatter(
    LatitudeFormatter(
        degree_symbol="°"
    )
)


ax2.tick_params(
    axis="both",
    labelsize=10
)


# --------------------------------------------------------------
# Grid lines
# --------------------------------------------------------------

ax2.gridlines(
    xlocs=lon_ticks,
    ylocs=lat_ticks,
    linewidth=0.4,
    linestyle="--",
    alpha=0.35,
    draw_labels=False
)


ax2.set_title(
    f"(b) Mean SST Anomaly During Most Intense MHW\n"
    f"{event_start.strftime('%d %b %Y')} – "
    f"{event_end.strftime('%d %b %Y')}",
    fontsize=12,
    fontweight="bold"
)


# ==============================================================
# 29. COLORBAR
# ==============================================================

cbar = fig.colorbar(
    cf,
    ax=ax2,
    orientation="vertical",
    pad=0.025,
    shrink=0.92
)

cbar.set_label(
    "SST Anomaly (°C)",
    fontsize=11,
    fontweight="bold"
)

cbar.ax.yaxis.set_major_formatter(
    mticker.FormatStrFormatter("%.1f")
)


# ==============================================================
# 30. OVERALL TITLE
# ==============================================================

fig.suptitle(
    "Most Intense Marine Heatwave Event in the Tropical North East Atlantic",
    fontsize=16,
    fontweight="bold",
    y=0.98
)


plt.show()


# ==============================================================
# 31. FINAL EVENT SUMMARY
# ==============================================================

print("\n" + "=" * 80)

print("MOST INTENSE MHW SUMMARY")

print("=" * 80)

print(
    f"Event dates          : "
    f"{event_start.strftime('%d %B %Y')} "
    f"to "
    f"{event_end.strftime('%d %B %Y')}"
)

print(
    f"Duration             : "
    f"{duration} days"
)

print(
    f"Maximum intensity    : "
    f"{max_intensity:.2f} °C"
)

print(
    f"Mean intensity       : "
    f"{mean_intensity:.2f} °C"
)

print(
    f"Cumulative intensity : "
    f"{cumulative_intensity:.2f} °C days"
)

print(
    f"Study region         : "
    f"0-30°N, 60-10°W"
)

print("=" * 80)


thr_ds.close()
ds.close()
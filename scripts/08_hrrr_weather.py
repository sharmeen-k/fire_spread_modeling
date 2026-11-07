"""
08_hrrr_weather.py
Hourly HRRR surface weather for one fire, plus the real VIIRS observation
time of every perimeter record.

Why the observation times matter
    The Largefire `time` field is a LABEL, not a UTC timestamp:
    00:00 = the ~01:30 local-solar-time VIIRS overpass, 12:00 = the ~13:30
    overpass; the date is the UTC date (FEDS convention, Chen et al. 2022).
    Real UTC time = date + (1.5 or 13.5 h) - clon/15 h  (about +9.5 h in SoCal).

What it does
    1. Converts record labels -> obs_time_utc  (obs_times.csv)
    2. Downloads HRRR analyses (fxx=0, sfc) hour by hour from AWS/Google via Herbie
       for [first obs - 24 h, last obs + 1 h], fields:
       UGRD/VGRD 10 m, TMP 2 m, RH 2 m, GUST surface
    3. Crops each hour to the script-07 grid box (+0.1 deg) at native 3 km
       and caches it (re-running resumes; nothing is re-downloaded)
    4. Rotates winds from grid-relative to earth-relative (HRRR GRIB winds are
       grid-relative on the Lambert grid; ~13 deg off in Ventura County if ignored),
       adds speed and meteorological direction (degrees the wind blows FROM)
    5. Writes one NetCDF (time, y, x) with lat/lon and UTM 11N x/y of every cell,
       a summary, and a time-series plot at the cell nearest the ignition.

Requires script 07 to have been run (uses data/processed/fires/<id>/grid.json).

Usage (repo root, .venv active):
    python scripts/08_hrrr_weather.py --fire-id 4299
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import xarray as xr
from pyproj import Transformer
from rasterio.warp import transform_bounds

REPO = Path(__file__).resolve().parents[1]
PERIM_GPKG = REPO / "data" / "processed" / "socal_perimeters.gpkg"
OUT_DIR = REPO / "outputs"

SEARCH = r":(?:UGRD|VGRD):10 m above|:(?:TMP|RH):2 m above|:GUST:surface"
SHORT_MAP = {"10u": "u10", "10v": "v10", "2t": "t2m", "2r": "rh2", "gust": "gust"}
NAME_MAP = {"u10": "u10", "v10": "v10", "t2m": "t2m", "r2": "rh2", "gust": "gust"}
NEEDED = ["u10", "v10", "t2m", "rh2", "gust"]
MARGIN_DEG = 0.1
LEAD_H = 24
DEFAULT_LOV, DEFAULT_LATIN1 = 262.5, 38.5   # HRRR Lambert conformal parameters


# ------------------------------------------------------------ obs times
def load_records(fire_id):
    names = [row[0] for row in pyogrio.list_layers(PERIM_GPKG)]
    layer = "perimeter" if "perimeter" in names else names[0]
    gdf = pyogrio.read_dataframe(PERIM_GPKG, layer=layer, read_geometry=False)
    gdf = gdf[gdf["fireID"] == fire_id].copy()
    if gdf.empty:
        sys.exit(f"fireID {fire_id} not found")
    lab = pd.to_datetime(gdf["time"])
    if lab.dt.tz is not None:
        lab = lab.dt.tz_localize(None)
    gdf["label_time"] = lab
    return gdf.sort_values("label_time").reset_index(drop=True)


def obs_times(df):
    hrs = df["label_time"].dt.hour
    if not hrs.isin([0, 12]).all():
        sys.exit(f"Unexpected label hours {sorted(hrs.unique())}; expected only 0 and 12")
    solar = np.where(hrs == 0, 1.5, 13.5)
    offset = -df["clon"].to_numpy(dtype=float) / 15.0
    obs = df["label_time"].dt.normalize() + pd.to_timedelta(solar + offset, unit="h")
    return obs.dt.tz_localize("UTC")


# ----------------------------------------------------------------- HRRR
def lon180(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def crop_window(lat, lon, box):
    w, s, e, n = box
    lon = lon180(lon)
    m = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    ys, xs = np.where(m)
    if ys.size == 0:
        sys.exit("Crop box does not overlap the HRRR grid")
    return slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)


def collect_vars(ds):
    """Herbie returns a Dataset or a list of Datasets (one per level type)."""
    dss = ds if isinstance(ds, list) else [ds]
    out, lat, lon, attrs = {}, None, None, {}
    for d in dss:
        if lat is None:
            lat, lon = d["latitude"].values, d["longitude"].values
        for name, da in d.data_vars.items():
            key = SHORT_MAP.get(da.attrs.get("GRIB_shortName", "")) or NAME_MAP.get(name)
            if key:
                out[key] = da.values.astype("float32")
                if key == "u10":
                    attrs["lov"] = float(da.attrs.get("GRIB_LoVInDegrees", DEFAULT_LOV))
                    attrs["latin1"] = float(da.attrs.get("GRIB_Latin1InDegrees", DEFAULT_LATIN1))
    return out, lat, lon, attrs


def fetch_hour(t, cache_dir, tmp_dir, box, win, first_report):
    """Returns (path or None, status, window)."""
    from herbie import Herbie
    f = cache_dir / f"{t:%Y%m%d%H}.nc"
    if f.exists():
        return f, "cached", win
    try:
        H = Herbie(t.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc", fxx=0,
                   priority=["aws", "google"], save_dir=tmp_dir, verbose=False)
        if H.grib is None:
            return None, "missing", win
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = H.xarray(SEARCH, remove_grib=True)
    except Exception as exc:  # network hiccup, missing file, decode error
        return None, f"error: {type(exc).__name__}: {str(exc)[:120]}", win
    vals, lat, lon, attrs = collect_vars(ds)
    lacking = [k for k in NEEDED if k not in vals]
    if lacking:
        return None, f"error: fields missing {lacking}", win
    if win is None:
        win = crop_window(lat, lon, box)
    ys, xs = win
    hour = xr.Dataset(
        {k: (("y", "x"), vals[k][ys, xs]) for k in NEEDED},
        coords={"lat": (("y", "x"), lat[ys, xs].astype("float64")),
                "lon": (("y", "x"), lon180(lon[ys, xs]).astype("float64"))},
        attrs=attrs)
    if first_report:
        print("  first hour decoded: fields", sorted(vals), "| HRRR attrs", attrs,
              "| crop", hour["u10"].shape)
    hour.to_netcdf(f)
    return f, "downloaded", win


# ------------------------------------------------------------ derived
def rotate_to_earth(u, v, lon, lov, latin1):
    """Grid-relative -> earth-relative for Lambert conformal (tangent cone).
    Same formula as NCL wrf_uvmet / wgrib2 -new_grid_winds earth."""
    n = np.sin(np.radians(latin1))
    alpha = np.radians(n * (lon180(lon) - lon180(lov)))
    ue = np.cos(alpha) * u + np.sin(alpha) * v
    ve = -np.sin(alpha) * u + np.cos(alpha) * v
    return ue, ve, np.degrees(alpha)


def wind_dir_from(ue, ve):
    return (270.0 - np.degrees(np.arctan2(ve, ue))) % 360.0


# ------------------------------------------------------------------ plot
def plot_series(path, ts, obs, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    t = ts.index
    ax[0].plot(t, ts["wspd10"], label="10 m wind")
    ax[0].plot(t, ts["gust"], label="gust", alpha=0.7)
    ax[0].set_ylabel("m/s"); ax[0].legend(loc="upper right")
    ax[1].scatter(t, ts["wdir10"], s=4)
    ax[1].axhspan(0, 110, color="orange", alpha=0.15, label="N-E offshore sector")
    ax[1].set_ylim(0, 360); ax[1].set_yticks([0, 90, 180, 270, 360])
    ax[1].set_ylabel("dir from (deg)"); ax[1].legend(loc="upper right")
    ax[2].plot(t, ts["rh2"], color="tab:green"); ax[2].set_ylabel("RH 2 m (%)")
    ax[3].plot(t, ts["t2m_c"], color="tab:red"); ax[3].set_ylabel("T 2 m (C)")
    for a in ax:
        for o in obs:
            a.axvline(o, color="grey", lw=0.4, alpha=0.5)
    ax[3].set_xlabel("UTC (grey lines = VIIRS observation times)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire-id", type=int, default=4299)
    ap.add_argument("--max-hours", type=int, default=None,
                    help="only fetch the first N hours (quick test)")
    args = ap.parse_args()
    fid = args.fire_id
    fdir = REPO / "data" / "processed" / "fires" / str(fid)
    grid_json = fdir / "grid.json"
    if not grid_json.exists():
        sys.exit(f"{grid_json} not found: run script 07 for this fire first")
    grid = json.loads(grid_json.read_text())
    cache_dir = REPO / "data" / "raw" / "hrrr" / str(fid)
    tmp_dir = REPO / "data" / "raw" / "hrrr" / "_tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # --- observation times
    df = load_records(fid)
    df["obs_time_utc"] = obs_times(df)
    cols = ["label_time", "obs_time_utc", "farea", "n_newpixels", "clat", "clon"]
    df[cols].to_csv(fdir / "obs_times.csv", index_label="record")
    gaps = df["obs_time_utc"].diff().dt.total_seconds().dropna() / 3600
    log(f"fireID {fid}: {len(df)} records")
    log(f"  label {df['label_time'].iloc[0]}  -> obs {df['obs_time_utc'].iloc[0]}")
    log(f"  label {df['label_time'].iloc[-1]} -> obs {df['obs_time_utc'].iloc[-1]}")
    log(f"  gap between consecutive obs (h): min {gaps.min():.2f}, max {gaps.max():.2f}")

    # --- window + crop box
    t0 = (df["obs_time_utc"].iloc[0] - pd.Timedelta(hours=LEAD_H)).floor("h")
    t1 = (df["obs_time_utc"].iloc[-1] + pd.Timedelta(hours=1)).ceil("h")
    hours = pd.date_range(t0, t1, freq="h")
    if args.max_hours:
        hours = hours[: args.max_hours]
    w, s, e, n = transform_bounds(grid["crs"], "EPSG:4326", *grid["bounds"], densify_pts=21)
    box = (w - MARGIN_DEG, s - MARGIN_DEG, e + MARGIN_DEG, n + MARGIN_DEG)
    log(f"HRRR window {hours[0]} -> {hours[-1]} ({len(hours)} hours), "
        f"box W {box[0]:.3f} S {box[1]:.3f} E {box[2]:.3f} N {box[3]:.3f}")

    # --- download / cache
    status, files, win = {}, {}, None
    first_report = True
    for k, t in enumerate(hours, 1):
        f, st, win = fetch_hour(t, cache_dir, tmp_dir, box, win, first_report)
        if st == "downloaded":
            first_report = False
        status[t] = st
        if f is not None:
            files[t] = f
        if k % 24 == 0 or k == len(hours) or not st.startswith(("cached", "downloaded")):
            print(f"  {k}/{len(hours)}  {t:%Y-%m-%d %H}Z  {st}")
    bad = {t: s_ for t, s_ in status.items() if t not in files}
    log(f"Hours ok: {len(files)}/{len(hours)}  "
        f"(downloaded {sum(v == 'downloaded' for v in status.values())}, "
        f"cached {sum(v == 'cached' for v in status.values())})")
    if bad:
        log(f"Hours missing/failed ({len(bad)}):")
        for t, s_ in list(bad.items())[:30]:
            log(f"  {t:%Y-%m-%d %H}Z  {s_}")
        log("  Re-run the script to retry failed hours (cached hours are skipped).")
    if not files:
        sys.exit("No HRRR hours available")

    # --- assemble
    times = sorted(files)
    parts = [xr.open_dataset(files[t]).load() for t in times]
    for p in parts:
        p.close()
    ds = xr.concat(parts, dim=pd.DatetimeIndex([t.tz_convert(None) for t in times], name="time"),
                   coords="minimal", compat="override")
    ds = ds.assign_coords(lat=parts[0]["lat"], lon=parts[0]["lon"])
    lov = parts[0].attrs.get("lov", DEFAULT_LOV)
    latin1 = parts[0].attrs.get("latin1", DEFAULT_LATIN1)
    ue, ve, alpha = rotate_to_earth(ds["u10"].values, ds["v10"].values,
                                    ds["lon"].values, lov, latin1)
    ds["u10_earth"] = (("time", "y", "x"), ue.astype("float32"))
    ds["v10_earth"] = (("time", "y", "x"), ve.astype("float32"))
    ds["wspd10"] = (("time", "y", "x"), np.hypot(ue, ve).astype("float32"))
    ds["wdir10"] = (("time", "y", "x"), wind_dir_from(ue, ve).astype("float32"))
    ds["t2m_c"] = ds["t2m"] - 273.15
    ds = ds.rename({"u10": "u10_grid", "v10": "v10_grid"})
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    xu, yu = tr.transform(ds["lon"].values, ds["lat"].values)
    ds = ds.assign_coords(x_utm=(("y", "x"), xu), y_utm=(("y", "x"), yu))
    units = {"u10_grid": "m/s grid-relative", "v10_grid": "m/s grid-relative",
             "u10_earth": "m/s eastward", "v10_earth": "m/s northward",
             "wspd10": "m/s", "wdir10": "deg, direction wind blows FROM",
             "t2m": "K", "t2m_c": "degC", "rh2": "%", "gust": "m/s"}
    for k, u in units.items():
        ds[k].attrs["units"] = u
    ds.attrs.update(source="HRRR sfc analysis (fxx=0) via Herbie, AWS/Google archive",
                    fireID=fid, lov=lov, latin1=latin1,
                    note="time is UTC (naive); winds rotated to earth-relative")
    out_nc = fdir / "hrrr_hourly.nc"
    ds.to_netcdf(out_nc)

    log(f"\nRotation: LoV {lov}, Latin1 {latin1} -> angle {alpha.min():.1f} to "
        f"{alpha.max():.1f} deg over the box (would be the direction error if ignored)")
    log(f"Crop: {ds.sizes['y']} x {ds.sizes['x']} HRRR cells (~3 km)")
    for k in ["wspd10", "gust", "wdir10", "rh2", "t2m_c"]:
        v = ds[k].values
        log(f"  {k:<7} min {np.nanmin(v):7.1f}  max {np.nanmax(v):7.1f}  "
            f"mean {np.nanmean(v):7.1f}  NaN {np.isnan(v).mean():.1%}")

    # --- ignition-cell time series
    lat0, lon0 = float(df["clat"].iloc[0]), float(df["clon"].iloc[0])
    d2 = (ds["lat"].values - lat0) ** 2 + ((ds["lon"].values - lon0) * np.cos(np.radians(lat0))) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    cell = ds.isel(y=iy, x=ix)
    ts = pd.DataFrame({k: cell[k].values for k in ["wspd10", "gust", "wdir10", "rh2", "t2m_c"]},
                      index=pd.DatetimeIndex(cell["time"].values))
    offshore = ts["wdir10"].between(0, 110) & (ts["wspd10"] >= 5) & (ts["rh2"] <= 25)
    log(f"\nIgnition-area cell ({float(cell['lat']):.3f}, {float(cell['lon']):.3f}):")
    log(f"  hours with N-E wind >= 5 m/s and RH <= 25%: {offshore.sum()} of {len(ts)} "
        "(rough Santa Ana preview, not the final label)")
    first_day = ts.loc[ts.index <= ts.index[0] + pd.Timedelta(hours=48)]
    log(f"  first 48 h: max wind {first_day['wspd10'].max():.1f} m/s, max gust "
        f"{first_day['gust'].max():.1f} m/s, min RH {first_day['rh2'].min():.0f}%")
    png = OUT_DIR / f"fire_{fid}_hrrr_timeseries.png"
    plot_series(png, ts, df["obs_time_utc"].dt.tz_convert(None),
                f"fireID {fid}: HRRR at cell nearest ignition ({lat0:.3f}, {lon0:.3f})")
    log(f"\nWrote {out_nc.relative_to(REPO)}, {(fdir / 'obs_times.csv').relative_to(REPO)}")
    log(f"Wrote {png.relative_to(REPO)}")
    (OUT_DIR / f"fire_{fid}_hrrr_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""
09_hrrr_station_check.py

Check hourly HRRR weather (from script 08) against surface weather stations
for one fire.

Stations
  - ASOS (airports) from the Iowa Environmental Mesonet (IEM). No key needed.
  - RAWS (fire-weather stations) from the Synoptic Weather API. Needs a token in
    the environment variable SYNOPTIC_TOKEN. If it is not set, RAWS is skipped.

What it does
  1. Reads data/processed/fires/<id>/hrrr_hourly.nc and prints its variables.
  2. Finds stations inside the HRRR box and downloads their records for the
     HRRR period (cached under data/raw/stations/<id>/, so re-runs are offline).
  3. Converts units (ASOS knots -> m/s, F -> C) and matches each HRRR hour to
     the station report closest to the top of that hour (within +/-30 min).
  4. Takes HRRR at the nearest 3 km cell to each station.
  5. Computes bias (HRRR - station), MAE, RMSE and correlation for wind speed,
     gust, wind direction, RH and temperature, for all hours and split into
     offshore / other hours (regional HRRR flag, independent of the stations).
  6. Uses the 30 m DEM from script 07 to measure where each station sits
     relative to its surroundings (ridge vs valley), and relates that to the
     wind bias.

Outputs
  outputs/fire_<id>_stations.csv               station list + terrain context
  outputs/fire_<id>_hrrr_station_pairs.csv     matched hourly pairs
  outputs/fire_<id>_hrrr_station_metrics.csv   metrics per station / network
  outputs/fire_<id>_hrrr_station_summary.txt   readable summary
  outputs/fire_<id>_station_map.png
  outputs/fire_<id>_hrrr_station_scatter.png
  outputs/fire_<id>_hrrr_station_timeseries.png

Usage (PowerShell, from C:\\fire_spread_modelling with .venv active)
  python scripts\\09_hrrr_station_check.py --fire-id 4299
  python scripts\\09_hrrr_station_check.py --fire-id 4299 --networks asos
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "fires"
RAW = ROOT / "data" / "raw" / "stations"
OUT = ROOT / "outputs"

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network/{net}.geojson"
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
RAWS_NETWORK_ID = "2"          # RAWS in Synoptic/MesoWest

KT_TO_MS = 0.514444
MATCH_TOL = pd.Timedelta(minutes=30)
MIN_WSPD_FOR_DIR = 2.0         # m/s; directions below this are not compared
MAX_CELL_DIST_KM = 3.0         # station must be this close to an HRRR cell centre
BOX_SHRINK_DEG = 0.03          # keep stations about one HRRR cell inside the crop
TERRAIN_WINDOW_M = 3000        # ~one HRRR cell, for the ridge/valley measure

# Rough regional offshore flag (same cell rule as script 08's indicator)
OFFSHORE_DIR = (0.0, 110.0)
OFFSHORE_WSPD = 5.0
OFFSHORE_RH = 25.0
OFFSHORE_CELL_FRAC = 0.20      # hour is "offshore" if >= 20% of cells qualify

VARS = ["wspd", "gust", "wdir", "rh", "t2m"]
UNITS = {"wspd": "m/s", "gust": "m/s", "wdir": "deg", "rh": "%", "t2m": "C"}


# ---------------------------------------------------------------- helpers
class ClientError(RuntimeError):
    """4xx response: retrying won't help (bad token, bad parameters)."""


def get_with_retry(url, params, tries=4, timeout=120):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise ClientError(last)
        except requests.RequestException as e:
            last = str(e)
        wait = 5 * (k + 1)
        print(f"    request failed ({last}); retrying in {wait} s")
        time.sleep(wait)
    raise RuntimeError(f"Request failed after {tries} tries: {last}")


def first_name(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def circ_diff(a, b):
    """a - b in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- HRRR
class Hrrr:
    """Thin wrapper that finds variable and coordinate names in hrrr_hourly.nc."""

    def __init__(self, path):
        self.ds = xr.open_dataset(path)
        ds = self.ds
        names = set(ds.variables)

        lat = first_name(names, ["latitude", "lat", "LAT"])
        lon = first_name(names, ["longitude", "lon", "LON"])
        if lat is None or lon is None:
            raise SystemExit(f"Can't find latitude/longitude in {path}. "
                             f"Variables: {sorted(names)}")
        self.lat = np.asarray(ds[lat].values, dtype=float)
        lonv = np.asarray(ds[lon].values, dtype=float)
        self.lon = np.where(lonv > 180, lonv - 360, lonv)
        if self.lat.ndim != 2:
            raise SystemExit(f"Expected 2-D latitude, got shape {self.lat.shape}")
        self.ydim, self.xdim = ds[lat].dims

        self.v = {
            "u": first_name(names, ["u10", "u10_earth", "u10e", "UGRD"]),
            "v": first_name(names, ["v10", "v10_earth", "v10e", "VGRD"]),
            "wspd": first_name(names, ["wspd10", "wspd", "ws10"]),
            "wdir": first_name(names, ["wdir10", "wdir", "wd10"]),
            "gust": first_name(names, ["gust", "GUST", "i10fg"]),
            "rh": first_name(names, ["r2", "rh2", "rh", "RH", "rh2m"]),
            "t2m_c": first_name(names, ["t2m_c"]),
            "t2m_k": first_name(names, ["t2m", "TMP"]),
        }
        ref = self.v["wspd"] or self.v["u"]
        if ref is None:
            raise SystemExit(f"No wind variable found. Variables: {sorted(names)}")

        # time axis = the dimension of the wind variable that isn't spatial
        tdims = [d for d in ds[ref].dims if d not in (self.ydim, self.xdim)]
        if len(tdims) != 1:
            raise SystemExit(f"Can't identify the time dimension of {ref}: {ds[ref].dims}")
        self.tdim = tdims[0]
        tcoord = None
        for c in ["valid_time", self.tdim, "time"]:
            if c in ds.variables and ds[c].dims == (self.tdim,) \
                    and np.issubdtype(ds[c].dtype, np.datetime64):
                tcoord = c
                break
        if tcoord is None:
            raise SystemExit(f"No datetime coordinate along '{self.tdim}'.")
        t = pd.DatetimeIndex(ds[tcoord].values)
        t = t.tz_localize(None) if t.tz is not None else t
        self.times = t.astype("datetime64[ns]")

    def describe(self):
        lines = ["HRRR file contents:"]
        for k, v in self.ds.data_vars.items():
            u = v.attrs.get("units", "")
            lines.append(f"  {k:12s} {str(v.dims):35s} {u}")
        lines.append(f"  time axis '{self.tdim}': {len(self.times)} hours, "
                     f"{self.times.min()} to {self.times.max()} UTC")
        lines.append(f"  grid {self.lat.shape}, lat {self.lat.min():.3f}..{self.lat.max():.3f}, "
                     f"lon {self.lon.min():.3f}..{self.lon.max():.3f}")
        lines.append("  using: " + ", ".join(f"{k}={v}" for k, v in self.v.items() if v))
        return "\n".join(lines)

    def bbox(self, shrink=BOX_SHRINK_DEG):
        return (self.lon.min() + shrink, self.lat.min() + shrink,
                self.lon.max() - shrink, self.lat.max() - shrink)

    def nearest_cell(self, lat, lon):
        d = haversine_km(lat, lon, self.lat, self.lon)
        iy, ix = np.unravel_index(np.argmin(d), d.shape)
        return int(iy), int(ix), float(d[iy, ix])

    def _field(self, key):
        n = self.v[key]
        return None if n is None else self.ds[n]

    def series_at(self, iy, ix):
        sel = {self.ydim: iy, self.xdim: ix}
        out = pd.DataFrame(index=self.times)

        def get(key):
            f = self._field(key)
            return None if f is None else np.asarray(f.isel(sel).values, dtype=float)

        u, v = get("u"), get("v")
        ws = get("wspd")
        out["wspd"] = ws if ws is not None else np.hypot(u, v)
        wd = get("wdir")
        if wd is None and u is not None:
            wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        out["wdir"] = wd
        out["gust"] = get("gust")
        out["rh"] = get("rh")
        tc = get("t2m_c")
        if tc is None:
            tk = get("t2m_k")
            tc = None if tk is None else tk - 273.15
        out["t2m"] = tc
        return out

    def offshore_hours(self):
        """Regional flag from HRRR only: share of cells meeting the offshore rule."""
        ws = self._field("wspd")
        wd = self._field("wdir")
        rh = self._field("rh")
        if ws is None or wd is None or rh is None:
            u, v = self._field("u"), self._field("v")
            if u is None or rh is None:
                return pd.Series(False, index=self.times)
            ws = np.hypot(u, v)
            wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        ok = ((wd >= OFFSHORE_DIR[0]) & (wd <= OFFSHORE_DIR[1])
              & (ws >= OFFSHORE_WSPD) & (rh <= OFFSHORE_RH))
        frac = ok.mean(dim=[self.ydim, self.xdim]).values
        return pd.Series(frac >= OFFSHORE_CELL_FRAC, index=self.times)


# ---------------------------------------------------------------- ASOS (IEM)
def asos_stations(bbox, cache_dir, refresh):
    f = cache_dir / "asos_CA_ASOS.geojson"
    if refresh or not f.exists():
        print("  downloading CA_ASOS station list from IEM")
        r = get_with_retry(IEM_NETWORK_URL.format(net="CA_ASOS"), params={})
        f.write_text(r.text, encoding="utf-8")
    gj = json.loads(f.read_text(encoding="utf-8"))
    lonmin, latmin, lonmax, latmax = bbox
    rows = []
    for feat in gj.get("features", []):
        p = feat.get("properties", {}) or {}
        lon, lat = feat["geometry"]["coordinates"][:2]
        if lonmin <= lon <= lonmax and latmin <= lat <= latmax:
            rows.append({
                "network": "ASOS",
                "stid": p.get("sid") or feat.get("id"),
                "name": p.get("sname") or p.get("name") or "",
                "lat": float(lat), "lon": float(lon),
                "elev_m": float(p["elevation"]) if p.get("elevation") not in (None, "") else np.nan,
            })
    return pd.DataFrame(rows)


def asos_data(stid, t0, t1, cache_dir, refresh):
    f = cache_dir / f"asos_{stid}_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.csv"
    if refresh or not f.exists():
        print(f"  downloading ASOS {stid}")
        params = [
            ("station", stid),
            ("data", "sknt"), ("data", "drct"), ("data", "gust"),
            ("data", "tmpf"), ("data", "relh"),
            ("sts", t0.strftime("%Y-%m-%dT%H:%MZ")),
            ("ets", t1.strftime("%Y-%m-%dT%H:%MZ")),
            ("tz", "Etc/UTC"), ("format", "onlycomma"),
            ("latlon", "no"), ("missing", "empty"), ("trace", "empty"),
            ("report_type", "3"), ("report_type", "4"),   # routine + specials
        ]
        r = get_with_retry(IEM_ASOS_URL, params=params)
        f.write_text(r.text, encoding="utf-8")
        time.sleep(1)   # IEM rate limit
    df = pd.read_csv(io.StringIO(f.read_text(encoding="utf-8")),
                     na_values=["", "M"])
    if df.empty or "valid" not in df.columns:
        return pd.DataFrame(columns=["time"] + VARS)
    out = pd.DataFrame({"time": pd.to_datetime(df["valid"])})
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    out["wspd"] = num("sknt") * KT_TO_MS
    out["gust"] = num("gust") * KT_TO_MS
    out["wdir"] = num("drct")
    out["rh"] = num("relh")
    out["t2m"] = (num("tmpf") - 32.0) * 5.0 / 9.0
    return out


# ---------------------------------------------------------------- RAWS (Synoptic)
def raws_fetch(bbox, t0, t1, cache_dir, refresh, token):
    f = cache_dir / f"raws_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.json"
    if refresh or not f.exists():
        if not token:
            return None
        print("  downloading RAWS from Synoptic (token taken from SYNOPTIC_TOKEN)")
        lonmin, latmin, lonmax, latmax = bbox
        params = {
            "token": token,
            "bbox": f"{lonmin:.4f},{latmin:.4f},{lonmax:.4f},{latmax:.4f}",
            "start": t0.strftime("%Y%m%d%H%M"),
            "end": t1.strftime("%Y%m%d%H%M"),
            "vars": "wind_speed,wind_direction,wind_gust,air_temp,relative_humidity",
            "network": RAWS_NETWORK_ID,
            "units": "metric",               # m/s and deg C
            "obtimezone": "UTC",
        }
        try:
            r = get_with_retry(SYNOPTIC_URL, params=params)
        except (ClientError, RuntimeError) as e:
            print(f"  Synoptic request failed: {e}")
            if "token" in str(e).lower():
                print("  -> SYNOPTIC_TOKEN must be a TOKEN generated from your API key,")
                print("     not the key itself. See the instructions in the chat.")
            print("  RAWS skipped; continuing with the other networks.")
            return None
        js = r.json()
        code = js.get("SUMMARY", {}).get("RESPONSE_CODE")
        if code != 1:
            msg = js.get("SUMMARY", {}).get("RESPONSE_MESSAGE")
            print(f"  Synoptic returned code {code}: {msg}")
            print("  RAWS skipped. (If this mentions history/access limits, see the notes.)")
            return None
        f.write_text(json.dumps(js), encoding="utf-8")   # token is not in the body
    return json.loads(f.read_text(encoding="utf-8"))


def raws_parse(js, bbox):
    stations, data = [], {}
    lonmin, latmin, lonmax, latmax = bbox
    keymap = {"wind_speed": "wspd", "wind_direction": "wdir", "wind_gust": "gust",
              "air_temp": "t2m", "relative_humidity": "rh"}
    for st in js.get("STATION", []):
        lat, lon = float(st["LATITUDE"]), float(st["LONGITUDE"])
        if not (lonmin <= lon <= lonmax and latmin <= lat <= latmax):
            continue
        # Synoptic gives ELEVATION in feet (station metadata is not unit-converted)
        try:
            elev_m = float(st.get("ELEVATION")) * 0.3048
        except (TypeError, ValueError):
            elev_m = np.nan
        obs = st.get("OBSERVATIONS", {})
        if "date_time" not in obs:
            continue
        df = pd.DataFrame({"time": pd.to_datetime(obs["date_time"], utc=True).tz_localize(None)})
        for syn, ours in keymap.items():
            keys = sorted(k for k in obs if k.startswith(syn + "_set_"))
            df[ours] = pd.to_numeric(pd.Series(obs[keys[0]]), errors="coerce") if keys else np.nan
        stations.append({"network": "RAWS", "stid": st["STID"], "name": st.get("NAME", ""),
                         "lat": lat, "lon": lon, "elev_m": elev_m})
        data[st["STID"]] = df
    return pd.DataFrame(stations), data


# ---------------------------------------------------------------- matching
def match_hourly(obs, hours):
    """For each HRRR hour, take the station value closest in time (<= 30 min), per variable."""
    base = pd.DataFrame({"time": pd.DatetimeIndex(hours).astype("datetime64[ns]")}).sort_values("time")
    out = base.copy()
    for v in VARS:
        if v not in obs or obs[v].isna().all():
            out[v] = np.nan
            continue
        o = obs[["time", v]].dropna().sort_values("time")
        o["time"] = o["time"].astype("datetime64[ns]")
        m = pd.merge_asof(base, o, on="time", direction="nearest", tolerance=MATCH_TOL)
        out[v] = m[v].values
    return out.set_index("time")


def metrics(o, h, var):
    ok = np.isfinite(o) & np.isfinite(h)
    o, h = np.asarray(o)[ok], np.asarray(h)[ok]
    n = len(o)
    if n < 3:
        return dict(n=n, obs_mean=np.nan, hrrr_mean=np.nan, bias=np.nan,
                    mae=np.nan, rmse=np.nan, r=np.nan)
    if var == "wdir":
        d = circ_diff(h, o)
        mean_d = np.degrees(np.arctan2(np.sin(np.radians(d)).mean(),
                                       np.cos(np.radians(d)).mean()))
        return dict(n=n, obs_mean=np.nan, hrrr_mean=np.nan, bias=mean_d,
                    mae=np.abs(d).mean(), rmse=np.sqrt((d ** 2).mean()), r=np.nan)
    d = h - o
    r = np.corrcoef(o, h)[0, 1] if o.std() > 0 and h.std() > 0 else np.nan
    return dict(n=n, obs_mean=o.mean(), hrrr_mean=h.mean(), bias=d.mean(),
                mae=np.abs(d).mean(), rmse=np.sqrt((d ** 2).mean()), r=r)


# ---------------------------------------------------------------- terrain
DEP3_URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
            "3DEPElevation/ImageServer/exportImage")


def dem_window_3dep(stid, x, y, crs_epsg, res, cache_dir, refresh):
    """3 km x 3 km DEM window around a station from USGS 3DEP (cached GeoTIFF)."""
    import rasterio
    half = TERRAIN_WINDOW_M / 2
    n = int(round(TERRAIN_WINDOW_M / res))
    f = cache_dir / f"dem3dep_{stid}.tif"
    if refresh or not f.exists():
        print(f"  downloading 3DEP terrain window for {stid}")
        params = {
            "bbox": f"{x - half},{y - half},{x + half},{y + half}",
            "bboxSR": crs_epsg, "imageSR": crs_epsg, "size": f"{n},{n}",
            "format": "tiff", "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation", "f": "image",
        }
        r = get_with_retry(DEP3_URL, params=params)
        if not r.content[:4] in (b"II*\x00", b"MM\x00*"):
            raise RuntimeError(f"3DEP did not return a GeoTIFF: {r.text[:200]}")
        f.write_bytes(r.content)
    with rasterio.open(f) as src:
        a = src.read(1, masked=True).astype(float).filled(np.nan)
    a[a < -1000] = np.nan
    return a


def terrain_context(stations, fire_dir, cache_dir=None, refresh=False):
    """Station elevation on the 30 m DEM, and how far it sits above/below
    the mean of a ~3 km window (about one HRRR cell). Positive = ridge-like."""
    dem_path = fire_dir / "dem.tif"
    stations = stations.copy()
    stations["dem_elev_m"] = np.nan
    stations["rel_elev_3km_m"] = np.nan
    stations["in_model_grid"] = False
    stations["terrain_src"] = ""
    if not dem_path.exists():
        print(f"  {dem_path} not found; skipping terrain context")
        return stations, None
    import rasterio
    from pyproj import Transformer
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype(float).filled(np.nan)
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        half = int(round(TERRAIN_WINDOW_M / 2 / abs(src.res[0])))
        for i, s in stations.iterrows():
            x, y = tf.transform(s["lon"], s["lat"])
            row, col = src.index(x, y)
            inside = (0 <= row - half and row + half < src.height
                      and 0 <= col - half and col + half < src.width)
            if 0 <= row < src.height and 0 <= col < src.width:
                stations.at[i, "in_model_grid"] = True
            if inside:
                win = dem[row - half:row + half + 1, col - half:col + half + 1]
                stations.at[i, "terrain_src"] = "grid"
            elif cache_dir is not None and s.get("n_hours", 0) > 0:
                # station (or its 3 km window) is outside the fire grid: fetch from 3DEP
                try:
                    win = dem_window_3dep(s["stid"], x, y, src.crs.to_epsg(),
                                          abs(src.res[0]), cache_dir, refresh)
                    stations.at[i, "terrain_src"] = "3dep"
                except Exception as e:
                    print(f"  {s['stid']}: 3DEP terrain window failed ({e})")
                    continue
            else:
                continue
            c = win[win.shape[0] // 2, win.shape[1] // 2]
            stations.at[i, "dem_elev_m"] = c
            stations.at[i, "rel_elev_3km_m"] = c - np.nanmean(win)
        info = dict(dem=dem, transform=src.transform, crs=src.crs, bounds=src.bounds)
    return stations, info


# ---------------------------------------------------------------- figures
def plot_map(stations, hr, info, fire_dir, path, fire_id):
    if info is None:
        return
    from pyproj import Transformer
    import rasterio
    b = info["bounds"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(info["dem"], extent=(b.left, b.right, b.bottom, b.top), cmap="terrain",
              vmin=-200, vmax=np.nanmax(info["dem"]))
    perim = fire_dir / "perim_last.tif"
    if perim.exists():
        with rasterio.open(perim) as src:
            p = src.read(1)
            px = b.left + (np.arange(p.shape[1]) + 0.5) * (b.right - b.left) / p.shape[1]
            py = b.top - (np.arange(p.shape[0]) + 0.5) * (b.top - b.bottom) / p.shape[0]
        ax.contour(px, py, p, levels=[0.5], colors="red", linewidths=1.2)
    tf = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True)
    hx, hy = tf.transform(hr.lon, hr.lat)
    ax.plot(hx.ravel(), hy.ravel(), ".", color="k", ms=1.5, alpha=0.4,
            label="HRRR cell centres")
    for net, mk, col in [("ASOS", "s", "white"), ("RAWS", "^", "yellow")]:
        s = stations[stations.network == net]
        if s.empty:
            continue
        x, y = tf.transform(s.lon.values, s.lat.values)
        ax.scatter(x, y, marker=mk, s=70, c=col, edgecolors="k", zorder=5, label=net)
        for xi, yi, t in zip(x, y, s.stid):
            ax.annotate(t, (xi, yi), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, weight="bold")
    ax.set_xlim(min(b.left, np.nanmin(hx)), max(b.right, np.nanmax(hx)))
    ax.set_ylim(min(b.bottom, np.nanmin(hy)), max(b.top, np.nanmax(hy)))
    ax.set_title(f"Fire {fire_id}: weather stations used to check HRRR "
                 f"(red = final perimeter)")
    ax.set_xlabel("UTM 11N easting (m)")
    ax.set_ylabel("northing (m)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(pairs, path, fire_id):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, v in zip(axes.ravel(), ["wspd", "gust", "rh", "t2m"]):
        for net, col in [("ASOS", "tab:blue"), ("RAWS", "tab:orange")]:
            d = pairs[pairs.network == net]
            ax.scatter(d[f"obs_{v}"], d[f"hrrr_{v}"], s=4, alpha=0.35, c=col, label=net)
        lo = np.nanmin([pairs[f"obs_{v}"].min(), pairs[f"hrrr_{v}"].min()])
        hi = np.nanmax([pairs[f"obs_{v}"].max(), pairs[f"hrrr_{v}"].max()])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"station {v} ({UNITS[v]})")
        ax.set_ylabel(f"HRRR {v} ({UNITS[v]})")
        ax.legend(fontsize=8)
    fig.suptitle(f"Fire {fire_id}: HRRR nearest cell vs station, hourly")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeseries(pairs, stations, obs_times, path, fire_id):
    counts = pairs.dropna(subset=["obs_wspd"]).groupby("stid").size()
    picks = []
    for net in ["RAWS", "ASOS"]:      # up to 2 per network, most data first
        ids = stations[stations.network == net].stid
        picks += list(counts[counts.index.isin(ids)].sort_values(ascending=False).index[:2])
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.4 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, picks):
        d = pairs[pairs.stid == sid]
        net = d.network.iloc[0]
        ax.plot(d.time, d.obs_wspd, lw=0.9, label=f"{net} {sid} observed")
        ax.plot(d.time, d.hrrr_wspd, lw=0.9, label="HRRR nearest cell")
        for t in obs_times:
            ax.axvline(t, color="grey", lw=0.3, alpha=0.5)
        ax.set_ylabel("wind (m/s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Fire {fire_id}: 10 m wind speed, HRRR vs stations "
                      f"(grey = VIIRS observation times, UTC)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fire-id", required=True, type=int)
    ap.add_argument("--networks", default="asos,raws",
                    help="comma list: asos, raws (default both)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached station downloads")
    args = ap.parse_args()
    nets = {n.strip().lower() for n in args.networks.split(",") if n.strip()}

    fire_dir = PROC / str(args.fire_id)
    nc = fire_dir / "hrrr_hourly.nc"
    if not nc.exists():
        sys.exit(f"{nc} not found. Run script 08 for this fire first.")
    cache = RAW / str(args.fire_id)
    cache.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)

    hr = Hrrr(nc)
    print(hr.describe())
    bbox = hr.bbox()
    t0 = hr.times.min().floor("h") - pd.Timedelta(hours=1)
    t1 = hr.times.max().ceil("h") + pd.Timedelta(hours=1)
    print(f"\nStation search box (lon/lat): {tuple(round(float(b), 3) for b in bbox)}")
    print(f"Period: {t0} to {t1} UTC")

    obs_times = []
    ot = fire_dir / "obs_times.csv"
    if ot.exists():
        obs_times = pd.to_datetime(pd.read_csv(ot)["obs_time_utc"], utc=True) \
            .dt.tz_localize(None).tolist()

    # -------- stations
    st_frames, obs_by_id = [], {}
    if "asos" in nets:
        print("\nASOS (IEM)")
        a = asos_stations(bbox, cache, args.refresh)
        if len(a):
            far = [hr.nearest_cell(la, lo)[2] > MAX_CELL_DIST_KM for la, lo in zip(a.lat, a.lon)]
            if any(far):
                print(f"  outside the HRRR crop, skipped: {', '.join(a.stid[far])}")
            a = a[~np.array(far)].reset_index(drop=True)
        print(f"  {len(a)} ASOS stations in box: {', '.join(a.stid) if len(a) else '-'}")
        for sid in a.stid:
            try:
                obs_by_id[sid] = asos_data(sid, t0, t1, cache, args.refresh)
            except Exception as e:
                print(f"  {sid}: download failed ({e}); skipped")
        st_frames.append(a)
    if "raws" in nets:
        print("\nRAWS (Synoptic)")
        token = os.environ.get("SYNOPTIC_TOKEN", "").strip()
        js = raws_fetch(bbox, t0, t1, cache, args.refresh, token)
        if js is None and not token:
            print("  SYNOPTIC_TOKEN is not set and no cached RAWS file; RAWS skipped")
        if js is not None:
            r, rd = raws_parse(js, bbox)
            print(f"  {len(r)} RAWS stations in box: {', '.join(r.stid) if len(r) else '-'}")
            st_frames.append(r)
            obs_by_id.update(rd)
    stations = pd.concat(st_frames, ignore_index=True) if st_frames else pd.DataFrame()
    if stations.empty:
        sys.exit("No stations found. Check the box above, or run with --networks asos.")

    # -------- nearest HRRR cell + matching
    offshore = hr.offshore_hours()
    for c in ["hrrr_iy", "hrrr_ix", "n_hours"]:
        stations[c] = 0
    stations["hrrr_dist_km"] = np.nan
    rows = []
    for i, s in stations.iterrows():
        iy, ix, dist = hr.nearest_cell(s.lat, s.lon)
        stations.at[i, "hrrr_iy"], stations.at[i, "hrrr_ix"] = iy, ix
        stations.at[i, "hrrr_dist_km"] = round(dist, 2)
        obs = obs_by_id.get(s.stid)
        if dist > MAX_CELL_DIST_KM or obs is None or obs.empty:
            stations.at[i, "n_hours"] = 0
            continue
        m = match_hourly(obs, hr.times)
        h = hr.series_at(iy, ix)
        df = pd.DataFrame({"time": hr.times, "network": s.network, "stid": s.stid,
                           "offshore": offshore.values})
        for v in VARS:
            df[f"obs_{v}"] = m[v].values
            df[f"hrrr_{v}"] = h[v].values
        # directions only where both winds are strong enough to have one
        weak = (df.obs_wspd < MIN_WSPD_FOR_DIR) | (df.hrrr_wspd < MIN_WSPD_FOR_DIR)
        df.loc[weak, ["obs_wdir", "hrrr_wdir"]] = np.nan
        stations.at[i, "n_hours"] = int(df.obs_wspd.notna().sum())
        rows.append(df)
    if not rows:
        sys.exit("Stations were found but none had usable data near an HRRR cell.")
    pairs = pd.concat(rows, ignore_index=True)
    stations["n_hours"] = stations["n_hours"].astype(int)

    stations, info = terrain_context(stations, fire_dir, cache, args.refresh)

    # -------- metrics
    mrows = []
    groups = [(s.network, s.stid, pairs[pairs.stid == s.stid]) for s in stations.itertuples()
              if s.n_hours > 0]
    for net in pairs.network.unique():
        groups.append((net, "ALL", pairs[pairs.network == net]))
    for net, sid, d in groups:
        for regime, sel in [("all", slice(None)), ("offshore", d.offshore),
                            ("other", ~d.offshore)]:
            dd = d if regime == "all" else d[sel]
            for v in VARS:
                m = metrics(dd[f"obs_{v}"].values, dd[f"hrrr_{v}"].values, v)
                mrows.append(dict(network=net, stid=sid, regime=regime, variable=v,
                                  units=UNITS[v], **m))
    met = pd.DataFrame(mrows)

    # -------- write tables
    f = args.fire_id
    stations.to_csv(OUT / f"fire_{f}_stations.csv", index=False, float_format="%.4f")
    pairs.to_csv(OUT / f"fire_{f}_hrrr_station_pairs.csv", index=False, float_format="%.3f")
    met.to_csv(OUT / f"fire_{f}_hrrr_station_metrics.csv", index=False, float_format="%.3f")

    # -------- summary
    L = [f"HRRR vs surface stations, fire {f}", "=" * 50, hr.describe(), "",
         f"Period {t0} to {t1} UTC; station box {tuple(round(float(b), 3) for b in bbox)}",
         f"Offshore hours (>= {OFFSHORE_CELL_FRAC:.0%} of HRRR cells with wind from "
         f"{OFFSHORE_DIR[0]:.0f}-{OFFSHORE_DIR[1]:.0f} deg, >= {OFFSHORE_WSPD} m/s, "
         f"RH <= {OFFSHORE_RH}%): {int(offshore.sum())} of {len(offshore)}",
         "Bias = HRRR - station. Wind direction compared only when both speeds "
         f">= {MIN_WSPD_FOR_DIR} m/s.", "",
         "Stations", "-" * 50]
    cols = ["network", "stid", "name", "lat", "lon", "elev_m", "dem_elev_m",
            "rel_elev_3km_m", "terrain_src", "in_model_grid", "hrrr_dist_km", "n_hours"]
    L.append(stations[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    L += ["", "Network totals (all hours)", "-" * 50]
    t = met[(met.stid == "ALL") & (met.regime == "all")]
    L.append(t[["network", "variable", "units", "n", "obs_mean", "hrrr_mean",
                "bias", "mae", "rmse", "r"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    L += ["", "Network totals by regime: wind speed and RH", "-" * 50]
    t = met[(met.stid == "ALL") & met.variable.isin(["wspd", "gust", "rh"])]
    L.append(t[["network", "regime", "variable", "n", "bias", "rmse"]]
             .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    L += ["", "Wind speed by station, with terrain position", "-" * 50]
    w = met[(met.stid != "ALL") & (met.regime == "all") & (met.variable == "wspd")] \
        .merge(stations[["stid", "rel_elev_3km_m", "dem_elev_m"]], on="stid")
    w = w.sort_values("rel_elev_3km_m", ascending=False)
    L.append(w[["network", "stid", "dem_elev_m", "rel_elev_3km_m", "n", "obs_mean",
                "hrrr_mean", "bias", "rmse"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    ok = w.dropna(subset=["rel_elev_3km_m", "bias"])
    if len(ok) >= 4:
        rr = np.corrcoef(ok.rel_elev_3km_m, ok.bias)[0, 1]
        L.append(f"\nCorrelation of station wind bias with relative elevation "
                 f"(n={len(ok)} stations): r = {rr:.2f}")
        L.append("Negative r means HRRR under-predicts wind more at ridge-like stations.")
    L += ["", "Notes",
          "- ASOS: 10 m, 2-min mean wind; gusts reported only when gusty, so gust stats",
          "  cover gusty hours only. RAWS: ~6.1 m (20 ft), 10-min mean wind, hourly peak gust.",
          "  RAWS winds are therefore expected to read lower than a true 10 m wind.",
          "- HRRR value is the nearest 3 km cell (no interpolation).",
          "- rel_elev_3km_m = station DEM height minus the mean DEM height in a 3 km",
          "  window (~one HRRR cell); > 0 ridge-like, < 0 valley-like.",
          "  terrain_src: grid = fire-grid DEM (script 07); 3dep = window fetched from USGS 3DEP."]
    txt = "\n".join(L)
    (OUT / f"fire_{f}_hrrr_station_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)

    # -------- figures
    plot_map(stations, hr, info, fire_dir, OUT / f"fire_{f}_station_map.png", f)
    plot_scatter(pairs, OUT / f"fire_{f}_hrrr_station_scatter.png", f)
    plot_timeseries(pairs, stations, obs_times, OUT / f"fire_{f}_hrrr_station_timeseries.png", f)
    print(f"\nWrote outputs/fire_{f}_* (stations, pairs, metrics, summary, 3 figures)")


if __name__ == "__main__":
    main()

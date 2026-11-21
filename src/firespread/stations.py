"""HRRR check against surface weather stations for one fire (formerly script 09).

Networks: ASOS (Iowa Environmental Mesonet, no key) and RAWS (Synoptic API;
token in env SYNOPTIC_TOKEN, skipped if unset or refused). Each HRRR hour is
paired with the station report closest to the top of the hour (+/- tolerance)
and the nearest 3 km cell. Hours that were gap-filled in the weather cube are
excluded. Metrics: bias (HRRR - station), MAE, RMSE, r, for all hours and for
offshore / other hours (regional flag from HRRR only). Terrain position of each
station (ridge vs valley) comes from the 30 m DEM, or a 3DEP window when the
station lies outside the fire grid.
"""

from __future__ import annotations

import io as stdio
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile

from . import http, io, logs, manifest
from .config import Config
from .stack import DEM_URL
from .weather import cube_path

log = logging.getLogger(__name__)

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network/{net}.geojson"
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
TOKEN_ENV = "SYNOPTIC_TOKEN"
KT_TO_MS = 0.514444
VARS = ["wspd", "gust", "wdir", "rh", "t2m"]
UNITS = {"wspd": "m/s", "gust": "m/s", "wdir": "deg", "rh": "%", "t2m": "C"}
CUBE_VARS = {"wspd": "wspd10", "gust": "gust", "wdir": "wdir10", "rh": "rh2", "t2m": "t2m"}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def circ_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


# -------------------------------------------------------------- HRRR access
class HrrrCube:
    def __init__(self, path: Path, cfg: Config):
        ds = io.read_cube(path)
        start = pd.Timestamp(ds.attrs["analysis_start_utc"]).tz_convert(None)
        keep = (ds["time"].values >= np.datetime64(start)) & (ds["qc_filled_f00"].values == 0)
        self.ds = ds.isel(time=np.where(keep)[0]).load()
        self.lat, self.lon = self.ds["latitude"].values, self.ds["longitude"].values
        self.times = pd.DatetimeIndex(self.ds["time"].values)
        self.cfg = cfg

    def describe(self) -> str:
        return (
            f"HRRR cube: {len(self.times)} unfilled hours in the analysis period, "
            f"{self.times.min()} to {self.times.max()} UTC; grid {self.lat.shape}, "
            f"lat {self.lat.min():.3f}..{self.lat.max():.3f}, lon {self.lon.min():.3f}..{self.lon.max():.3f}"
        )

    def bbox(self):
        s = self.cfg.stations.box_shrink_deg
        return (self.lon.min() + s, self.lat.min() + s, self.lon.max() - s, self.lat.max() - s)

    def nearest_cell(self, lat, lon):
        d = haversine_km(lat, lon, self.lat, self.lon)
        iy, ix = np.unravel_index(np.argmin(d), d.shape)
        return int(iy), int(ix), float(d[iy, ix])

    def series_at(self, iy, ix) -> pd.DataFrame:
        c = self.ds.isel(y=iy, x=ix)
        return pd.DataFrame({k: c[v].values.astype(float) for k, v in CUBE_VARS.items()}, index=self.times)

    def offshore_hours(self) -> pd.Series:
        o = self.cfg.stations.offshore
        ok = (
            (self.ds["wdir10"] >= o.dir_min)
            & (self.ds["wdir10"] <= o.dir_max)
            & (self.ds["wspd10"] >= o.wspd_min)
            & (self.ds["rh2"] <= o.rh_max)
        )
        return pd.Series(ok.mean(dim=["y", "x"]).values >= o.cell_fraction, index=self.times)


# ---------------------------------------------------------------------- ASOS
def asos_stations(cfg, bbox, cache, refresh):
    f = cache / "asos_CA_ASOS.geojson"
    if refresh or not f.exists():
        log.info("  downloading CA_ASOS station list from IEM")
        url = IEM_NETWORK_URL.format(net="CA_ASOS")
        f.write_text(http.get(url).text, encoding="utf-8")
        manifest.record(cfg.data_dir, source="IEM", url=url, path=f, step="stations")
    gj = json.loads(f.read_text(encoding="utf-8"))
    lonmin, latmin, lonmax, latmax = bbox
    rows = []
    for feat in gj.get("features", []):
        p = feat.get("properties", {}) or {}
        lon, lat = feat["geometry"]["coordinates"][:2]
        if lonmin <= lon <= lonmax and latmin <= lat <= latmax:
            rows.append(
                {
                    "network": "ASOS",
                    "stid": p.get("sid") or feat.get("id"),
                    "name": p.get("sname") or p.get("name") or "",
                    "lat": float(lat),
                    "lon": float(lon),
                    "elev_m": float(p["elevation"]) if p.get("elevation") not in (None, "") else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=["network", "stid", "name", "lat", "lon", "elev_m"])


def asos_data(cfg, stid, t0, t1, cache, refresh):
    f = cache / f"asos_{stid}_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.csv"
    if refresh or not f.exists():
        log.info("  downloading ASOS %s", stid)
        params = [
            ("station", stid),
            ("data", "sknt"),
            ("data", "drct"),
            ("data", "gust"),
            ("data", "tmpf"),
            ("data", "relh"),
            ("sts", t0.strftime("%Y-%m-%dT%H:%MZ")),
            ("ets", t1.strftime("%Y-%m-%dT%H:%MZ")),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("missing", "empty"),
            ("trace", "empty"),
            ("report_type", "3"),
            ("report_type", "4"),
        ]
        r = http.get(IEM_ASOS_URL, params=params)
        f.write_text(r.text, encoding="utf-8")
        manifest.record(cfg.data_dir, source="IEM ASOS", url=r.url, path=f, step="stations")
        time.sleep(1)  # IEM rate limit
    df = pd.read_csv(stdio.StringIO(f.read_text(encoding="utf-8")), na_values=["", "M"])
    if df.empty or "valid" not in df.columns:
        return pd.DataFrame(columns=["time", *VARS])

    def num(c):
        return pd.to_numeric(df[c], errors="coerce") if c in df else np.nan

    return pd.DataFrame(
        {
            "time": pd.to_datetime(df["valid"]),
            "wspd": num("sknt") * KT_TO_MS,
            "gust": num("gust") * KT_TO_MS,
            "wdir": num("drct"),
            "rh": num("relh"),
            "t2m": (num("tmpf") - 32.0) * 5.0 / 9.0,
        }
    )


# ---------------------------------------------------------------------- RAWS
def raws_fetch(cfg, bbox, t0, t1, cache, refresh):
    f = cache / f"raws_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.json"
    token = os.environ.get(TOKEN_ENV, "").strip()
    if refresh or not f.exists():
        if not token:
            log.info("  %s is not set and no cached RAWS file; RAWS skipped", TOKEN_ENV)
            return None
        log.info("  downloading RAWS from Synoptic (token from %s)", TOKEN_ENV)
        lonmin, latmin, lonmax, latmax = bbox
        params = {
            "token": token,
            "bbox": f"{lonmin:.4f},{latmin:.4f},{lonmax:.4f},{latmax:.4f}",
            "start": t0.strftime("%Y%m%d%H%M"),
            "end": t1.strftime("%Y%m%d%H%M"),
            "vars": "wind_speed,wind_direction,wind_gust,air_temp,relative_humidity",
            "network": cfg.stations.raws_network,
            "units": "metric",
            "obtimezone": "UTC",
        }
        try:
            r = http.get(SYNOPTIC_URL, params=params)
            js = r.json()
        except (http.ClientError, RuntimeError, ValueError) as e:
            msg = http.redact(str(e), token)
            log.warning("  Synoptic request failed: %s", msg)
            if "token" in msg.lower():
                log.warning(
                    "  -> %s must be a TOKEN generated from your API key, not the key itself", TOKEN_ENV
                )
            log.warning("  RAWS skipped; continuing with the other networks")
            return None
        code = js.get("SUMMARY", {}).get("RESPONSE_CODE")
        if code != 1:
            log.warning(
                "  Synoptic returned code %s: %s; RAWS skipped",
                code,
                js.get("SUMMARY", {}).get("RESPONSE_MESSAGE"),
            )
            return None
        f.write_text(json.dumps(js), encoding="utf-8")
        manifest.record(
            cfg.data_dir, source="Synoptic RAWS", url=http.redact(r.url, token), path=f, step="stations"
        )
    return json.loads(f.read_text(encoding="utf-8"))


def raws_parse(js, bbox):
    stations, data = [], {}
    lonmin, latmin, lonmax, latmax = bbox
    keymap = {
        "wind_speed": "wspd",
        "wind_direction": "wdir",
        "wind_gust": "gust",
        "air_temp": "t2m",
        "relative_humidity": "rh",
    }
    for st in js.get("STATION", []):
        lat, lon = float(st["LATITUDE"]), float(st["LONGITUDE"])
        if not (lonmin <= lon <= lonmax and latmin <= lat <= latmax):
            continue
        try:
            elev_m = float(st.get("ELEVATION")) * 0.3048  # Synoptic metadata elevation is in feet
        except (TypeError, ValueError):
            elev_m = np.nan
        obs = st.get("OBSERVATIONS", {})
        if "date_time" not in obs:
            continue
        df = pd.DataFrame({"time": pd.to_datetime(obs["date_time"], utc=True).tz_localize(None)})
        for syn, ours in keymap.items():
            keys = sorted(k for k in obs if k.startswith(syn + "_set_"))
            df[ours] = pd.to_numeric(pd.Series(obs[keys[0]]), errors="coerce") if keys else np.nan
        stations.append(
            {
                "network": "RAWS",
                "stid": st["STID"],
                "name": st.get("NAME", ""),
                "lat": lat,
                "lon": lon,
                "elev_m": elev_m,
            }
        )
        data[st["STID"]] = df
    return pd.DataFrame(stations, columns=["network", "stid", "name", "lat", "lon", "elev_m"]), data


# ------------------------------------------------------------------ matching
def match_hourly(obs, hours, tol):
    base = pd.DataFrame({"time": pd.DatetimeIndex(hours).astype("datetime64[ns]")}).sort_values("time")
    out = base.copy()
    for v in VARS:
        if v not in obs or obs[v].isna().all():
            out[v] = np.nan
            continue
        o = obs[["time", v]].dropna().sort_values("time")
        o["time"] = o["time"].astype("datetime64[ns]")
        m = pd.merge_asof(base, o, on="time", direction="nearest", tolerance=tol)
        out[v] = m[v].to_numpy()
    return out.set_index("time")


def metrics(o, h, var):
    ok = np.isfinite(o) & np.isfinite(h)
    o, h = np.asarray(o)[ok], np.asarray(h)[ok]
    n = len(o)
    if n < 3:
        return dict(n=n, obs_mean=np.nan, hrrr_mean=np.nan, bias=np.nan, mae=np.nan, rmse=np.nan, r=np.nan)
    if var == "wdir":
        d = circ_diff(h, o)
        mean_d = np.degrees(np.arctan2(np.sin(np.radians(d)).mean(), np.cos(np.radians(d)).mean()))
        return dict(
            n=n,
            obs_mean=np.nan,
            hrrr_mean=np.nan,
            bias=mean_d,
            mae=np.abs(d).mean(),
            rmse=np.sqrt((d**2).mean()),
            r=np.nan,
        )
    d = h - o
    r = np.corrcoef(o, h)[0, 1] if o.std() > 0 and h.std() > 0 else np.nan
    return dict(
        n=n,
        obs_mean=o.mean(),
        hrrr_mean=h.mean(),
        bias=d.mean(),
        mae=np.abs(d).mean(),
        rmse=np.sqrt((d**2).mean()),
        r=r,
    )


# ------------------------------------------------------------------- terrain
def dem_window_3dep(cfg, stid, x, y, epsg, res, window_m, cache, refresh):
    half, n = window_m / 2, int(round(window_m / res))
    f = cache / f"dem3dep_{stid}.tif"
    if refresh or not f.exists():
        log.info("  downloading 3DEP terrain window for %s", stid)
        params = {
            "bbox": f"{x - half},{y - half},{x + half},{y + half}",
            "bboxSR": epsg,
            "imageSR": epsg,
            "size": f"{n},{n}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "image",
        }
        r = http.get(DEM_URL, params=params)
        if r.content[:4] not in (b"II*\x00", b"MM\x00*"):
            raise RuntimeError(f"3DEP did not return a GeoTIFF: {r.text[:200]}")
        f.write_bytes(r.content)
        manifest.record(cfg.data_dir, source="USGS 3DEP ImageServer", url=r.url, path=f, step="stations")
    with MemoryFile(f.read_bytes()) as mf, mf.open() as src:
        a = src.read(1, masked=True).astype(float).filled(np.nan)
    a[a < -1000] = np.nan
    return a


def terrain_context(cfg, stations, fire_id, cache, refresh):
    from pyproj import Transformer

    dem_path = cfg.fire_dir(fire_id) / "static" / "dem.tif"
    st = stations.copy()
    st["dem_elev_m"], st["rel_elev_3km_m"] = np.nan, np.nan
    st["in_model_grid"], st["terrain_src"] = False, ""
    if not dem_path.exists():
        log.warning("  %s not found; terrain context skipped", dem_path)
        return st, None
    win_m = cfg.stations.terrain_window_m
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype(float).filled(np.nan)
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        half = int(round(win_m / 2 / abs(src.res[0])))
        for i, s in st.iterrows():
            x, y = tf.transform(s["lon"], s["lat"])
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                st.at[i, "in_model_grid"] = True
            if 0 <= row - half and row + half < src.height and 0 <= col - half and col + half < src.width:
                win, srcname = dem[row - half : row + half + 1, col - half : col + half + 1], "grid"
            elif s["n_hours"] > 0:
                try:
                    win = dem_window_3dep(
                        cfg, s["stid"], x, y, src.crs.to_epsg(), abs(src.res[0]), win_m, cache, refresh
                    )
                    srcname = "3dep"
                except Exception as e:
                    log.warning("  %s: 3DEP terrain window failed (%s)", s["stid"], e)
                    continue
            else:
                continue
            c = win[win.shape[0] // 2, win.shape[1] // 2]
            st.at[i, "dem_elev_m"], st.at[i, "rel_elev_3km_m"] = c, c - np.nanmean(win)
            st.at[i, "terrain_src"] = srcname
        info = dict(dem=dem, crs=src.crs, bounds=src.bounds)
    return st, info


# ------------------------------------------------------------------- figures
def plot_map(stations, hr, info, fire_dir, path, fire_id):
    if info is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Transformer

    b = info["bounds"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(
        info["dem"],
        extent=(b.left, b.right, b.bottom, b.top),
        cmap="terrain",
        vmin=-200,
        vmax=np.nanmax(info["dem"]),
    )
    perim = fire_dir / "perimeters" / "perim_last.tif"
    if perim.exists():
        p = io.read_raster(perim, physical=False)[0]
        px = b.left + (np.arange(p.shape[1]) + 0.5) * (b.right - b.left) / p.shape[1]
        py = b.top - (np.arange(p.shape[0]) + 0.5) * (b.top - b.bottom) / p.shape[0]
        ax.contour(px, py, p, levels=[0.5], colors="red", linewidths=1.2)
    tf = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True)
    hx, hy = tf.transform(hr.lon, hr.lat)
    ax.plot(hx.ravel(), hy.ravel(), ".", color="k", ms=1.5, alpha=0.4, label="HRRR cell centres")
    for net, mk, col in [("ASOS", "s", "white"), ("RAWS", "^", "yellow")]:
        s = stations[stations.network == net]
        if s.empty:
            continue
        x, y = tf.transform(s.lon.to_numpy(), s.lat.to_numpy())
        ax.scatter(x, y, marker=mk, s=70, c=col, edgecolors="k", zorder=5, label=net)
        for xi, yi, t in zip(x, y, s.stid):
            ax.annotate(t, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    ax.set_xlim(min(b.left, np.nanmin(hx)), max(b.right, np.nanmax(hx)))
    ax.set_ylim(min(b.bottom, np.nanmin(hy)), max(b.top, np.nanmax(hy)))
    ax.set_title(f"Fire {fire_id}: weather stations used to check HRRR (red = final perimeter)")
    ax.set_xlabel("UTM 11N easting (m)")
    ax.set_ylabel("northing (m)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(pairs, path, fire_id):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, v in zip(axes.ravel(), ["wspd", "gust", "rh", "t2m"]):
        for net, col in [("ASOS", "tab:blue"), ("RAWS", "tab:orange")]:
            d = pairs[pairs.network == net]
            if len(d):
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = pairs.dropna(subset=["obs_wspd"]).groupby("stid").size()
    picks = []
    for net in ["RAWS", "ASOS"]:
        ids = stations[stations.network == net].stid
        picks += list(counts[counts.index.isin(ids)].sort_values(ascending=False).index[:2])
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.4 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, picks):
        d = pairs[pairs.stid == sid]
        ax.plot(d.time, d.obs_wspd, lw=0.9, label=f"{d.network.iloc[0]} {sid} observed")
        ax.plot(d.time, d.hrrr_wspd, lw=0.9, label="HRRR nearest cell")
        for t in obs_times:
            ax.axvline(t, color="grey", lw=0.3, alpha=0.5)
        ax.set_ylabel("wind (m/s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"Fire {fire_id}: 10 m wind speed, HRRR vs stations (grey = VIIRS observation times, UTC)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def run(cfg: Config, fire_id: int, networks=("asos", "raws"), refresh: bool = False) -> pd.DataFrame:
    log_ = logs.setup(cfg.outputs_dir / "logs", "stations", fire_id)
    sc = cfg.stations
    cache = cfg.raw_dir / "stations" / str(fire_id)
    cache.mkdir(parents=True, exist_ok=True)
    hr = HrrrCube(cube_path(cfg, fire_id), cfg)
    log_.info(hr.describe())
    bbox = hr.bbox()
    t0 = hr.times.min().floor("h") - pd.Timedelta(hours=1)
    t1 = hr.times.max().ceil("h") + pd.Timedelta(hours=1)
    obs_times = (
        io.read_table(cfg.fire_dir(fire_id) / "perimeters" / "obs_times.csv", ["obs_time_utc"])[
            "obs_time_utc"
        ]
        .dt.tz_convert(None)
        .tolist()
    )

    frames, obs_by_id = [], {}
    if "asos" in networks:
        log_.info("ASOS (IEM)")
        a = asos_stations(cfg, bbox, cache, refresh)
        if len(a):
            far = np.array([hr.nearest_cell(la, lo)[2] > sc.max_cell_dist_km for la, lo in zip(a.lat, a.lon)])
            if far.any():
                log_.info("  outside the HRRR crop, skipped: %s", ", ".join(a.stid[far]))
            a = a[~far].reset_index(drop=True)
        log_.info("  %d ASOS stations: %s", len(a), ", ".join(a.stid) or "-")
        for sid in a.stid:
            try:
                obs_by_id[sid] = asos_data(cfg, sid, t0, t1, cache, refresh)
            except Exception as e:
                log_.warning("  %s: download failed (%s); skipped", sid, e)
        frames.append(a)
    if "raws" in networks:
        log_.info("RAWS (Synoptic)")
        js = raws_fetch(cfg, bbox, t0, t1, cache, refresh)
        if js is not None:
            r, rd = raws_parse(js, bbox)
            log_.info("  %d RAWS stations: %s", len(r), ", ".join(r.stid) or "-")
            frames.append(r)
            obs_by_id.update(rd)
    stations = (
        pd.concat([f for f in frames if len(f)], ignore_index=True)
        if any(len(f) for f in frames)
        else pd.DataFrame()
    )
    if stations.empty:
        raise RuntimeError("No stations found in the HRRR box")

    offshore = hr.offshore_hours()
    tol = pd.Timedelta(minutes=sc.match_tolerance_min)
    stations[["hrrr_iy", "hrrr_ix", "n_hours"]] = 0
    stations["hrrr_dist_km"] = np.nan
    rows = []
    for i, s in stations.iterrows():
        iy, ix, dist = hr.nearest_cell(s.lat, s.lon)
        stations.loc[i, ["hrrr_iy", "hrrr_ix", "hrrr_dist_km"]] = [iy, ix, round(dist, 2)]
        obs = obs_by_id.get(s.stid)
        if dist > sc.max_cell_dist_km or obs is None or obs.empty:
            continue
        m = match_hourly(obs, hr.times, tol)
        h = hr.series_at(iy, ix)
        df = pd.DataFrame(
            {"time": hr.times, "network": s.network, "stid": s.stid, "offshore": offshore.to_numpy()}
        )
        for v in VARS:
            df[f"obs_{v}"] = m[v].to_numpy()
            df[f"hrrr_{v}"] = h[v].to_numpy()
        weak = (df.obs_wspd < sc.min_wspd_for_dir) | (df.hrrr_wspd < sc.min_wspd_for_dir)
        df.loc[weak, ["obs_wdir", "hrrr_wdir"]] = np.nan
        stations.loc[i, "n_hours"] = int(df.obs_wspd.notna().sum())
        rows.append(df)
    if not rows:
        raise RuntimeError("Stations were found but none had usable data near an HRRR cell")
    pairs = pd.concat(rows, ignore_index=True)
    stations, info = terrain_context(cfg, stations, fire_id, cache, refresh)

    mrows = []
    groups = [
        (s.network, s.stid, pairs[pairs.stid == s.stid]) for s in stations.itertuples() if s.n_hours > 0
    ]
    groups += [(net, "ALL", pairs[pairs.network == net]) for net in pairs.network.unique()]
    for net, sid, d in groups:
        for regime, dd in (("all", d), ("offshore", d[d.offshore]), ("other", d[~d.offshore])):
            for v in VARS:
                mrows.append(
                    dict(
                        network=net,
                        stid=sid,
                        regime=regime,
                        variable=v,
                        units=UNITS[v],
                        **metrics(dd[f"obs_{v}"].to_numpy(), dd[f"hrrr_{v}"].to_numpy(), v),
                    )
                )
    met = pd.DataFrame(mrows)

    out = cfg.outputs_dir
    io.write_table(stations, out / f"fire_{fire_id}_stations.csv")
    io.write_table(pairs, out / f"fire_{fire_id}_hrrr_station_pairs.csv")
    io.write_table(met, out / f"fire_{fire_id}_hrrr_station_metrics.csv")

    fmt = lambda x: f"{x:.2f}"  # noqa: E731
    o = sc.offshore
    rep = logs.Report(log_)
    rep(f"HRRR vs surface stations, fire {fire_id}")
    rep("=" * 50)
    rep(hr.describe())
    rep(
        f"Station box {tuple(round(float(b), 3) for b in bbox)}; reports matched within +/-{sc.match_tolerance_min} min"
    )
    rep(
        f"Offshore hours (>= {o.cell_fraction:.0%} of HRRR cells with wind from {o.dir_min:.0f}-{o.dir_max:.0f} deg, "
        f">= {o.wspd_min} m/s, RH <= {o.rh_max}%): {int(offshore.sum())} of {len(offshore)}"
    )
    rep(f"Bias = HRRR - station. Direction compared only when both speeds >= {sc.min_wspd_for_dir} m/s.\n")
    rep("Stations\n" + "-" * 50)
    cols = [
        "network",
        "stid",
        "name",
        "lat",
        "lon",
        "elev_m",
        "dem_elev_m",
        "rel_elev_3km_m",
        "terrain_src",
        "in_model_grid",
        "hrrr_dist_km",
        "n_hours",
    ]
    rep.block(stations[cols].to_string(index=False, float_format=fmt))
    rep("\nNetwork totals (all hours)\n" + "-" * 50)
    t = met[(met.stid == "ALL") & (met.regime == "all")]
    rep.block(
        t[
            ["network", "variable", "units", "n", "obs_mean", "hrrr_mean", "bias", "mae", "rmse", "r"]
        ].to_string(index=False, float_format=fmt)
    )
    rep("\nNetwork totals by regime: wind speed, gust and RH\n" + "-" * 50)
    t = met[(met.stid == "ALL") & met.variable.isin(["wspd", "gust", "rh"])]
    rep.block(
        t[["network", "regime", "variable", "n", "bias", "rmse"]].to_string(index=False, float_format=fmt)
    )
    rep("\nWind speed by station, with terrain position\n" + "-" * 50)
    w = (
        met[(met.stid != "ALL") & (met.regime == "all") & (met.variable == "wspd")]
        .merge(stations[["stid", "rel_elev_3km_m", "dem_elev_m"]], on="stid")
        .sort_values("rel_elev_3km_m", ascending=False)
    )
    rep.block(
        w[
            ["network", "stid", "dem_elev_m", "rel_elev_3km_m", "n", "obs_mean", "hrrr_mean", "bias", "rmse"]
        ].to_string(index=False, float_format=fmt)
    )
    ok = w.dropna(subset=["rel_elev_3km_m", "bias"])
    if len(ok) >= 4 and ok.rel_elev_3km_m.std() > 0:
        rep(
            f"\nCorrelation of station wind bias with relative elevation (n={len(ok)}): "
            f"r = {np.corrcoef(ok.rel_elev_3km_m, ok.bias)[0, 1]:.2f} "
            "(negative = HRRR under-predicts more at ridge-like stations)"
        )
    rep("\nNotes")
    rep(
        "- ASOS: 10 m, 2-min mean wind; gusts only reported when gusty, so gust stats cover gusty hours only."
    )
    rep("  RAWS: ~6.1 m (20 ft), 10-min mean wind, hourly peak gust; expected to read lower than 10 m wind.")
    rep("- HRRR value is the nearest 3 km cell; gap-filled HRRR hours are excluded.")
    rep("- rel_elev_3km_m = station DEM height minus mean DEM height in a 3 km window; > 0 ridge-like.")
    rep.save(out / f"fire_{fire_id}_hrrr_station_summary.txt")

    plot_map(stations, hr, info, cfg.fire_dir(fire_id), out / f"fire_{fire_id}_station_map.png", fire_id)
    plot_scatter(pairs, out / f"fire_{fire_id}_hrrr_station_scatter.png", fire_id)
    plot_timeseries(pairs, stations, obs_times, out / f"fire_{fire_id}_hrrr_station_timeseries.png", fire_id)
    return met

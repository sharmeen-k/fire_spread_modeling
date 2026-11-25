"""Hourly HRRR surface weather for one fire (formerly script 08, extended).

For every hour from (first observation - spin-up) to (last observation + 1 h):
  * HRRR analysis (f00): UGRD/VGRD 10 m, TMP/RH 2 m, GUST
  * HRRR 1-h forecast issued an hour earlier (f01): DSWRF and APCP (0-1 h);
    analyses carry no solar radiation or rainfall
  * once per fire: the HRRR land mask
Each hour is cropped to the fire grid box (+ margin) at native 3 km and cached,
so re-runs resume. Winds are rotated from grid-relative to earth-relative.
Hours missing from the archive are filled by linear interpolation in time
(rain: 0) and flagged in qc_filled_f00 / qc_filled_f01.

Output: data/processed/fires/<id>/weather/hrrr_hourly.zarr (CF cube, see io.py)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from rasterio.warp import transform_bounds

from . import io, logs, manifest
from .config import Config

log = logging.getLogger(__name__)

SEARCH_F00 = r":(?:UGRD|VGRD):10 m above|:(?:TMP|RH):2 m above|:GUST:surface"
SEARCH_F01 = r":(?:DSWRF|APCP):surface:"
SEARCH_LAND = r":LAND:surface:"
F00_SHORT = {"10u": "u10", "10v": "v10", "2t": "t2m", "2r": "rh2", "gust": "gust"}
F00_NAME = {"u10": "u10", "v10": "v10", "t2m": "t2m", "r2": "rh2", "gust": "gust"}
F00_NEEDED = ["u10", "v10", "t2m", "rh2", "gust"]
DEFAULT_LOV, DEFAULT_LATIN1 = 262.5, 38.5


def cache_dir(cfg: Config, fire_id: int) -> Path:
    return cfg.raw_dir / "hrrr" / str(fire_id)


def cube_path(cfg: Config, fire_id: int) -> Path:
    return cfg.fire_dir(fire_id) / "weather" / "hrrr_hourly.zarr"


# ------------------------------------------------------------------ helpers
def crop_window(lat, lon, box):
    w, s, e, n = box
    lon = io.lon180(lon)
    m = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    ys, xs = np.where(m)
    if ys.size == 0:
        raise ValueError("Crop box does not overlap the HRRR grid")
    return slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)


def rotate_to_earth(u, v, lon, lov, latin1):
    """Grid-relative -> earth-relative for Lambert conformal (tangent cone);
    same formula as NCL wrf_uvmet / wgrib2 -new_grid_winds earth."""
    n = np.sin(np.radians(latin1))
    alpha = np.radians(n * (io.lon180(lon) - io.lon180(lov)))
    ue = np.cos(alpha) * u + np.sin(alpha) * v
    ve = -np.sin(alpha) * u + np.cos(alpha) * v
    return ue, ve, np.degrees(alpha)


def wind_dir_from(ue, ve):
    return (270.0 - np.degrees(np.arctan2(ve, ue))) % 360.0


def _datasets(ds):
    return ds if isinstance(ds, list) else [ds]


def _herbie(init: pd.Timestamp, fxx: int, cfg: Config, tmp: Path):
    from herbie import Herbie

    return Herbie(
        init.strftime("%Y-%m-%d %H:%M"),
        model="hrrr",
        product="sfc",
        fxx=fxx,
        priority=cfg.hrrr.priority,
        save_dir=tmp,
        verbose=False,
    )


class HrrrCache:
    """Per-hour crops of HRRR fields, cached as NetCDF under data/raw/hrrr/<id>/."""

    def __init__(self, cfg: Config, fire_id: int, box, retry_missing: bool = False, record: bool = True):
        self.cfg, self.fire_id, self.box = cfg, fire_id, box
        self.retry_missing = retry_missing
        self.record = record  # worker processes leave manifest writing to the parent
        self.last_url = ""
        self.dir = cache_dir(cfg, fire_id)
        self.tmp = cfg.raw_dir / "hrrr" / "_tmp"
        (self.dir / "f01").mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.logged: set[str] = set()

    def path(self, kind: str, t: pd.Timestamp) -> Path:
        # f00 files keep the original script-08 layout so existing caches are reused
        return self.dir / f"{t:%Y%m%d%H}.nc" if kind == "f00" else self.dir / "f01" / f"{t:%Y%m%d%H}.nc"

    def _save(self, fields, lat, lon, path, url, attrs=None):
        ys, xs = crop_window(lat, lon, self.box)
        ds = xr.Dataset(
            {k: (("y", "x"), np.asarray(v, dtype="float32")[ys, xs]) for k, v in fields.items()},
            coords={
                "lat": (("y", "x"), lat[ys, xs].astype("float64")),
                "lon": (("y", "x"), io.lon180(lon[ys, xs])),
            },
            attrs=attrs or {},
        )
        ds.to_netcdf(path)
        self.last_url = str(url)
        if not self.record:
            return
        manifest.record(
            self.cfg.data_dir,
            source="NOAA HRRR (Herbie)",
            url=str(url),
            path=path,
            step="weather",
            fire_id=self.fire_id,
        )

    def _inventory_once(self, H, search, key):
        if key in self.logged:
            return
        self.logged.add(key)
        try:
            lines = H.inventory(search)["search_this"].tolist()
            log.info("  GRIB messages used for %s: %s", key, "; ".join(lines))
        except Exception:  # inventory is informational only
            pass

    def get(self, kind: str, t: pd.Timestamp) -> tuple[Path | None, str]:
        """kind f00: analysis valid at t. f01: forecast from t-1h valid at t."""
        f = self.path(kind, t)
        if f.exists():
            return f, "cached"
        marker = f.with_suffix(".missing")  # archive confirmed it has no file for this hour
        if marker.exists() and not self.retry_missing:
            return None, "missing (known)"
        init, fxx, search = (
            (t, 0, SEARCH_F00) if kind == "f00" else (t - pd.Timedelta(hours=1), 1, SEARCH_F01)
        )
        try:
            H = _herbie(init, fxx, self.cfg, self.tmp)
            if getattr(H, "grib", None) is None:
                marker.write_text(f"not in archive (checked {io.now_utc()})", encoding="utf-8")
                return None, "missing"
            self._inventory_once(H, search, kind)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dss = _datasets(H.xarray(search, remove_grib=True))
        except Exception as exc:  # missing archive file, network, decode error
            return None, f"error: {type(exc).__name__}: {str(exc)[:120]}"
        lat, lon = dss[0]["latitude"].values, dss[0]["longitude"].values
        fields, attrs = {}, {}
        for d in dss:
            step = d.attrs.get("GRIB_stepType", "")
            for name, da in d.data_vars.items():
                short = da.attrs.get("GRIB_shortName", "")
                if kind == "f00":
                    key = F00_SHORT.get(short) or F00_NAME.get(name)
                    if key:
                        fields[key] = da.values
                        if key == "u10":
                            attrs["lov"] = float(da.attrs.get("GRIB_LoVInDegrees", DEFAULT_LOV))
                            attrs["latin1"] = float(da.attrs.get("GRIB_Latin1InDegrees", DEFAULT_LATIN1))
                else:
                    st = da.attrs.get("GRIB_stepType", step)
                    if short == "dswrf" or name.lower().endswith("dswrf"):
                        if "dswrf" not in fields or st == "instant":  # prefer the value at t
                            fields["dswrf"] = da.values
                    elif short in ("tp", "apcp") or name in ("tp", "apcp"):
                        fields["apcp"] = da.values
        need = F00_NEEDED if kind == "f00" else ["dswrf", "apcp"]
        lacking = [k for k in need if k not in fields]
        if lacking:
            return None, f"error: fields missing {lacking}"
        self._save(fields, lat, lon, f, getattr(H, "grib", ""), attrs)
        return f, "downloaded"

    def land(self, t: pd.Timestamp) -> np.ndarray | None:
        f = self.dir / "land.nc"
        if not f.exists():
            try:
                H = _herbie(t, 0, self.cfg, self.tmp)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = _datasets(H.xarray(SEARCH_LAND, remove_grib=True))[0]
                name = next(iter(d.data_vars))
                self._save(
                    {"land": d[name].values},
                    d["latitude"].values,
                    d["longitude"].values,
                    f,
                    getattr(H, "grib", ""),
                )
                if not self.record:  # _save() already recorded it otherwise
                    manifest.record(
                        self.cfg.data_dir,
                        source="NOAA HRRR (Herbie)",
                        url=self.last_url,
                        path=f,
                        step="weather",
                        fire_id=self.fire_id,
                    )
            except Exception as exc:
                log.warning("Land mask not available (%s); all cells treated as land", str(exc)[:120])
                return None
        with xr.open_dataset(f) as d:
            return (d["land"].values > 0.5).astype("int8")


# ------------------------------------------------------------------ download
def _fetch_task(cfg: Config, fire_id: int, box, retry_missing: bool, kind: str, t: pd.Timestamp):
    """Runs in a worker process: fetch one field set for one hour."""
    c = HrrrCache(cfg, fire_id, box, retry_missing=retry_missing, record=False)
    f, st = c.get(kind, t)
    return kind, t, f, st, c.last_url


def download(cfg: Config, fire_id: int, box, cache: HrrrCache, hours, retry_missing: bool):
    """Fetch every (kind, hour) not yet cached, in parallel processes (cfg.hrrr.workers)."""
    got = {"f00": {}, "f01": {}}
    status = {"f00": {}, "f01": {}}
    todo = []
    for t in hours:
        for kind in ("f00", "f01"):
            f = cache.path(kind, t)
            if f.exists():
                got[kind][t], status[kind][t] = f, "cached"
            elif f.with_suffix(".missing").exists() and not retry_missing:
                status[kind][t] = "missing (known)"
            else:
                todo.append((kind, t))
    log.info(
        "HRRR: %d field sets cached, %d to fetch with %d worker(s)",
        sum(map(len, got.values())),
        len(todo),
        cfg.hrrr.workers,
    )

    def done(kind, t, f, st, url, n):
        status[kind][t] = st
        if f is not None:
            got[kind][t] = Path(f)
            if st == "downloaded":
                manifest.record(
                    cfg.data_dir,
                    source="NOAA HRRR (Herbie)",
                    url=url,
                    path=Path(f),
                    step="weather",
                    fire_id=fire_id,
                )
        elif st != "missing (known)":
            log.info("  %s %s %s", kind, f"{t:%Y-%m-%d %H}Z", st)
        if n % 48 == 0 or n == len(todo):
            log.info("  %d/%d fetched", n, len(todo))

    # the first task of each kind runs here, so its GRIB inventory is logged
    n = 0
    for kind in ("f00", "f01"):
        first = next((x for x in todo if x[0] == kind), None)
        if first:
            todo.remove(first)
            todo.insert(0, first)
    serial = todo if cfg.hrrr.workers == 1 else todo[:2]
    for kind, t in serial:
        f, st = cache.get(kind, t)
        n += 1
        done(kind, t, f, st, cache.last_url, n)
    rest = todo[len(serial) :]
    if rest:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=cfg.hrrr.workers) as pool:
            futs = [pool.submit(_fetch_task, cfg, fire_id, box, retry_missing, kind, t) for kind, t in rest]
            for fut in as_completed(futs):
                n += 1
                done(*fut.result(), n)
    return got, status


# ------------------------------------------------------------------ assemble
def _stack(files: dict, hours: pd.DatetimeIndex, names: list[str]):
    """Load cached hours -> dict name -> (nt, ny, nx) array with NaN where missing."""
    first = next(iter(files.values()))
    with xr.open_dataset(first) as d0:
        lat, lon = d0["lat"].values, d0["lon"].values
        attrs = dict(d0.attrs)
    arr = {n: np.full((len(hours), *lat.shape), np.nan, dtype="float32") for n in names}
    pos = {t: i for i, t in enumerate(hours)}
    for t, f in files.items():
        with xr.open_dataset(f) as d:
            for n in names:
                arr[n][pos[t]] = d[n].values
    return arr, lat, lon, attrs


def _fill(a: np.ndarray, hours: pd.DatetimeIndex, zero: bool = False) -> np.ndarray:
    df = pd.DataFrame(a.reshape(len(hours), -1), index=hours)
    df = df.fillna(0.0) if zero else df.interpolate(method="time", limit_direction="both")
    return df.to_numpy(dtype="float32").reshape(a.shape)


def plot_series(path, ts, obs, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    t = ts.index
    ax[0].plot(t, ts["wspd10"], label="10 m wind")
    ax[0].plot(t, ts["gust"], label="gust", alpha=0.7)
    ax[0].set_ylabel("m/s")
    ax[0].legend(loc="upper right")
    ax[1].scatter(t, ts["wdir10"], s=4)
    ax[1].axhspan(0, 110, color="orange", alpha=0.15, label="N-E offshore sector")
    ax[1].set_ylim(0, 360)
    ax[1].set_yticks([0, 90, 180, 270, 360])
    ax[1].set_ylabel("dir from (deg)")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, ts["rh2"], color="tab:green")
    ax[2].set_ylabel("RH 2 m (%)")
    ax[3].plot(t, ts["t2m"], color="tab:red")
    ax[3].set_ylabel("T 2 m (C)")
    for a in ax:
        for o in obs:
            a.axvline(o, color="grey", lw=0.4, alpha=0.5)
    ax[3].set_xlabel("UTC (grey lines = VIIRS observation times)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def run(cfg: Config, fire_id: int, max_hours: int | None = None, retry_missing: bool = False) -> Path:
    log_ = logs.setup(cfg.outputs_dir / "logs", "weather", fire_id)
    rep = logs.Report(log_)
    fdir = cfg.fire_dir(fire_id)
    grid = io.Grid.from_json(fdir / "grid.json")
    obs = io.read_table(fdir / "perimeters" / "obs_times.csv", ["obs_time_utc"])
    t_obs = io.to_utc(obs["obs_time_utc"])

    lead = pd.Timedelta(hours=cfg.hrrr.lead_hours)
    spin = pd.Timedelta(days=cfg.fuel_moisture.spinup_days)
    analysis_start = (t_obs[0] - lead).floor("h")
    t0 = (t_obs[0] - max(lead, spin)).floor("h")
    t1 = (t_obs[-1] + pd.Timedelta(hours=1)).ceil("h")
    hours = pd.date_range(t0, t1, freq="h")
    if max_hours:
        hours = hours[:max_hours]
    w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    m = cfg.hrrr.margin_deg
    box = (w - m, s - m, e + m, n + m)
    rep(f"fireID {fire_id}: {len(t_obs)} observations {t_obs[0]} -> {t_obs[-1]}")
    rep(
        f"HRRR hours {hours[0]} -> {hours[-1]} ({len(hours)}); analysis period from {analysis_start} "
        f"(earlier hours = {cfg.fuel_moisture.spinup_days:g}-day fuel moisture spin-up)"
    )
    rep(f"Box W {box[0]:.3f} S {box[1]:.3f} E {box[2]:.3f} N {box[3]:.3f}")

    cache = HrrrCache(cfg, fire_id, box, retry_missing=retry_missing, record=False)  # download() records
    got, status = download(cfg, fire_id, box, cache, hours, retry_missing)
    for kind in ("f00", "f01"):
        st = status[kind].values()
        rep(
            f"{kind}: ok {len(got[kind])}/{len(hours)} (downloaded {sum(x == 'downloaded' for x in st)}, "
            f"cached {sum(x == 'cached' for x in st)}); missing "
            + (", ".join(f"{t:%m-%d %H}Z" for t in hours if t not in got[kind]) or "none")
        )
    if not got["f00"] or not got["f01"]:
        raise RuntimeError("No HRRR hours available; check the network and re-run")
    miss = 1 - len(got["f00"]) / len(hours)
    if miss > cfg.hrrr.max_missing_share:
        raise RuntimeError(
            f"{miss:.0%} of HRRR analysis hours are missing (limit {cfg.hrrr.max_missing_share:.0%}). This is usually "
            "a network or throttling problem during download, not the archive: re-run with --retry-missing "
            "(fewer workers helps). Raise hrrr.max_missing_share only if the archive really lacks the hours."
        )

    a00, lat, lon, attrs = _stack(got["f00"], hours, F00_NEEDED)
    a01, _, _, _ = _stack(got["f01"], hours, ["dswrf", "apcp"])
    lov, latin1 = attrs.get("lov", DEFAULT_LOV), attrs.get("latin1", DEFAULT_LATIN1)
    ue, ve, alpha = rotate_to_earth(a00["u10"], a00["v10"], lon, lov, latin1)
    ue, ve = _fill(ue, hours), _fill(ve, hours)
    t2m = _fill(a00["t2m"], hours) - 273.15
    rh2 = np.clip(_fill(a00["rh2"], hours), 0.0, 100.0)
    gust = _fill(a00["gust"], hours)
    dswrf = np.clip(_fill(a01["dswrf"], hours), 0.0, None)
    apcp = np.clip(_fill(a01["apcp"], hours, zero=True), 0.0, None)
    qc00 = np.array([t not in got["f00"] for t in hours], dtype="int8")
    qc01 = np.array([t not in got["f01"] for t in hours], dtype="int8")
    land = None
    for t in sorted(got["f00"])[:: max(1, len(got["f00"]) // 5)][
        :6
    ]:  # the land mask comes from any archived hour
        land = cache.land(t)
        if land is not None:
            break
    if land is None:
        raise RuntimeError("Could not download the HRRR land mask; check the network and re-run")

    xs, ys = io.hrrr_projected_xy(lat, lon)
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xu, yu = tr.transform(lon, lat)
    ds = io.new_cube(
        hours, lat, lon, y=("y", ys), x=("x", xs), x_utm=(("y", "x"), xu), y_utm=(("y", "x"), yu)
    )
    ds["y"].attrs.update(standard_name="projection_y_coordinate", units="m")
    ds["x"].attrs.update(standard_name="projection_x_coordinate", units="m")
    ds["crs"] = xr.DataArray(np.int32(0), attrs=io.HRRR_GRID_MAPPING)
    dims = ("time", "y", "x")
    ds["u10"], ds["v10"] = (dims, ue.astype("float32")), (dims, ve.astype("float32"))
    ds["wspd10"] = (dims, np.hypot(ue, ve).astype("float32"))
    ds["wdir10"] = (dims, wind_dir_from(ue, ve).astype("float32"))
    ds["gust"], ds["t2m"], ds["rh2"] = (dims, gust), (dims, t2m.astype("float32")), (dims, rh2)
    ds["dswrf"], ds["apcp"] = (dims, dswrf), (dims, apcp)
    ds["qc_filled_f00"], ds["qc_filled_f01"] = ("time", qc00), ("time", qc01)
    if land is not None:
        ds["land"] = (("y", "x"), land)
    out = io.write_cube(
        ds,
        cube_path(cfg, fire_id),
        title=f"HRRR hourly surface weather, fire {fire_id}",
        source="NOAA HRRR via Herbie (AWS/Google archive): f00 analyses; DSWRF/APCP from f01 forecasts",
        fireID=fire_id,
        analysis_start_utc=io.iso_z(analysis_start),
        spinup_start_utc=io.iso_z(t0),
        hrrr_lov=lov,
        hrrr_latin1=latin1,
        comment="Winds rotated from HRRR grid-relative to earth-relative. Missing archive hours are "
        "time-interpolated and flagged in qc_filled_*.",
    )
    rep(f"\nRotation: LoV {lov}, Latin1 {latin1} -> {alpha.min():.1f} to {alpha.max():.1f} deg over the box")
    rep(
        f"Crop: {lat.shape[0]} x {lat.shape[1]} HRRR cells (~3 km); land cells "
        f"{'n/a' if land is None else int(land.sum())}"
    )
    period = ds.sel(time=slice(io.utc_naive([analysis_start])[0], None))
    for k in ["wspd10", "gust", "wdir10", "rh2", "t2m", "dswrf", "apcp"]:
        v = period[k].values
        rep(f"  {k:<7} min {np.nanmin(v):8.1f}  max {np.nanmax(v):8.1f}  mean {np.nanmean(v):8.2f}")

    # ignition-area time series (analysis period)
    rec = io.read_table(fdir / "perimeters" / "obs_times.csv")
    lat0, lon0 = float(rec["clat"].iloc[0]), float(rec["clon"].iloc[0])
    d2 = (lat - lat0) ** 2 + ((lon - lon0) * np.cos(np.radians(lat0))) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    cell = period.isel(y=iy, x=ix)
    ts = pd.DataFrame(
        {k: cell[k].values for k in ["wspd10", "gust", "wdir10", "rh2", "t2m"]},
        index=pd.DatetimeIndex(cell["time"].values),
    )
    o = cfg.stations.offshore
    off = ts["wdir10"].between(o.dir_min, o.dir_max) & (ts["wspd10"] >= o.wspd_min) & (ts["rh2"] <= o.rh_max)
    rep(
        f"\nIgnition-area cell ({lat[iy, ix]:.3f}, {lon[iy, ix]:.3f}): offshore-rule hours {off.sum()} of {len(ts)}"
    )
    first2 = ts.loc[ts.index <= ts.index[0] + pd.Timedelta(hours=48)]
    rep(
        f"  first 48 h: max wind {first2['wspd10'].max():.1f} m/s, max gust {first2['gust'].max():.1f} m/s, "
        f"min RH {first2['rh2'].min():.0f}%"
    )
    png = cfg.outputs_dir / f"fire_{fire_id}_hrrr_timeseries.png"
    plot_series(
        png,
        ts,
        t_obs.tz_convert(None),
        f"fireID {fire_id}: HRRR at cell nearest ignition ({lat0:.3f}, {lon0:.3f})",
    )
    rep(f"\nWrote {out.relative_to(cfg.root)} and {png.relative_to(cfg.root)}")
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_hrrr_summary.txt")
    return out

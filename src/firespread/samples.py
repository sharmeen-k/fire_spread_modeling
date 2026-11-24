"""Training samples: one per 12-hour perimeter step, on a coarse ML grid around each fire.

The ML grid is the fire's 30 m grid aggregated by `samples.block` (default 12 -> 360 m, close to the
375 m VIIRS pixels behind the Largefire perimeters). For step k (observation k -> k+1):

inputs (channels, see CHANNELS)
- burned_t: fraction of each block inside perimeter k
- terrain: elevation, slope, aspect as sin/cos (block means)
- fuels: fraction of each Scott & Burgan group (GR, GS, SH, TU, TL, SB, NB); canopy CC, CH, CBH, CBD
- weather over (t_k, t_k+1]: mean u10/v10, max wind speed, max gust, mean temperature, min RH, total rain
  (HRRR, bilinear to block centres; ocean cells filled from the nearest land cell)
- fuel moisture over the step: mean 1-h, 10-h, 100-h (fm100 per the config switch); live herb/woody
- physics (Rothermel, behave port): every `rothermel_hours` hours on every `rothermel_stride`-th 30 m cell,
  then block means of head-fire spread rate, its x/y components along the spread direction, and
  fireline intensity; the max over the step of the block-mean spread rate
- step descriptors as constant planes: hours since the first observation, step length, local hour of t_k
target
- burned_t1: fraction of each block inside perimeter k+1

Output per fire: data/processed/fires/<id>/samples/samples.zarr with `inputs` (sample, channel, y, x),
`target` (sample, y, x), `valid` (y, x: block fully inside the fire grid) and per-sample times. These
are the shards uploaded to Google Drive for training on Colab.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from pyproj import CRS, Transformer

from . import gridmet, io, logs
from .config import Config
from .fuel_moisture import surface_dead_moisture
from .live_fuel_moisture import cube_path as live_path
from .perimeters import load_records
from .surface_spread import (
    Bilinear,
    SpreadInputs,
    fill,
    nearest_land_index,
    true_north_azimuth,
    window,
    window_transform,
)
from .weather import cube_path as weather_path

log = logging.getLogger(__name__)

FUEL_GROUPS = {
    "NB": (91, 99),
    "GR": (101, 109),
    "GS": (121, 124),
    "SH": (141, 149),
    "TU": (161, 165),
    "TL": (181, 189),
    "SB": (201, 204),
}
STATIC = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    *[f"fuel_{g}" for g in FUEL_GROUPS],
    "cc",
    "ch",
    "cbh",
    "cbd",
    "nodata",
]
WEATHER = ["u10_mean", "v10_mean", "wspd_max", "gust_max", "t2m_mean", "rh_min", "rain_sum"]
MOISTURE = ["fm1_mean", "fm10_mean", "fm100_mean", "lfm_herb", "lfm_woody"]
PHYSICS = ["ros_mean", "ros_x", "ros_y", "ros_blockmax", "fli_mean"]
STEP = ["hours_since_start", "step_hours", "local_hour"]
CHANNELS = ["burned_t", *STATIC, *WEATHER, *MOISTURE, *PHYSICS, *STEP]


def out_path(cfg: Config, fire_id: int):
    return cfg.fire_dir(fire_id) / "samples" / "samples.zarr"


def block_mean(a: np.ndarray, b: int) -> np.ndarray:
    """NaN-aware mean over b x b blocks (array already a multiple of b in both directions)."""
    h, w = a.shape[0] // b, a.shape[1] // b
    v = a.reshape(h, b, w, b)
    with np.errstate(invalid="ignore"):
        n = np.isfinite(v).sum((1, 3))
        s = np.nansum(v, (1, 3))
        return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype("float32")


def run(cfg: Config, fire_id: int) -> None:
    log_ = logs.setup(cfg.outputs_dir / "logs", "samples", fire_id)
    sc = cfg.samples
    b = sc.block
    fdir = cfg.fire_dir(fire_id)
    grid = io.Grid.from_json(fdir / "grid.json")
    recs = load_records(cfg.socal_perimeters, fire_id).to_crs(grid.crs)
    if len(recs) < 2:
        raise RuntimeError(f"fire {fire_id} has {len(recs)} perimeter record(s); need 2+")
    union, _ = io.read_raster(fdir / "perimeters" / "perim_union.tif", physical=False)
    win = window(union > 0, int(sc.buffer_m / grid.res))
    # extend the window to whole blocks (may run past the grid; those cells are NaN)
    r0, c0 = win[0].start, win[1].start
    nby, nbx = -(-(win[0].stop - r0) // b), -(-(win[1].stop - c0) // b)
    H, W = nby * b, nbx * b
    full = (slice(r0, r0 + H), slice(c0, c0 + W))
    inside = (slice(r0, min(r0 + H, grid.height)), slice(c0, min(c0 + W, grid.width)))
    hh, ww = inside[0].stop - r0, inside[1].stop - c0
    tr = window_transform(grid, full)
    log_.info(
        "Fire %d: %d records -> %d samples; ML grid %d x %d blocks of %d m",
        fire_id,
        len(recs),
        len(recs) - 1,
        nbx,
        nby,
        b * grid.res,
    )

    def padded(a):
        out = np.full((H, W), np.nan, dtype="float64")
        out[:hh, :ww] = a
        return out

    # ------------------------------------------------------------- static
    sdir = fdir / "static"
    rd = lambda k, **kw: io.read_raster(sdir / f"{k}.tif", **kw)[0][inside]  # noqa: E731
    fb = rd("fbfm40", physical=False).astype(np.int64)
    fbp = padded(np.where((fb > 0) & (fb < 256), fb, np.nan))
    static = {"elevation": block_mean(padded(rd("dem")), b), "slope": block_mean(padded(rd("slope")), b)}
    asp = padded(rd("aspect"))
    asp = np.where(asp < 0, np.nan, asp)
    static["aspect_sin"] = block_mean(np.sin(np.deg2rad(asp)), b)
    static["aspect_cos"] = block_mean(np.cos(np.deg2rad(asp)), b)
    for g, (lo, hi) in FUEL_GROUPS.items():
        static[f"fuel_{g}"] = block_mean(((fbp >= lo) & (fbp <= hi)).astype(float), b)  # share of all cells
    for k in ("cc", "ch", "cbh", "cbd"):
        static[k] = block_mean(padded(rd(k)), b)
    static["nodata"] = block_mean(np.where(np.isfinite(fbp), 0.0, 1.0), b)
    valid = np.zeros((nby, nbx), bool)
    valid[: hh // b, : ww // b] = True

    # ------------------------------------------------- block centres -> HRRR
    ys = tr.f + (np.arange(nby) + 0.5) * b * tr.e
    xs = tr.c + (np.arange(nbx) + 0.5) * b * tr.a
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(X, Y)
    wx = io.read_cube(weather_path(cfg, fire_id))
    hx, hy = Transformer.from_crs("EPSG:4326", CRS.from_cf(io.HRRR_GRID_MAPPING), always_xy=True).transform(
        lon, lat
    )
    interp = Bilinear(wx["x"].values, wx["y"].values, hx.ravel(), hy.ravel())
    nearest = nearest_land_index(wx["land"].values.astype(bool)) if "land" in wx else None
    north = np.deg2rad(true_north_azimuth(lon.ravel(), lat.ravel(), grid.crs)).reshape(nby, nbx)

    def field(ds, var, idx):
        a = np.asarray(ds[var].isel(time=idx).values, float)
        return np.stack([interp(fill(a[i], nearest)) for i in range(a.shape[0])]).reshape(-1, nby, nbx)

    dead = surface_dead_moisture(cfg, fire_id)
    live = io.read_cube(live_path(cfg, fire_id))
    live_days = gridmet.local_day(live["time"].values)
    wx_t = pd.DatetimeIndex(wx["time"].values)
    dead_t = pd.DatetimeIndex(dead["time"].values)

    # Rothermel on a sub-sampled set of 30 m cells of the (in-grid part of the) window
    spread = SpreadInputs(cfg, fire_id, inside, stride=sc.rothermel_stride)

    def raster(geom):
        m = rasterio.features.rasterize([(geom, 1)], out_shape=(H, W), transform=tr, dtype="uint8")
        return block_mean(m.astype(float), b)

    t_obs = io.utc_naive(recs["obs_time_utc"])
    n = len(recs) - 1
    X_ = np.full((n, len(CHANNELS), nby, nbx), np.nan, dtype="float32")
    Y_ = np.full((n, nby, nbx), np.nan, dtype="float32")
    burned = [raster(g) for g in recs.geometry]
    t0 = time.time()
    for k in range(n):
        ta, tb = t_obs[k], t_obs[k + 1]
        hrs = wx_t[(wx_t > ta) & (wx_t <= tb) & wx_t.isin(dead_t)]
        if len(hrs) == 0:  # step shorter than an hour: the nearest modelled hour at or before t_k+1
            hrs = dead_t[[int(np.clip(np.searchsorted(dead_t, tb, side="right") - 1, 0, len(dead_t) - 1))]]
        iw = wx_t.get_indexer(hrs)
        idd = dead_t.get_indexer(hrs)
        ch = {"burned_t": burned[k], **static}
        u, v = field(wx, "u10", iw), field(wx, "v10", iw)
        # rotate earth-relative u/v to grid axes (true north lies at grid azimuth `north`)
        cn, sn = np.cos(north), np.sin(north)
        ug, vg = u * cn + v * sn, -u * sn + v * cn
        ch.update(
            u10_mean=ug.mean(0),
            v10_mean=vg.mean(0),
            wspd_max=np.hypot(u, v).max(0),
            gust_max=field(wx, "gust", iw).max(0),
            t2m_mean=field(wx, "t2m", iw).mean(0),
            rh_min=field(wx, "rh2", iw).min(0),
            rain_sum=field(wx, "apcp", iw).sum(0),
        )
        ch.update(
            fm1_mean=field(dead, "fm1", idd).mean(0),
            fm10_mean=field(dead, "fm10", idd).mean(0),
            fm100_mean=field(dead, "fm100", idd).mean(0),
        )
        day = gridmet.local_day([ta])[0]
        il = int(
            np.clip(
                np.searchsorted(live_days.values, np.datetime64(day), side="right") - 1, 0, len(live_days) - 1
            )
        )
        ch.update(lfm_herb=field(live, "lfm_herb", [il])[0], lfm_woody=field(live, "lfm_woody", [il])[0])

        # physics
        ph = hrs[:: sc.rothermel_hours]
        acc = {k_: np.zeros((nby, nbx)) for k_ in ("ros", "rx", "ry", "fli")}
        bmax = np.zeros((nby, nbx))
        for t in ph:
            f = spread.fire(t)
            ros = spread.to_window(f["ros_m_min"])
            d = np.deg2rad(spread.to_window(f["direction"]))
            fli = spread.to_window(f["fireline_intensity_kw_m"])
            bm = block_mean(padded(ros), b)
            acc["ros"] += np.nan_to_num(bm)
            acc["rx"] += np.nan_to_num(block_mean(padded(ros * np.sin(d)), b))
            acc["ry"] += np.nan_to_num(block_mean(padded(ros * np.cos(d)), b))
            acc["fli"] += np.nan_to_num(block_mean(padded(fli), b))
            bmax = np.maximum(bmax, np.nan_to_num(bm))
        m = len(ph)
        ch.update(
            ros_mean=acc["ros"] / m,
            ros_x=acc["rx"] / m,
            ros_y=acc["ry"] / m,
            ros_blockmax=bmax,
            fli_mean=acc["fli"] / m,
        )
        local = (pd.Timestamp(ta) - pd.Timedelta(hours=8)).hour + (pd.Timestamp(ta).minute / 60)
        ch.update(
            hours_since_start=np.full((nby, nbx), (ta - t_obs[0]).total_seconds() / 3600),
            step_hours=np.full((nby, nbx), (tb - ta).total_seconds() / 3600),
            local_hour=np.full((nby, nbx), local),
        )
        for i, name in enumerate(CHANNELS):
            X_[k, i] = ch[name]
        Y_[k] = burned[k + 1]
        el = time.time() - t0
        log_.info(
            "  sample %d/%d: %d weather h, %d Rothermel h, %.0f s elapsed, ~%.0f s left",
            k + 1,
            n,
            len(hrs),
            m,
            el,
            el / (k + 1) * (n - k - 1),
        )

    # blocks without fuel data (ocean, outside LANDFIRE): fuel fractions/canopy 0, flagged by `nodata`;
    # any other gap inside the grid is filled with 0 and reported
    vx = X_[:, :, valid]
    gaps = {c: int((~np.isfinite(vx[:, i])).sum()) for i, c in enumerate(CHANNELS)}
    X_[:, :, valid] = np.nan_to_num(vx, nan=0.0)
    X_[:, :, ~valid] = np.nan
    Y_[:, ~valid] = np.nan
    ds = xr.Dataset(
        {
            "inputs": (("sample", "channel", "y", "x"), X_),
            "target": (("sample", "y", "x"), Y_),
            "valid": (("y", "x"), valid),
        },
        coords={
            "channel": CHANNELS,
            "y": ys,
            "x": xs,
            "sample": np.arange(n),
            "t_start": ("sample", io.utc_naive(t_obs[:-1])),
            "t_end": ("sample", io.utc_naive(t_obs[1:])),
        },
    )
    ds.attrs.update(
        title=f"Training samples, fire {fire_id}",
        fireID=fire_id,
        crs=grid.crs,
        block_m=b * grid.res,
        rothermel_hours=sc.rothermel_hours,
        rothermel_stride=sc.rothermel_stride,
        fm100_source=cfg.fuel_moisture.fm100_source,
        conventions="UTC times; x/y block centres",
        comment="inputs: see channel names; target = fraction of each block burned at t_end",
    )
    path = out_path(cfg, fire_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {"inputs": {"chunks": (1, len(CHANNELS), nby, nbx)}, "target": {"chunks": (1, nby, nbx)}}
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False, encoding=enc)

    rep = logs.Report(log_)
    rep(f"Samples, fire {fire_id}")
    rep("=" * 50)
    rep(
        f"{n} samples; {nbx} x {nby} blocks of {b * grid.res:.0f} m ({valid.sum()} inside the grid); "
        f"{len(CHANNELS)} channels"
    )
    filled = {c: g for c, g in gaps.items() if g}
    if filled:
        rep("Filled with 0 (block-samples): " + ", ".join(f"{c} {g}" for c, g in filled.items()))
    rep(
        f"Rothermel every {sc.rothermel_hours} h on every {sc.rothermel_stride}th 30 m cell "
        f"({int(spread.cells.sum())} cells)"
    )
    grow = [(np.nansum(Y_[k] - X_[k, 0]) * (b * grid.res) ** 2 / 1e6) for k in range(n)]
    rep(
        f"Burned area growth per step (km2): min {min(grow):.2f}, median {np.median(grow):.2f}, max {max(grow):.2f}"
    )
    tab = pd.DataFrame(
        {
            c: [np.nanmin(X_[:, i]), np.nanmedian(X_[:, i]), np.nanmax(X_[:, i])]
            for i, c in enumerate(CHANNELS)
        },
        index=["min", "median", "max"],
    ).T
    rep.block(tab.to_string(float_format=lambda x: f"{x:.3g}"))
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_samples_summary.txt")
    log_.info("Wrote %s (%.0f s)", path.relative_to(cfg.root), time.time() - t0)

"""Rothermel surface spread on the fire grid, and a check against observed perimeter advance.

For every 12-hour step between consecutive perimeter observations, the behave port (`rothermel`) is run
each hour on the 30 m cells of a window around the fire (final extent + `roi_buffer_m`):

- fuels and canopy: LANDFIRE FBFM40, CC, CH, CBH (crown ratio = (CH - CBH) / CH) -> wind adjustment factor;
- terrain: slope and aspect from 3DEP;
- weather: HRRR 10 m u/v (bilinear from the 3 km grid, rotated from true north to grid north);
- dead fuel moisture: Nelson 1-h and 10-h, 100-h per `fuel_moisture.fm100_source`;
- live fuel moisture: NFDRS4 herbaceous and woody of the same local day.

HRRR-grid fields are filled over the ocean from the nearest land cell before interpolation.
Per step, the mean and maximum forward spread rate are written to a CF Zarr cube (one time per step,
labelled with the step's end observation). The check compares, on the cells newly burned in each step,
the observed advance rate (distance from the previous perimeter / step length) with the Rothermel
forward rate, using the error measures common in spread-model evaluation (Cruz & Alexander 2013):
mean absolute percent error, mean bias in log space, share within a factor of 2, under/over-prediction.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import rasterio.features
import shapely
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from . import gridmet, io, logs, rothermel
from .config import Config
from .fuel_moisture import surface_dead_moisture
from .live_fuel_moisture import cube_path as live_path
from .perimeters import load_records
from .weather import cube_path as weather_path

log = logging.getLogger(__name__)


def cube_path(cfg: Config, fire_id: int):
    return cfg.fire_dir(fire_id) / "spread" / "rothermel_steps.zarr"


# ------------------------------------------------------------------ helpers
def window(mask: np.ndarray, pad: int) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (
        slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)),
        slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)),
    )


def window_transform(grid: io.Grid, win: tuple[slice, slice]):
    return from_origin(
        grid.bounds[0] + win[1].start * grid.res, grid.bounds[3] - win[0].start * grid.res, grid.res, grid.res
    )


def true_north_azimuth(lon, lat, crs: str) -> np.ndarray:
    """Grid azimuth (deg clockwise from grid north) of true north at each point (meridian convergence)."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    x1, y1 = tr.transform(lon, np.asarray(lat) + 1e-3)
    return np.rad2deg(np.arctan2(np.asarray(x1) - x0, np.asarray(y1) - y0))


class Bilinear:
    """Bilinear weights from a regular (y, x) source grid to target points given in source coordinates."""

    def __init__(self, xs, ys, xt, yt):
        self.idx, self.w = [], []
        fx = self._frac(np.asarray(xs), xt)
        fy = self._frac(np.asarray(ys), yt)
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        x1, y1 = np.minimum(x0 + 1, len(xs) - 1), np.minimum(y0 + 1, len(ys) - 1)
        ax, ay = fx - x0, fy - y0
        for yy, xx, w in (
            (y0, x0, (1 - ay) * (1 - ax)),
            (y0, x1, (1 - ay) * ax),
            (y1, x0, ay * (1 - ax)),
            (y1, x1, ay * ax),
        ):
            self.idx.append((yy, xx))
            self.w.append(w)

    @staticmethod
    def _frac(axis, v):
        i = np.arange(len(axis), dtype=float)
        if axis[0] > axis[-1]:
            axis, i = axis[::-1], i[::-1]
        return np.clip(np.interp(v, axis, i), 0, len(axis) - 1)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        return sum(w * field[yy, xx] for (yy, xx), w in zip(self.idx, self.w))


def nearest_land_index(land: np.ndarray):
    """Index arrays that map every HRRR cell to its nearest land cell (for filling ocean values)."""
    if land is None or land.all() or not land.any():
        return None
    ly, lx = np.nonzero(land)
    yy, xx = np.mgrid[0 : land.shape[0], 0 : land.shape[1]]
    d2 = (yy[..., None] - ly) ** 2 + (xx[..., None] - lx) ** 2  # small HRRR crop: brute force is fine
    k = d2.argmin(-1)
    return ly[k], lx[k]


_NEAREST_CACHE: dict = {}


def fill(field: np.ndarray, nearest=None) -> np.ndarray:
    """Fill ocean cells from the nearest land cell, then any remaining gap from the nearest valid cell."""
    if nearest is not None:
        field = np.where(np.isfinite(field), field, field[nearest])
    ok = np.isfinite(field)
    if ok.all() or not ok.any():
        return field
    key = (ok.shape, ok.tobytes())
    if key not in _NEAREST_CACHE:
        if len(_NEAREST_CACHE) > 64:
            _NEAREST_CACHE.clear()
        _NEAREST_CACHE[key] = nearest_land_index(ok)
    return np.where(ok, field, field[_NEAREST_CACHE[key]])


def error_stats(obs, pred) -> dict:
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(o) & np.isfinite(p) & (o > 0) & (p > 0)
    o, p = o[ok], p[ok]
    if len(o) == 0:
        return dict(
            n=0,
            mape=np.nan,
            mean_log_ratio=np.nan,
            geo_ratio=np.nan,
            within_2x=np.nan,
            under=np.nan,
            over=np.nan,
            spearman=np.nan,
        )
    lr = np.log(p / o)
    rho = pd.Series(o).rank().corr(pd.Series(p).rank()) if len(o) > 2 else np.nan  # Spearman
    return dict(
        n=len(o),
        mape=100 * np.mean(np.abs(p - o) / o),
        mean_log_ratio=lr.mean(),
        geo_ratio=float(np.exp(lr.mean())),
        within_2x=100 * np.mean(np.abs(lr) <= np.log(2)),
        under=100 * np.mean(p < o),
        over=100 * np.mean(p > o),
        spearman=rho,
    )


# --------------------------------------------------------------------- inputs
class SpreadInputs:
    """Surface fire inputs for the cells of a grid window, at any hour of the weather cube."""

    def __init__(self, cfg: Config, fire_id: int, win: tuple[slice, slice], stride: int = 1):
        fdir = cfg.fire_dir(fire_id)
        self.grid = io.Grid.from_json(fdir / "grid.json")
        self.win = win
        sdir = fdir / "static"
        r = lambda k, **kw: io.read_raster(sdir / f"{k}.tif", **kw)[0][win]  # noqa: E731
        fb = r("fbfm40", physical=False).astype(np.int64)
        slope, aspect = r("slope"), r("aspect")
        cc, ch, cbh = r("cc"), r("ch"), r("cbh")
        table = rothermel.fuel_table()
        self.shape = fb.shape
        self.cells = table.burnable(fb) & np.isfinite(slope)  # cells where spread is computed
        if stride > 1:  # every stride-th cell in both directions (coarser sampling for training features)
            sub = np.zeros_like(self.cells)
            sub[stride // 2 :: stride, stride // 2 :: stride] = True
            self.cells &= sub
        sel = self.cells
        self.fm = fb[sel]
        self.slope = np.nan_to_num(slope[sel])
        self.aspect = np.where(aspect[sel] < 0, 0.0, np.nan_to_num(aspect[sel]))
        cc_, ch_, cbh_ = (np.nan_to_num(a[sel]) for a in (cc, ch, cbh))
        self.cc = np.clip(cc_ / 100.0, 0, 1)
        self.ch = np.maximum(ch_, 0.0)
        self.cr = np.where(
            self.ch > 0, np.clip((self.ch - cbh_) / np.where(self.ch > 0, self.ch, 1), 0, 1), 0.0
        )

        # cell centres -> lon/lat -> HRRR Lambert x/y
        g = self.grid
        rows, cols = np.nonzero(sel)
        rows, cols = rows + win[0].start, cols + win[1].start
        xs = g.bounds[0] + (cols + 0.5) * g.res
        ys = g.bounds[3] - (rows + 0.5) * g.res
        lon, lat = Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
        self.north = true_north_azimuth(lon, lat, g.crs)

        self.wx = io.read_cube(weather_path(cfg, fire_id))
        lcc = CRS.from_cf(io.HRRR_GRID_MAPPING)
        hx, hy = Transformer.from_crs("EPSG:4326", lcc, always_xy=True).transform(lon, lat)
        self.interp = Bilinear(self.wx["x"].values, self.wx["y"].values, hx, hy)
        land = self.wx["land"].values.astype(bool) if "land" in self.wx else None
        self.nearest = nearest_land_index(land)
        self.dead = surface_dead_moisture(cfg, fire_id)
        self.live = io.read_cube(live_path(cfg, fire_id))
        self.live_days = gridmet.local_day(self.live["time"].values)
        self.fm100_source = cfg.fuel_moisture.fm100_source
        self.wind_limit = cfg.surface_spread.wind_limit
        self._cache: dict = {}
        for name, ds in (("weather", self.wx), ("dead", self.dead), ("live", self.live)):
            if ds["latitude"].shape != self.wx["latitude"].shape:
                raise ValueError(f"{name} cube is not on the weather cube's HRRR grid")

    def _hrrr(self, ds, var, i) -> np.ndarray:
        key = (id(ds), var, i)
        if key not in self._cache:
            self._cache[key] = self.interp(fill(np.asarray(ds[var].isel(time=i).values, float), self.nearest))
        return self._cache[key]

    def at(self, t: pd.Timestamp) -> dict:
        """Inputs at hour t (naive UTC) for the computed cells; moisture as fractions, wind 10 m m/s."""
        t = pd.Timestamp(t)
        self._cache.clear()
        iw = int(np.searchsorted(self.wx["time"].values, np.datetime64(t)))
        idd = int(np.searchsorted(self.dead["time"].values, np.datetime64(t)))
        day = gridmet.local_day([t])[0]
        il = int(
            np.clip(
                np.searchsorted(self.live_days.values, np.datetime64(day), side="right") - 1,
                0,
                len(self.live_days) - 1,
            )
        )
        for name, ds, i in (("weather", self.wx, iw), ("dead fuel moisture", self.dead, idd)):
            if i >= ds.sizes["time"] or pd.Timestamp(ds["time"].values[i]) != t:
                raise KeyError(f"{t} is not an hour of the {name} cube")
        u, v = self._hrrr(self.wx, "u10", iw), self._hrrr(self.wx, "v10", iw)
        wdir_true = np.mod(np.rad2deg(np.arctan2(-u, -v)), 360.0)
        return dict(
            fuel_model=self.fm,
            m1=self._hrrr(self.dead, "fm1", idd) / 100,
            m10=self._hrrr(self.dead, "fm10", idd) / 100,
            m100=self._hrrr(self.dead, "fm100", idd) / 100,
            m_herb=self._hrrr(self.live, "lfm_herb", il) / 100,
            m_woody=self._hrrr(self.live, "lfm_woody", il) / 100,
            wind_10m_ms=np.hypot(u, v),
            wind_from_deg=np.mod(wdir_true + self.north, 360.0),
            slope_deg=self.slope,
            aspect_deg=self.aspect,
            canopy_cover=self.cc,
            canopy_height_m=self.ch,
            crown_ratio=self.cr,
        )

    def fire(
        self, t, chunk: int = 250_000, keys=("ros_m_min", "direction", "fireline_intensity_kw_m")
    ) -> dict:
        """Rothermel outputs at hour t for the computed cells (spread direction in grid azimuth)."""
        x = self.at(t)
        n = len(self.fm)
        out = {k: np.empty(n, "float32") for k in keys}
        for a in range(0, n, chunk):
            s = slice(a, a + chunk)
            f = rothermel.surface_fire_si(**{k: v[s] for k, v in x.items()}, wind_limit=self.wind_limit)
            for k in keys:
                out[k][s] = f[k]
        return out

    def to_window(self, values, fill_value=np.nan) -> np.ndarray:
        a = np.full(self.shape, fill_value, dtype="float32")
        a[self.cells] = values
        return a


# ------------------------------------------------------------------------ run
def run(cfg: Config, fire_id: int) -> None:
    log_ = logs.setup(cfg.outputs_dir / "logs", "surface_spread", fire_id)
    sc = cfg.surface_spread
    fdir = cfg.fire_dir(fire_id)
    grid = io.Grid.from_json(fdir / "grid.json")
    recs = load_records(cfg.socal_perimeters, fire_id).to_crs(grid.crs)
    union, _ = io.read_raster(fdir / "perimeters" / "perim_union.tif", physical=False)
    win = window(union > 0, int(sc.roi_buffer_m / grid.res))
    inp = SpreadInputs(cfg, fire_id, win)
    h, w = inp.shape
    tr = window_transform(grid, win)
    log_.info(
        "Window %d x %d cells (%d burnable) around fire %d; fm100 source %s",
        w,
        h,
        inp.cells.sum(),
        fire_id,
        inp.fm100_source,
    )

    def raster(geom):
        return rasterio.features.rasterize([(geom, 1)], out_shape=(h, w), transform=tr, dtype="uint8") > 0

    hours = pd.DatetimeIndex(inp.wx["time"].values)
    dead_hours = pd.DatetimeIndex(inp.dead["time"].values)
    t_obs = io.utc_naive(recs["obs_time_utc"])
    rows, cube_written, t0 = [], False, time.time()
    path = cube_path(cfg, fire_id)
    lat_lon = None
    for k in range(len(recs) - 1):
        ta, tb = t_obs[k], t_obs[k + 1]
        step_h = hours[(hours > ta) & (hours <= tb) & hours.isin(dead_hours)][:: sc.hour_stride]
        if len(step_h) == 0:
            log_.warning("Step %d (%s -> %s) has no weather hours; skipped", k, ta, tb)
            continue
        s = np.zeros(inp.cells.sum(), "float64")
        mx = np.zeros(inp.cells.sum(), "float32")
        fmax = np.zeros(inp.cells.sum(), "float32")
        for t in step_h:
            f = inp.fire(t)
            s += f["ros_m_min"]
            mx = np.maximum(mx, f["ros_m_min"])
            fmax = np.maximum(fmax, f["fireline_intensity_kw_m"])
        ros_mean = inp.to_window(s / len(step_h))
        ros_max, fli_max = inp.to_window(mx), inp.to_window(fmax)

        # observed advance on newly burned cells
        prev, cur = raster(recs.geometry.iloc[k]), raster(recs.geometry.iloc[k + 1])
        new = cur & ~prev
        minutes = (tb - ta).total_seconds() / 60.0
        row = dict(
            step=k,
            t_start_utc=io.to_utc([ta])[0],
            t_end_utc=io.to_utc([tb])[0],
            hours=len(step_h),
            new_cells=int(new.sum()),
            new_km2=new.sum() * grid.res**2 / 1e6,
        )
        if new.any() and prev.any():
            ny_, nx_ = np.nonzero(new)
            px, py = tr.c + (nx_ + 0.5) * tr.a, tr.f + (ny_ + 0.5) * tr.e
            d = shapely.distance(shapely.points(px, py), recs.geometry.iloc[k])
            pr = ros_mean[new]
            row.update(
                obs_p90_m_min=np.percentile(d, 90) / minutes,
                obs_max_m_min=d.max() / minutes,
                pred_p90_m_min=np.nanpercentile(pr, 90) if np.isfinite(pr).any() else np.nan,
                pred_max_m_min=np.nanmax(ros_max[new]) if np.isfinite(ros_max[new]).any() else np.nan,
                burnable_share=float(np.isfinite(pr).mean()),
            )
        rows.append(row)

        # cube: one time per step
        tlab = pd.Timestamp(tb).round("min")
        data = {"ros_mean": ros_mean, "ros_max": ros_max, "fli_max": fli_max}
        if not cube_written:
            yy, xx = np.mgrid[0:h, 0:w]
            xs = tr.c + (xx + 0.5) * tr.a
            ys = tr.f + (yy + 0.5) * tr.e
            lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
            lat_lon = (lat, lon)
            ds = io.new_cube([tlab], lat, lon, y=("y", ys[:, 0]), x=("x", xs[0]))
            ds["crs"] = xr.DataArray(np.int32(0), attrs=CRS.from_user_input(grid.crs).to_cf())
            for name, a in data.items():
                ds[name] = (("time", "y", "x"), a[None])
            io.write_cube(
                ds,
                path,
                title=f"Rothermel surface spread per perimeter step, fire {fire_id}",
                source="behave-port Rothermel (firespread.rothermel) on HRRR, Nelson/NFDRS4 moisture, "
                "LANDFIRE fuels, 3DEP terrain",
                time_units="minutes since 1970-01-01 00:00:00",
                chunks=(1, 512, 512),
                fireID=fire_id,
                fm100_source=inp.fm100_source,
                hour_stride=sc.hour_stride,
                wind_limit=str(sc.wind_limit).lower(),
                comment="Each time is the end observation of a step; values summarize the hourly runs "
                "(t_start, t_end]. NaN = not computed (non-burnable or outside the window).",
            )
            cube_written = True
        else:
            xr.Dataset(
                {n: (("time", "y", "x"), a[None]) for n, a in data.items()},
                coords={"time": ("time", io.utc_naive([tlab]))},
            ).to_zarr(path, append_dim="time", zarr_format=3, consolidated=False)
        el = time.time() - t0
        log_.info(
            "  step %d/%d: %d h, %d new cells, %.0f s elapsed, ~%.0f s left",
            k + 1,
            len(recs) - 1,
            len(step_h),
            row["new_cells"],
            el,
            el / (k + 1) * (len(recs) - 2 - k),
        )

    df = pd.DataFrame(rows)
    io.write_table(df, cfg.outputs_dir / f"fire_{fire_id}_surface_spread_steps.csv")
    _report(cfg, fire_id, df, inp, log_)
    if lat_lon is not None:
        _plot(cfg, fire_id, df, path, recs, tr, (h, w), raster)


def _report(cfg, fire_id, df, inp, log_):
    sc = cfg.surface_spread
    rep = logs.Report(log_)
    fmt = lambda x: f"{x:.2f}"  # noqa: E731
    rep(f"Rothermel surface spread vs observed advance, fire {fire_id}")
    rep("=" * 60)
    rep(
        f"Window {inp.shape[1]} x {inp.shape[0]} cells at {inp.grid.res:.0f} m, {int(inp.cells.sum())} burnable; "
        f"{len(df)} steps; hourly runs (stride {sc.hour_stride} h); 100-h moisture: {inp.fm100_source}; "
        f"wind limit {'on' if sc.wind_limit else 'off'}"
    )
    rep("Observed: distance of newly burned cells from the previous perimeter / step length (p90).")
    rep("Predicted: Rothermel forward spread rate, mean over the step's hours, p90 over the same cells.\n")
    ok = df[df["new_cells"] >= sc.min_new_cells] if "obs_p90_m_min" in df else df.iloc[:0]
    st = error_stats(ok.get("obs_p90_m_min", []), ok.get("pred_p90_m_min", []))
    rep(f"Steps with >= {sc.min_new_cells} new cells: {st['n']}")
    if st["n"]:
        rep(
            f"  MAPE {st['mape']:.0f} %   geometric mean ratio pred/obs {st['geo_ratio']:.2f}   "
            f"within 2x {st['within_2x']:.0f} %"
        )
        rep(
            f"  under-predicted {st['under']:.0f} %   over-predicted {st['over']:.0f} %   "
            f"Spearman r {st['spearman']:.2f}"
        )
    cols = [
        c
        for c in (
            "step",
            "t_end_utc",
            "hours",
            "new_km2",
            "obs_p90_m_min",
            "pred_p90_m_min",
            "obs_max_m_min",
            "pred_max_m_min",
            "burnable_share",
        )
        if c in df
    ]
    rep("\nBy step (m/min)\n" + "-" * 60)
    rep.block(df[cols].to_string(index=False, float_format=fmt))
    rep("\nNotes")
    rep(
        "- Largefire perimeters come from 375 m VIIRS detections; advance rates below ~0.5 m/min are not resolved."
    )
    rep(
        "- Rothermel gives the head-fire rate; the observed p90 advance mixes head, flank and backing spread and"
    )
    rep("  includes spotting, so over-prediction on slow steps and under-prediction on spotting-driven runs")
    rep("  are expected. Crown fire is not modelled (chaparral is a surface fuel in Rothermel).")
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_surface_spread_summary.txt")


def _plot(cfg, fire_id, df, path, recs, tr, shape, raster):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    if "obs_p90_m_min" in df:
        ax.semilogy(
            df["t_end_utc"], df["obs_p90_m_min"], "o-", color="black", ms=3, label="observed advance (p90)"
        )
        ax.semilogy(
            df["t_end_utc"],
            df["pred_p90_m_min"],
            "s-",
            color="tab:red",
            ms=3,
            label="Rothermel forward rate (p90, step mean)",
        )
    ax.set_ylabel("m/min")
    ax.set_title(f"Fire {fire_id}: spread rate per 12-h step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    k = int(df["new_cells"].idxmax())
    cube = io.read_cube(path)
    a = cube["ros_max"].isel(time=k).values
    im = ax.imshow(a, cmap="inferno", vmin=0, vmax=np.nanpercentile(a, 99) if np.isfinite(a).any() else 1)
    step = int(df.loc[k, "step"])
    for j, col in ((step, "cyan"), (step + 1, "lime")):
        m = raster(recs.geometry.iloc[j])
        ax.contour(m.astype(float), levels=[0.5], colors=col, linewidths=0.8)
    ax.set_title(
        f"Max Rothermel ROS, largest step (ends {df.loc[k, 't_end_utc']:%Y-%m-%d %H:%M} UTC)\n"
        "cyan = start perimeter, green = end",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="m/min", shrink=0.8)
    fig.tight_layout()
    fig.savefig(cfg.outputs_dir / f"fire_{fire_id}_surface_spread.png", dpi=150)
    plt.close(fig)

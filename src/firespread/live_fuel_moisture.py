"""Daily live herbaceous and woody fuel moisture for one fire.

Model: the Growing Season Index (GSI) live fuel moisture model of NFDRS
version 4 (Jolly et al. 2005), run through the USDA Forest Service Missoula
Fire Sciences Lab's NFDRS4 library exactly as NFDRS4 runs it on daily data:
NFDRS4.Init(...) once, then NFDRS4.UpdateDaily(...) each day with the
observation-time temperature and RH, daily min/max temperature and min RH,
24-h precipitation and wind, reading MCHERB, MCWOOD and m_GSI.

Inputs: gridMET daily weather (~4 km): tmmx, tmmn, rmin, pr, vs, plus fm100 and
fm1000 (passed for NFDRS4's index calculations only; they do not affect live
moisture). The GSI averages over 21 days and herbs reset every January, so the
model starts `live_fuel_moisture.spinup_days` (default 365) before the fire.
It runs once per gridMET cell under the fire's HRRR land cells; values are
mapped to the HRRR grid.

Output: data/processed/fires/<id>/fuel_moisture/live_fuel_moisture.zarr
        (daily at the NFDRS observation hour, ~13:00 PST)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import gridmet, io, logs
from .config import Config
from .weather import cube_path as weather_path

log = logging.getLogger(__name__)

FUEL_MODEL = "X"  # NFDRS fuel model; affects the indices only, not live moisture
SLOPE_CLASS = 1
KBDI_THRESHOLD = 100
DAILY_VARS = ("tmmx", "tmmn", "rmin", "pr", "vs", "fm100", "fm1000")


def cube_path(cfg: Config, fire_id: int) -> Path:
    return cfg.fire_dir(fire_id) / "fuel_moisture" / "live_fuel_moisture.zarr"


def k_to_f(k):
    return (np.asarray(k) - 273.15) * 9.0 / 5.0 + 32.0


def run_cell(nfdrs4, lat: float, days: pd.DatetimeIndex, d: dict[str, np.ndarray], herbs_annual: bool):
    """Run NFDRS4's daily live fuel moisture for one location.
    d: daily series (tmmx K, tmmn K, rmin %, pr mm, vs m/s, fm100 %, fm1000 %).
    Returns herb %, woody % and GSI arrays."""
    n = len(days)
    avg_precip_in = float(np.nansum(d["pr"]) / 25.4 * 365.25 / n)
    m = nfdrs4.NFDRS4()
    m.Init(
        float(lat),
        FUEL_MODEL,
        SLOPE_CLASS,
        avg_precip_in,
        False,
        True,
        herbs_annual,
        KBDI_THRESHOLD,
        13,
        False,
    )
    tmax, tmin = k_to_f(d["tmmx"]), k_to_f(d["tmmn"])
    herb, wood, gsi = np.empty(n), np.empty(n), np.empty(n)
    for k, day in enumerate(days):
        m.UpdateDaily(
            int(day.year),
            int(day.month),
            int(day.day),
            int(day.day_of_year),
            float(tmax[k]),
            float(tmin[k]),
            float(tmax[k]),  # obs-time temp ~ daily max (afternoon)
            float(d["rmin"][k]),
            float(d["rmin"][k]),  # obs-time RH ~ daily min (afternoon)
            float(d["pr"][k]) / 25.4,  # inches
            float(d["vs"][k]) * 2.23694,  # mph
            float(d["fm100"][k]),
            float(d["fm100"][k]),
            float(d["fm100"][k]),
            float(d["fm1000"][k]),
            float((tmax[k] - 32.0) * 5.0 / 9.0),
            False,
        )
        herb[k], wood[k], gsi[k] = m.MCHERB, m.MCWOOD, m.m_GSI
    return herb, wood, gsi


def _fill_gaps(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a)
    return s.ffill().bfill().to_numpy()


def run(cfg: Config, fire_id: int) -> Path:
    log_ = logs.setup(cfg.outputs_dir / "logs", "live_fuel_moisture", fire_id)
    try:
        import nfdrs4
    except ImportError as e:
        raise RuntimeError("Package nfdrs4 is not installed; see README (Fire Lab NFDRS4)") from e
    lc = cfg.live_fuel_moisture
    fdir = cfg.fire_dir(fire_id)
    wx = io.read_cube(weather_path(cfg, fire_id))
    lat, lon = wx["latitude"].values, wx["longitude"].values
    land = wx["land"].values.astype(bool) if "land" in wx else np.ones(lat.shape, bool)
    obs = io.read_table(fdir / "perimeters" / "obs_times.csv", ["obs_time_utc"])
    t_obs = io.to_utc(obs["obs_time_utc"])
    first_day = gridmet.local_day(t_obs[:1].tz_convert(None))[0]
    last_day = gridmet.local_day(t_obs[-1:].tz_convert(None))[0]
    days = pd.date_range(first_day - pd.Timedelta(days=lc.spinup_days), last_day, freq="D")
    log_.info(
        "Live fuel moisture (NFDRS4 GSI): %d days %s -> %s (fire from %s)",
        len(days),
        days[0].date(),
        days[-1].date(),
        first_day.date(),
    )

    bbox = (lon.min() - 0.05, lat.min() - 0.05, lon.max() + 0.05, lat.max() + 0.05)
    years = range(days[0].year, days[-1].year + 1)
    g = {}
    for key in DAILY_VARS:
        da = gridmet.fetch(cfg, key, years, bbox).sel(day=slice(days[0], days[-1]))
        if da.sizes["day"] != len(days):
            raise RuntimeError(f"gridMET {key}: {da.sizes['day']} days available, {len(days)} needed")
        g[key] = da

    # one model run per gridMET cell under the HRRR land cells
    iy, ix = gridmet.nearest_index(g["tmmx"], lat, lon)
    pairs = sorted({(int(a), int(b)) for a, b, ok in zip(iy.ravel(), ix.ravel(), land.ravel()) if ok})
    glat = g["tmmx"]["lat"].values
    log_.info("Running %d gridMET cells under %d HRRR land cells", len(pairs), int(land.sum()))
    vals = {key: g[key].values for key in DAILY_VARS}
    res = {}
    t0 = time.time()
    for n, (a, b) in enumerate(pairs, 1):
        series = {key: _fill_gaps(vals[key][:, a, b].astype(float)) for key in DAILY_VARS}
        if not np.isfinite(series["tmmx"]).any():
            continue  # gridMET has no data here (ocean)
        res[(a, b)] = run_cell(nfdrs4, glat[a], days, series, lc.herbs_annual)
        if n % 200 == 0 or n == len(pairs):
            log_.info("  %d/%d cells, %.0f s", n, len(pairs), time.time() - t0)

    shape = (len(days), *lat.shape)
    herb, wood, gsi = (np.full(shape, np.nan, dtype="float32") for _ in range(3))
    for yy, xx in zip(*np.nonzero(land)):
        r = res.get((int(iy[yy, xx]), int(ix[yy, xx])))
        if r is not None:
            herb[:, yy, xx], wood[:, yy, xx], gsi[:, yy, xx] = r

    times = days + pd.Timedelta(hours=lc.obs_hour_utc)  # local (PST) day -> NFDRS obs hour in UTC
    ds = io.new_cube(times, lat, lon)
    ds["lfm_herb"] = (("time", "y", "x"), herb)
    ds["lfm_woody"] = (("time", "y", "x"), wood)
    ds["gsi"] = (("time", "y", "x"), gsi)
    ds["land"] = (("y", "x"), land.astype("int8"))
    out = io.write_cube(
        ds,
        cube_path(cfg, fire_id),
        title=f"Live fuel moisture, fire {fire_id}",
        source="NFDRS4 GSI live fuel moisture (Jolly et al. 2005; USFS Missoula Fire Sciences Lab), "
        "driven by gridMET daily weather (Abatzoglou 2013)",
        fireID=fire_id,
        nfdrs4_version=getattr(nfdrs4, "__version__", "unknown"),
        spinup_days=lc.spinup_days,
        fire_start_day=str(first_day.date()),
        herbs_annual=str(lc.herbs_annual).lower(),
        comment="Daily values at the NFDRS observation hour (~13:00 PST). Herbaceous and woody limits are "
        "NFDRS4 defaults (herb 30-250%, woody 60-200%).",
    )
    log_.info("Wrote %s", out.relative_to(cfg.root))

    # summary
    rep = logs.Report(log_)
    fire = days >= first_day
    rep(f"Live fuel moisture, fire {fire_id}")
    rep("=" * 50)
    rep(
        f"Model: NFDRS4 GSI ({getattr(nfdrs4, '__version__', '')}); gridMET daily weather; spin-up "
        f"{lc.spinup_days} days from {days[0].date()}; herbs annual: {lc.herbs_annual}"
    )
    rep(f"gridMET cells modeled: {len(res)} (under {int(land.sum())} HRRR land cells)\n")
    rep("Land cells, fire period (%; GSI 0-1)\n" + "-" * 50)
    rows = []
    for name, arr in (("herbaceous", herb), ("woody", wood), ("GSI", gsi)):
        v = arr[fire][:, land]
        rows.append(
            dict(
                variable=name,
                min=np.nanmin(v),
                p10=np.nanpercentile(v, 10),
                median=np.nanmedian(v),
                p90=np.nanpercentile(v, 90),
                max=np.nanmax(v),
            )
        )
    rep.block(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    rec = io.read_table(fdir / "perimeters" / "obs_times.csv")
    clat, clon = float(rec["clat"].iloc[0]), float(rec["clon"].iloc[0])
    d2 = (lat - clat) ** 2 + ((lon - clon) * np.cos(np.radians(clat))) ** 2
    i0, j0 = np.unravel_index(np.argmin(np.where(land, d2, np.inf)), d2.shape)
    rep(
        f"\nNearest land cell to ignition ({lat[i0, j0]:.3f}, {lon[i0, j0]:.3f}), monthly means before the fire\n"
        + "-" * 50
    )
    ser = pd.DataFrame({"herb": herb[:, i0, j0], "woody": wood[:, i0, j0], "GSI": gsi[:, i0, j0]}, index=days)
    rep.block(ser.resample("MS").mean().tail(13).to_string(float_format=lambda x: f"{x:.2f}"))
    k = int(np.searchsorted(days, first_day))
    rep(
        f"\nOn the first fire day ({first_day.date()}): herb {herb[k, i0, j0]:.0f}%, "
        f"woody {wood[k, i0, j0]:.0f}%, GSI {gsi[k, i0, j0]:.2f}"
    )
    rep("\nNotes")
    rep(
        "- Values at the NFDRS limits (herb 30%, woody 60%) mean the model considers the vegetation dormant/cured."
    )
    rep("- Check against FEMS field samples (chamise) before using these in the spread model.")
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_live_fuel_moisture_summary.txt")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax[0].plot(days, ser["herb"], label="herbaceous", color="tab:olive")
    ax[0].plot(days, ser["woody"], label="woody", color="tab:brown")
    ax[0].set_ylabel("live fuel moisture (%)")
    ax[0].legend(loc="upper right")
    ax[1].plot(days, ser["GSI"], color="tab:green")
    ax[1].set_ylabel("GSI (0-1)")
    for a in ax:
        a.axvspan(first_day, days[-1], color="red", alpha=0.12, label="fire")
    ax[0].set_title(
        f"Fire {fire_id}: NFDRS4 live fuel moisture at the land cell nearest ignition "
        f"({lat[i0, j0]:.3f}, {lon[i0, j0]:.3f}); red = fire period"
    )
    fig.tight_layout()
    fig.savefig(cfg.outputs_dir / f"fire_{fire_id}_live_fuel_moisture_timeseries.png", dpi=150)
    plt.close(fig)
    return out

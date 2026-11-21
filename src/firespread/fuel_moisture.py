"""Hourly dead fuel moisture for one fire (formerly script 10).

Model: Nelson (2000) dead fuel moisture, as implemented in the USDA Forest
Service Missoula Fire Sciences Lab's NFDRS4 library (package `nfdrs4`), the
model behind NFDRS version 4. Inputs per HRRR cell and hour come from the
weather cube: 2 m temperature and RH (analyses), solar radiation and 1-h rain
(1-h forecasts). Each stick starts at the gridMET 100-h (1/10/100-h sticks) or
1000-h value on the first spin-up day, then is updated hourly exactly as NFDRS4
drives it: update(elapsed 1 h, T degC, RH g/g, solar W/m2, rain cm/h,
0.0218 cal/cm3, amount=True) and read medianRadialMoisture().
gridMET fm100/fm1000 are kept as an independent daily check.

Output: data/processed/fires/<id>/fuel_moisture/dead_fuel_moisture.zarr
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import http, io, logs, manifest
from .config import Config
from .weather import cube_path as weather_path

log = logging.getLogger(__name__)

BP = 0.0218  # stick barometric pressure (cal/cm3), NFDRS4 default
GRIDMET_VARS = {"fm100": "dead_fuel_moisture_100hr", "fm1000": "dead_fuel_moisture_1000hr"}
STICK = {1: "1", 10: "10", 100: "100", 1000: "1000"}


def cube_path(cfg: Config, fire_id: int) -> Path:
    return cfg.fire_dir(fire_id) / "fuel_moisture" / "dead_fuel_moisture.zarr"


# ------------------------------------------------------------------- gridMET
def gridmet_fetch(cfg: Config, key: str, years, bbox, t0, t1) -> xr.DataArray:
    cache = cfg.raw_dir / "gridmet"
    cache.mkdir(parents=True, exist_ok=True)
    lonmin, latmin, lonmax, latmax = bbox
    frames = []
    for y in years:
        f = cache / f"{key}_{y}_{latmin:.2f}_{latmax:.2f}_{lonmin:.2f}_{lonmax:.2f}.nc"
        if not f.exists():
            params = {
                "var": GRIDMET_VARS[key],
                "north": latmax,
                "south": latmin,
                "west": lonmin,
                "east": lonmax,
                "horizStride": 1,
                "time_start": f"{y}-01-01T00:00:00Z",
                "time_end": f"{y}-12-31T23:59:59Z",
                "accept": "netcdf",
            }
            last = None
            for tmpl in cfg.fuel_moisture.gridmet_urls:
                url = tmpl.format(var=key, year=y)
                try:
                    log.info("  downloading gridMET %s %s (%s)", key, y, url.split("/")[2])
                    r = http.get(url, params=params, timeout=300)
                    if r.content[:3] in (b"CDF", b"\x89HD"):
                        f.write_bytes(r.content)
                        manifest.record(
                            cfg.data_dir,
                            source="gridMET (Abatzoglou 2013) NCSS",
                            url=r.url,
                            path=f,
                            step="fuel_moisture",
                        )
                        break
                    last = f"not NetCDF: {r.content[:120]!r}"
                except Exception as e:  # try the next mirror
                    last = str(e)
                log.warning("    failed: %s", last)
            else:
                raise RuntimeError(f"gridMET {key} {y} could not be downloaded: {last}")
        with xr.open_dataset(f) as ds:
            name = GRIDMET_VARS[key] if GRIDMET_VARS[key] in ds else next(iter(ds.data_vars))
            da = ds[name].load()
        tdim = next(d for d in da.dims if d not in ("lat", "lon"))
        frames.append(da.rename({tdim: "day"}))
    da = xr.concat(frames, dim="day").sortby("day")
    return da.sel(day=slice(t0.tz_convert(None).normalize(), t1.tz_convert(None).normalize()))


def gridmet_to_cells(da: xr.DataArray, lat, lon) -> np.ndarray:
    glat, glon = da["lat"].values, da["lon"].values
    iy = np.abs(glat[:, None] - lat.ravel()[None, :]).argmin(axis=0)
    ix = np.abs(glon[:, None] - lon.ravel()[None, :]).argmin(axis=0)
    return da.values[:, iy, ix].reshape((da.sizes["day"], *lat.shape)).astype(float)


# --------------------------------------------------------------------- Nelson
def make_stick(nfdrs4, cls: int):
    D = nfdrs4.DeadFuelMoisture
    s = getattr(D, f"createDeadFuelMoisture{STICK[cls]}")()
    getattr(s, f"initDeadFuelMoisture{STICK[cls]}")()
    return s


def run_nelson(nfdrs4, cls, T, RH, SW, P_cm, w0) -> tuple[np.ndarray, int]:
    """T degC, RH fraction, SW W/m2, P cm per hour; arrays (nt,). Returns % and rejected updates."""
    s = make_stick(nfdrs4, cls)
    s.initializeEnvironment(
        float(T[0]), float(RH[0]), float(SW[0]), 0.0, float(T[0]), float(RH[0]), float(w0)
    )
    out = np.empty(len(T))
    out[0] = w0 * 100.0
    bad = 0
    for k in range(1, len(T)):
        ok = s.update(1.0, float(T[k]), float(RH[k]), float(SW[k]), float(P_cm[k]), BP, True)
        bad += not ok
        out[k] = s.medianRadialMoisture() * 100.0
    return out, bad


def ignition_cell(fire_dir: Path, lat, lon):
    rec = io.read_table(fire_dir / "perimeters" / "obs_times.csv")
    clat, clon = float(rec["clat"].iloc[0]), float(rec["clon"].iloc[0])
    d = (lat - clat) ** 2 + ((lon - clon) * np.cos(np.radians(clat))) ** 2
    iy, ix = np.unravel_index(np.argmin(d), d.shape)
    return int(iy), int(ix), clat, clon


# ---------------------------------------------------------------------- main
def run(cfg: Config, fire_id: int, max_cells: int = 0, all_cells: bool = False) -> Path:
    log_ = logs.setup(cfg.outputs_dir / "logs", "fuel_moisture", fire_id)
    try:
        import nfdrs4
    except ImportError as e:
        raise RuntimeError("Package nfdrs4 is not installed; see README (Fire Lab NFDRS4)") from e
    fm = cfg.fuel_moisture
    fdir = cfg.fire_dir(fire_id)
    wx = io.read_cube(weather_path(cfg, fire_id)).load()
    lat, lon = wx["latitude"].values, wx["longitude"].values
    ny, nx = lat.shape
    full = pd.DatetimeIndex(wx["time"].values)
    start = pd.Timestamp(wx.attrs["analysis_start_utc"]).tz_convert(None)
    keep = full >= start
    land = wx["land"].values.astype(bool) if "land" in wx else None
    T = np.clip(wx["t2m"].values, -59.9, 59.9)  # NFDRS4 rejects values outside its ranges
    RH = np.clip(wx["rh2"].values / 100.0, 0.001, 1.0)
    SW = np.clip(wx["dswrf"].values, 0.0, 1999.0)
    P_cm = np.clip(wx["apcp"].values, 0.0, None) / 10.0
    spin_h = int((~keep).sum())
    log_.info(
        "Weather cube: %d hours (%d spin-up) %s -> %s; grid %d x %d",
        len(full),
        spin_h,
        full[0],
        full[-1],
        ny,
        nx,
    )

    # gridMET
    bbox = (lon.min() - 0.05, lat.min() - 0.05, lon.max() + 0.05, lat.max() + 0.05)
    t0u, t1u = io.to_utc([full[0]])[0], io.to_utc([full[-1]])[0]
    years = sorted({t0u.year, t1u.year})
    g = {}
    for key in ("fm100", "fm1000"):
        da = gridmet_fetch(cfg, key, years, bbox, t0u, t1u)
        g[key] = (pd.DatetimeIndex(da["day"].values), gridmet_to_cells(da, lat, lon))
        log_.info(
            "  gridMET %s: %d days, %.1f-%.1f %%",
            key,
            g[key][1].shape[0],
            np.nanmin(g[key][1]),
            np.nanmax(g[key][1]),
        )

    def start_value(key):
        days, v = g[key]
        i = max(int(np.searchsorted(days, full[0].normalize(), side="right")) - 1, 0)
        w = v[i] / 100.0
        return np.where(np.isfinite(w), w, 0.15)

    w0 = {
        1: start_value("fm100"),
        10: start_value("fm100"),
        100: start_value("fm100"),
        1000: start_value("fm1000"),
    }

    # Nelson
    iy0, ix0, clat, clon = ignition_cell(fdir, lat, lon)
    cells = [(iy, ix) for iy in range(ny) for ix in range(nx) if all_cells or land is None or land[iy, ix]]
    if land is not None and not all_cells:
        log_.info("Skipping %d ocean cells (HRRR land mask)", ny * nx - len(cells))
    if max_cells:
        cells = sorted(cells, key=lambda c: (c[0] - iy0) ** 2 + (c[1] - ix0) ** 2)[:max_cells]
    res = {c: np.full((len(full), ny, nx), np.nan, dtype="float32") for c in fm.classes}
    bad_total, t_start = 0, time.time()
    log_.info("Nelson model: %d cells x %d classes x %d hours", len(cells), len(fm.classes), len(full))
    for n, (iy, ix) in enumerate(cells, 1):
        for c in fm.classes:
            out, bad = run_nelson(
                nfdrs4, c, T[:, iy, ix], RH[:, iy, ix], SW[:, iy, ix], P_cm[:, iy, ix], w0[c][iy, ix]
            )
            res[c][:, iy, ix] = out
            bad_total += bad
        if n in (5, 50) or n % 200 == 0 or n == len(cells):
            el = time.time() - t_start
            log_.info(
                "  %d/%d cells, %.0f s elapsed, ~%.0f s left", n, len(cells), el, el / n * (len(cells) - n)
            )

    # write
    tk = full[keep]
    ds = io.new_cube(tk, lat, lon, day=("day", pd.DatetimeIndex(g["fm100"][0])))
    for c in fm.classes:
        ds[f"fm{c}"] = (("time", "y", "x"), res[c][keep])
    for key in ("fm100", "fm1000"):
        ds[f"{key}_gridmet"] = (("day", "y", "x"), g[key][1].astype("float32"))
    if land is not None:
        ds["land"] = (("y", "x"), land.astype("int8"))
    out_path = cube_path(cfg, fire_id).with_suffix(".test.zarr") if max_cells else cube_path(cfg, fire_id)
    io.write_cube(
        ds,
        out_path,
        title=f"Dead fuel moisture, fire {fire_id}",
        source="Nelson (2000) model via NFDRS4 (USFS Missoula Fire Sciences Lab), driven by HRRR; "
        "gridMET fm100/fm1000 (Abatzoglou 2013)",
        fireID=fire_id,
        nfdrs4_version=getattr(nfdrs4, "__version__", "unknown"),
        spinup_hours=spin_h,
        cells_modeled=len(cells),
        initialization="gridMET fm100 (1/10/100-h) and fm1000 (1000-h) at spin-up start",
    )
    log_.info("Wrote %s", out_path.relative_to(cfg.root))

    # summary
    rep = logs.Report(log_)
    mask = (np.ones((ny, nx), bool) if land is None else land) & np.isfinite(res[fm.classes[0]][-1])
    rep(f"Dead fuel moisture, fire {fire_id}")
    rep("=" * 50)
    rep(
        f"Model: Nelson (2000) via NFDRS4 {getattr(nfdrs4, '__version__', '')}; hourly; {len(cells)} HRRR cells; "
        f"spin-up {spin_h} h from {full[0]} UTC"
    )
    rep(
        f"Gap-filled weather hours in the run: analyses {int(wx['qc_filled_f00'].sum())}, "
        f"solar/rain {int(wx['qc_filled_f01'].sum())} (rain set to 0)"
    )
    rep(f"Rejected NFDRS4 updates: {bad_total}")
    rep(f"Land cells summarized: {int(mask.sum())} of {ny * nx}\n")
    rep("Moisture over land cells, analysis period (%)\n" + "-" * 50)
    rows = []
    for c in fm.classes:
        v = res[c][keep][:, mask]
        rows.append(
            dict(
                cls=f"{c}-h",
                min=np.nanmin(v),
                p10=np.nanpercentile(v, 10),
                median=np.nanmedian(v),
                p90=np.nanpercentile(v, 90),
                max=np.nanmax(v),
            )
        )
    rep.block(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    rep(
        f"\nNelson vs gridMET (Nelson at {fm.afternoon_utc_hour}Z ~ 13:00 PST vs gridMET same day)\n"
        + "-" * 50
    )
    rows = []
    for c, key in ((100, "fm100"), (1000, "fm1000")):
        if c not in fm.classes:
            continue
        days, gv = g[key]
        sel = (full.hour == fm.afternoon_utc_hour) & keep
        gi = np.array([days.get_loc(d) if d in days else -1 for d in full[sel].normalize()])
        ok = gi >= 0
        a = res[c][sel][ok][:, mask].ravel()
        b = gv[gi[ok]][:, mask].ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 2:
            rows.append(
                dict(
                    cls=f"{c}-h",
                    n=int(m.sum()),
                    nelson=a[m].mean(),
                    gridmet=b[m].mean(),
                    bias=(a[m] - b[m]).mean(),
                    rmse=np.sqrt(((a[m] - b[m]) ** 2).mean()),
                    r=np.corrcoef(a[m], b[m])[0, 1],
                )
            )
    if rows:
        rep.block(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    obs_t = io.read_table(fdir / "perimeters" / "obs_times.csv", ["obs_time_utc"])[
        "obs_time_utc"
    ].dt.tz_convert(None)
    rep(
        f"\nCell nearest ignition ({lat[iy0, ix0]:.3f}, {lon[iy0, ix0]:.3f}) at the first 8 observations\n"
        + "-" * 50
    )
    rows = []
    for t in list(obs_t)[:8]:
        k = int(np.argmin(np.abs(full - t)))
        d = {"obs_time_utc": t.strftime("%m-%d %H:%M"), "T_C": T[k, iy0, ix0], "RH_%": RH[k, iy0, ix0] * 100}
        d.update({f"fm{c}": res[c][k, iy0, ix0] for c in fm.classes})
        rows.append(d)
    rep.block(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    rep("\nNotes")
    rep("- 1-h and 10-h respond within hours; 100-h and 1000-h carry their gridMET start value for weeks.")
    rep("- gridMET values are daily and represent afternoon conditions.")
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_fuel_moisture_summary.txt")

    _plots(cfg, fire_id, full, keep, res, g, wx, land, (iy0, ix0, clat, clon), obs_t)
    return out_path


def _plots(cfg, fire_id, full, keep, res, g, wx, land, ign, obs_t):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fm = cfg.fuel_moisture
    iy, ix, clat, clon = ign
    lat, lon = wx["latitude"].values, wx["longitude"].values
    tk = full[keep]
    cols = {1: "tab:red", 10: "tab:orange", 100: "tab:green", 1000: "tab:blue"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for c in fm.classes:
        axes[0].plot(tk, res[c][keep][:, iy, ix], lw=1, color=cols[c], label=f"{c}-h Nelson")
    for key, c in (("fm100", 100), ("fm1000", 1000)):
        days, gv = g[key]
        sel = days >= tk.min().normalize()
        axes[0].plot(
            days[sel] + pd.Timedelta(hours=fm.afternoon_utc_hour),
            gv[sel, iy, ix],
            "o",
            ms=3,
            color=cols[c],
            label=f"{c}-h gridMET",
        )
    axes[0].set_ylabel("dead fuel moisture (%)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(tk, wx["rh2"].values[keep][:, iy, ix], color="tab:green", lw=0.9)
    axes[1].set_ylabel("RH 2 m (%)")
    axes[2].bar(tk, wx["apcp"].values[keep][:, iy, ix], width=1 / 24, color="tab:blue")
    axes[2].set_ylabel("rain (mm/h)")
    ax3 = axes[2].twinx()
    ax3.plot(tk, wx["dswrf"].values[keep][:, iy, ix], color="goldenrod", lw=0.7)
    ax3.set_ylabel("solar (W/m2)")
    for ax in axes:
        for t in obs_t:
            ax.axvline(t, color="grey", lw=0.3, alpha=0.5)
    axes[0].set_title(
        f"Fire {fire_id}: Nelson dead fuel moisture at the HRRR cell nearest ignition "
        f"({lat[iy, ix]:.3f}, {lon[iy, ix]:.3f}); grey = VIIRS times (UTC)"
    )
    fig.tight_layout()
    fig.savefig(cfg.outputs_dir / f"fire_{fire_id}_fuel_moisture_timeseries.png", dpi=150)
    plt.close(fig)

    k = int(np.argmin(np.abs(full - obs_t.iloc[0])))
    show = [c for c in (1, 10) if c in fm.classes]
    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 5), squeeze=False)
    for ax, c in zip(axes[0], show):
        v = res[c][k].copy()
        if land is not None:
            v[~land] = np.nan
        pm = ax.pcolormesh(lon, lat, v, cmap="YlOrBr_r", shading="nearest")
        fig.colorbar(pm, ax=ax, label="%")
        ax.plot(clon, clat, "k*", ms=12, label="first perimeter centroid")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"{c}-h fuel moisture, {full[k]:%Y-%m-%d %H}Z (first observation)")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    fig.tight_layout()
    fig.savefig(cfg.outputs_dir / f"fire_{fire_id}_fuel_moisture_map.png", dpi=150)
    plt.close(fig)

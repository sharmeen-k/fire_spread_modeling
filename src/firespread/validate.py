"""Check every non-excluded fire's pipeline outputs (`firespread validate`).

Per fire: perimeters, terrain, fuels, HRRR weather cube, dead and live fuel moisture, samples (including
whether the physics channels are all zero). Prints a table and a problem list and writes
outputs/validation_report.json. Returns the problem list.
"""

import json

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .perimeters import load_records


def share_nan(a):
    a = np.asarray(a, float)
    return float(np.isnan(a).mean()) if a.size else 1.0


def run(cfg: Config) -> list[str]:
    split = pd.read_csv(cfg.outputs_dir / "split.csv")
    fires = split[split["split"] != "excluded"]
    problems, rows = [], []

    def flag(fid, msg):
        problems.append(f"{fid}: {msg}")

    for r in fires.itertuples():
        fid, d = int(r.fireID), cfg.fire_dir(int(r.fireID))
        row = dict(fire=fid, split=r.split)
        try:
            # ------------------------------------------------ perimeters
            recs = load_records(cfg.socal_perimeters, fid)
            row["records"] = len(recs)
            ot = d / "perimeters" / "obs_times.csv"
            if not ot.exists():
                flag(fid, "no perimeters/obs_times.csv (stack not run)")
            else:
                n = len(io.read_table(ot))
                if n != len(recs):
                    flag(fid, f"obs_times.csv has {n} rows, catalog {len(recs)}")
            t_obs = io.utc_naive(recs["obs_time_utc"])
            if (np.diff(t_obs.values).astype("timedelta64[h]").astype(int) <= 0).any():
                flag(fid, "perimeter times not increasing")
            area = recs.to_crs(3310).area.values
            if (np.diff(area) < -0.01 * area[:-1]).any():
                flag(fid, "perimeter area shrinks between records (>1%)")

            # ------------------------------------------------ static layers
            sdir = d / "static"
            for k in ("dem", "slope", "aspect", "fbfm40", "cc", "ch", "cbh", "cbd"):
                if not (sdir / f"{k}.tif").exists():
                    flag(fid, f"static/{k}.tif missing")
            if (sdir / "dem.tif").exists():
                dem, _ = io.read_raster(sdir / "dem.tif")
                row["dem_nan"] = share_nan(dem)
                if np.nanmax(dem) > 4500 or np.nanmin(dem) < -100:
                    flag(fid, f"DEM range {np.nanmin(dem):.0f}..{np.nanmax(dem):.0f} m looks wrong")
            if (sdir / "fbfm40.tif").exists() and (d / "perimeters" / "perim_union.tif").exists():
                fb, _ = io.read_raster(sdir / "fbfm40.tif", physical=False)
                un, _ = io.read_raster(d / "perimeters" / "perim_union.tif", physical=False)
                inside = fb[un > 0]
                row["fuel_missing_in_fire"] = (
                    float(((inside < 91) | (inside > 204)).mean()) if inside.size else np.nan
                )
                row["nonburnable_in_fire"] = (
                    float(((inside >= 91) & (inside <= 99)).mean()) if inside.size else np.nan
                )
                if row["fuel_missing_in_fire"] > 0.05:
                    flag(
                        fid,
                        f"fuels missing for {row['fuel_missing_in_fire']:.0%} of the burned area (LANDFIRE coverage)",
                    )

            # ------------------------------------------------ weather
            wp = d / "weather" / "hrrr_hourly.zarr"
            if not wp.exists():
                flag(fid, "no weather cube")
            else:
                wx = io.read_cube(wp)
                t = pd.DatetimeIndex(wx["time"].values)
                row["wx_hours"] = len(t)
                if (np.diff(t.values) != np.timedelta64(1, "h")).any():
                    flag(fid, "weather time axis has gaps or duplicates")
                if t[0] > t_obs[0] - pd.Timedelta(hours=24) or t[-1] < t_obs[-1]:
                    flag(
                        fid,
                        f"weather {t[0]}..{t[-1]} does not cover obs {t_obs[0]}..{t_obs[-1]} (+24 h before)",
                    )
                if "land" not in wx:
                    flag(fid, "weather cube has no 'land' variable (old weather step; re-run weather)")
                    land = np.ones(wx["latitude"].shape, bool)
                else:
                    land = wx["land"].values.astype(bool)
                for v, lo, hi in (("wspd10", 0, 60), ("gust", 0, 80), ("t2m", -30, 56), ("rh2", 0, 100.5)):
                    a = wx[v].values[:, land]
                    if share_nan(a) > 0:
                        flag(fid, f"weather {v} has {share_nan(a):.1%} NaN over land")
                    if np.nanmin(a) < lo or np.nanmax(a) > hi:
                        flag(
                            fid,
                            f"weather {v} range {np.nanmin(a):.1f}..{np.nanmax(a):.1f} outside {lo}..{hi}",
                        )
                filled = float(wx["qc_filled_f00"].values.mean()) if "qc_filled_f00" in wx else np.nan
                row["wx_filled"] = filled
                if filled > 0.05:
                    flag(fid, f"{filled:.1%} of HRRR analysis hours gap-filled (try weather --retry-missing)")
                # gap-filled stretches longer than 6 h inside the fire period
                if "qc_filled_f00" in wx:
                    q = pd.Series(wx["qc_filled_f00"].values.astype(bool), index=t)
                    q = q[(q.index >= t_obs[0]) & (q.index <= t_obs[-1])]
                    run = (q != q.shift()).cumsum()
                    longest = int(q.groupby(run).sum().max()) if len(q) else 0
                    if longest > 6:
                        flag(fid, f"{longest} consecutive gap-filled HRRR hours during the fire")

            # ------------------------------------------------ dead fuel moisture
            dp = d / "fuel_moisture" / "dead_fuel_moisture.zarr"
            if not dp.exists():
                flag(fid, "no dead fuel moisture")
            else:
                fm = io.read_cube(dp)
                td = pd.DatetimeIndex(fm["time"].values)
                if td[0] > t_obs[0] or td[-1] < t_obs[-1]:
                    flag(fid, f"dead fuel moisture {td[0]}..{td[-1]} does not cover the fire")
                land_fm = (
                    land
                    if wp.exists() and land.shape == fm["latitude"].shape
                    else np.ones(fm["latitude"].shape, bool)
                )
                for v, lo, hi in (("fm1", 1, 65), ("fm10", 1, 65), ("fm100", 1, 65), ("fm1000", 1, 65)):
                    a = fm[v].values[:, land_fm]
                    if share_nan(a) > 0:
                        flag(fid, f"{v} has {share_nan(a):.1%} NaN over land")
                    if np.nanmin(a) < lo or np.nanmax(a) > hi:
                        flag(fid, f"{v} range {np.nanmin(a):.1f}..{np.nanmax(a):.1f}")
                row["fm1_med"] = float(np.nanmedian(fm["fm1"].values[:, land_fm]))

            # ------------------------------------------------ live fuel moisture
            lp = d / "fuel_moisture" / "live_fuel_moisture.zarr"
            if not lp.exists():
                flag(fid, "no live fuel moisture")
            else:
                lf = io.read_cube(lp)
                tl = pd.DatetimeIndex(lf["time"].values)
                if tl[0] > t_obs[0] - pd.Timedelta(days=300) or tl[-1] < t_obs[-1] - pd.Timedelta(days=1):
                    flag(
                        fid,
                        f"live fuel moisture {tl[0].date()}..{tl[-1].date()} does not cover spin-up + fire",
                    )
                lfl = lf["land"].values.astype(bool) if "land" in lf else np.ones(lf["latitude"].shape, bool)
                for v, lo, hi in (("lfm_herb", 30, 250), ("lfm_woody", 60, 200)):
                    a = lf[v].values[:, lfl]
                    if share_nan(a) > 0:
                        flag(fid, f"{v} has {share_nan(a):.1%} NaN over land")
                    if np.nanmin(a) < lo - 1e-3 or np.nanmax(a) > hi + 1e-3:
                        flag(fid, f"{v} range {np.nanmin(a):.0f}..{np.nanmax(a):.0f} outside NFDRS4 limits")
                near = (tl >= t_obs[0] - pd.Timedelta(days=1)) & (tl <= t_obs[-1])
                row["woody_at_fire"] = (
                    float(np.nanmedian(lf["lfm_woody"].values[near][:, lfl])) if near.any() else np.nan
                )

            # ------------------------------------------------ samples
            sp = d / "samples" / "samples.zarr"
            row["samples"] = (sp / "zarr.json").exists()
            if row["samples"]:
                import xarray as xr

                sm = xr.open_zarr(sp, zarr_format=3, consolidated=False)
                v = sm["valid"].values
                ros = sm["inputs"].sel(channel="ros_mean").values[:, v]
                row["ros_max"] = float(np.nanmax(ros)) if ros.size else 0.0
                if row["ros_max"] <= 0:
                    flag(
                        fid,
                        "samples have all-zero physics channels (fuels or moisture missing); rebuild them",
                    )
        except Exception as e:  # keep going; report the failure
            flag(fid, f"check crashed: {type(e).__name__}: {str(e)[:120]}")
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\n{len(df)} fires checked; {df.get('samples', pd.Series(dtype=bool)).sum()} with samples")
    print(f"\nPROBLEMS ({len(problems)}):")
    for p in problems:
        print("  " + p)
    out = cfg.outputs_dir / "validation_report.json"
    out.write_text(json.dumps({"problems": problems, "fires": df.to_dict("records")}, indent=1, default=str))
    print(f"\nSaved {out}")
    return problems

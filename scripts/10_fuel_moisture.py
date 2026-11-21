"""
10_fuel_moisture.py

Hourly dead fuel moisture (1-h, 10-h, 100-h, 1000-h) for one fire with the
Nelson (2000) dead fuel moisture model, as implemented in the USDA Forest
Service Missoula Fire Sciences Lab's NFDRS4 library (Python package `nfdrs4`,
github.com/firelab/NFDRS4). This is the model behind NFDRS version 4.

Inputs per 3 km HRRR cell, hourly:
  - 2 m temperature and RH: from script 08 (hrrr_hourly.nc, HRRR analyses).
    For the spin-up days before that file starts, HRRR analyses are downloaded here.
  - Solar radiation (DSWRF) and 1-hour rainfall (APCP 0-1 h): not available in
    the analyses (f00), so they come from the HRRR 1-hour forecast (f01) issued
    one hour earlier, valid at the same hour.
  - gridMET 100-h and 1000-h fuel moisture (daily, ~4 km): used to set each
    stick's starting moisture at the start of the spin-up, and as an independent
    check of the Nelson 100-h and 1000-h results.

Method
  1. Hourly series run from (first HRRR hour - spin-up days) to the last HRRR hour.
     Missing HRRR hours are filled by linear interpolation in time (rain: 0).
  2. For each cell and size class: create the NFDRS4 stick, initialize it to
     the gridMET value, then call update() once per hour (elapsed time 1 h,
     rain as an hourly amount, as NFDRS4 itself does). Moisture is the stick's
     median radial moisture (also as in NFDRS4).
  3. Results are kept for the script 08 period (spin-up dropped).

Outputs
  data/processed/fires/<id>/fuel_moisture_hourly.nc
      fm1, fm10, fm100, fm1000 (Nelson, %, hourly), fm100_gridmet and
      fm1000_gridmet (%, daily), dswrf (W/m2), apcp (mm/h), land (HRRR land mask)
  outputs/fire_<id>_fuel_moisture_summary.txt
  outputs/fire_<id>_fuel_moisture_timeseries.png   (cell nearest ignition)
  outputs/fire_<id>_fuel_moisture_map.png          (1-h and 10-h at first observation)

Caches (re-runs resume): data/raw/hrrr_fm/<id>/, data/raw/gridmet/

Usage (PowerShell, .venv active)
  python scripts\\10_fuel_moisture.py --fire-id 4299
  python scripts\\10_fuel_moisture.py --fire-id 4299 --max-cells 20   # quick test
"""

import argparse
import json
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

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "fires"
OUT = ROOT / "outputs"

CLASSES = {1: "1", 10: "10", 100: "100", 1000: "1000"}
BP = 0.0218                    # stick barometric pressure (cal/cm3), NFDRS4 default
GRIDMET_URLS = [               # NetCDF Subset Service, per-year files
    "https://www.reacchpna.org/thredds/ncss/grid/MET/{v}/{v}_{y}.nc",
    "http://thredds.northwestknowledge.net:8080/thredds/ncss/MET/{v}/{v}_{y}.nc",
]
GRIDMET_VARS = {"fm100": "dead_fuel_moisture_100hr", "fm1000": "dead_fuel_moisture_1000hr"}
LOCAL_AFTERNOON_UTC = 21       # ~13:00 PST, the time of day NFDRS/gridMET values represent


# ---------------------------------------------------------------- HRRR (script 08 file)
def load_hrrr08(path):
    ds = xr.open_dataset(path)
    need = ["t2m_c", "rh2", "latitude", "longitude"]
    miss = [n for n in need if n not in ds.variables]
    if miss:
        sys.exit(f"{path} lacks {miss}. Variables: {sorted(ds.variables)}")
    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    times = pd.DatetimeIndex(ds["time"].values)
    if times.tz is not None:
        times = times.tz_localize(None)
    return ds, lat, lon, times.astype("datetime64[ns]")


# ---------------------------------------------------------------- HRRR downloads
class HrrrFetcher:
    """Downloads single HRRR fields with Herbie and crops them to the script 08 grid."""

    def __init__(self, lat, lon, cache, tmp):
        self.lat, self.lon = lat, lon
        self.cache, self.tmp = cache, tmp
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.idx_file = cache / "crop_index.json"
        self.crop = json.loads(self.idx_file.read_text()) if self.idx_file.exists() else None
        self.logged = set()

    def _crop(self, da):
        """Cut the script 08 window out of a full CONUS field (checked by lat/lon)."""
        la = np.asarray(da["latitude"].values, float)
        lo = np.asarray(da["longitude"].values, float)
        lo = np.where(lo > 180, lo - 360, lo)
        ny, nx = self.lat.shape
        if self.crop is None:
            d = (la - self.lat[0, 0]) ** 2 + (lo - self.lon[0, 0]) ** 2
            iy, ix = np.unravel_index(np.argmin(d), d.shape)
            self.crop = {"iy": int(iy), "ix": int(ix), "ny": ny, "nx": nx}
            self.idx_file.write_text(json.dumps(self.crop))
        iy, ix = self.crop["iy"], self.crop["ix"]
        sl = (slice(iy, iy + ny), slice(ix, ix + nx))
        err = max(np.abs(la[sl] - self.lat).max(), np.abs(lo[sl] - self.lon).max())
        if err > 1e-3:
            raise RuntimeError(f"HRRR crop does not match the script 08 grid (max error {err:.4f} deg)")
        return np.asarray(da.values, float)[sl]

    @staticmethod
    def _datasets(H, search):
        ds = H.xarray(search, remove_grib=True)
        return ds if isinstance(ds, list) else [ds]

    def _log_inventory(self, H, search, key):
        if key in self.logged:
            return
        self.logged.add(key)
        try:
            inv = H.inventory(search)
            print(f"  GRIB messages used for {key}:")
            for line in inv["search_this"].tolist():
                print(f"    {line}")
        except Exception:
            pass

    def get(self, kind, valid, offline=False):
        """kind 'f00' -> t2m (C), rh (%) at valid; 'f01' -> dswrf (W/m2), apcp (mm)."""
        f = self.cache / f"{kind}_{valid:%Y%m%d%H}.npz"
        if f.exists():
            z = np.load(f)
            return {k: z[k] for k in z.files}
        if offline:
            return None
        from herbie import Herbie
        if kind == "f00":
            init, fxx, search = valid, 0, r":(?:TMP|RH):2 m above ground:"
        else:
            init, fxx, search = valid - pd.Timedelta(hours=1), 1, r":(?:DSWRF|APCP):surface:"
        try:
            H = Herbie(init.to_pydatetime(), model="hrrr", product="sfc", fxx=fxx,
                       priority=["aws", "google"], save_dir=str(self.tmp), verbose=False)
            self._log_inventory(H, search, kind)
            out = {}
            for ds in self._datasets(H, search):
                step = ds.attrs.get("GRIB_stepType", "")
                for name, da in ds.data_vars.items():
                    st = da.attrs.get("GRIB_stepType", step)
                    if kind == "f00" and name in ("t2m", "TMP"):
                        out["t2m"] = self._crop(da) - 273.15
                    elif kind == "f00" and name in ("r2", "rh", "RH"):
                        out["rh"] = self._crop(da)
                    elif kind == "f01" and name.lower().endswith("dswrf"):
                        # prefer the instantaneous value at the valid time
                        if "dswrf" not in out or st == "instant":
                            out["dswrf"] = self._crop(da)
                    elif kind == "f01" and name in ("tp", "apcp", "APCP"):
                        out["apcp"] = self._crop(da)
            want = {"f00": {"t2m", "rh"}, "f01": {"dswrf", "apcp"}}[kind]
            if not want <= set(out):
                raise RuntimeError(f"fields found {sorted(out)}, needed {sorted(want)}")
        except Exception as e:
            print(f"    {kind} {valid:%Y-%m-%d %H}Z missing ({str(e)[:120]})")
            return None
        np.savez_compressed(f, **out)
        return out

    def land(self, valid, offline=False):
        f = self.cache / "land.npy"
        if f.exists():
            return np.load(f)
        if offline:
            return None
        from herbie import Herbie
        try:
            H = Herbie(valid.to_pydatetime(), model="hrrr", product="sfc", fxx=0,
                       priority=["aws", "google"], save_dir=str(self.tmp), verbose=False)
            for ds in self._datasets(H, r":LAND:surface:"):
                for name, da in ds.data_vars.items():
                    m = self._crop(da) > 0.5
                    np.save(f, m)
                    return m
        except Exception as e:
            print(f"  land mask not available ({str(e)[:120]}); all cells treated as land")
        return None


# ---------------------------------------------------------------- gridMET
def gridmet_fetch(key, years, bbox, t0, t1, cache_dir):
    """Daily gridMET fm100/fm1000 for the bbox (cached NetCDF per year)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lonmin, latmin, lonmax, latmax = bbox
    frames = []
    for y in years:
        f = cache_dir / f"{key}_{y}_{latmin:.2f}_{latmax:.2f}_{lonmin:.2f}_{lonmax:.2f}.nc"
        if not f.exists():
            params = {"var": GRIDMET_VARS[key], "north": latmax, "south": latmin,
                      "west": lonmin, "east": lonmax, "horizStride": 1,
                      "time_start": f"{y}-01-01T00:00:00Z", "time_end": f"{y}-12-31T23:59:59Z",
                      "accept": "netcdf"}
            last = None
            for tmpl in GRIDMET_URLS:
                url = tmpl.format(v=key, y=y)
                try:
                    print(f"  downloading gridMET {key} {y} ({url.split('/')[2]})")
                    r = requests.get(url, params=params, timeout=300)
                    if r.status_code == 200 and r.content[:3] in (b"CDF", b"\x89HD"):
                        f.write_bytes(r.content)
                        break
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                except requests.RequestException as e:
                    last = str(e)
                print(f"    failed: {last}")
            else:
                raise RuntimeError(f"gridMET {key} {y} could not be downloaded: {last}")
        ds = xr.open_dataset(f)
        name = GRIDMET_VARS[key] if GRIDMET_VARS[key] in ds else list(ds.data_vars)[0]
        da = ds[name]
        tdim = [d for d in da.dims if d not in ("lat", "lon")][0]
        da = da.rename({tdim: "day"})
        frames.append(da.load())
    da = xr.concat(frames, dim="day").sortby("day")
    d0, d1 = t0.normalize(), t1.normalize()
    return da.sel(day=slice(d0, d1))


def gridmet_to_cells(da, lat, lon):
    """Nearest gridMET cell for every HRRR cell -> array (day, ny, nx)."""
    glat, glon = da["lat"].values, da["lon"].values
    iy = np.abs(glat[:, None] - lat.ravel()[None, :]).argmin(axis=0)
    ix = np.abs(glon[:, None] - lon.ravel()[None, :]).argmin(axis=0)
    v = da.values[:, iy, ix].reshape((da.sizes["day"],) + lat.shape)
    return v.astype(float)


# ---------------------------------------------------------------- Nelson model
def make_stick(nfdrs4, cls):
    D = nfdrs4.DeadFuelMoisture
    s = getattr(D, f"createDeadFuelMoisture{CLASSES[cls]}")()
    getattr(s, f"initDeadFuelMoisture{CLASSES[cls]}")()
    return s


def run_nelson(nfdrs4, cls, T, RH, SW, P, w0):
    """T (C), RH (fraction), SW (W/m2), P (cm per hour): arrays (nt,). w0: g/g.
    Returns moisture in % (nt,) and the number of rejected updates."""
    s = make_stick(nfdrs4, cls)
    s.initializeEnvironment(float(T[0]), float(RH[0]), float(SW[0]), 0.0,
                            float(T[0]), float(RH[0]), float(w0))
    out = np.empty(len(T))
    out[0] = w0 * 100.0
    bad = 0
    for k in range(1, len(T)):
        ok = s.update(1.0, float(T[k]), float(RH[k]), float(SW[k]), float(P[k]), BP, True)
        bad += (not ok)
        out[k] = s.medianRadialMoisture() * 100.0
    return out, bad


# ---------------------------------------------------------------- helpers
def ignition_cell(fire_dir, lat, lon):
    """HRRR cell nearest the centroid of the first perimeter (perim_first.tif)."""
    f = fire_dir / "perim_first.tif"
    if not f.exists():
        return None
    import rasterio
    from pyproj import Transformer
    with rasterio.open(f) as src:
        m = src.read(1) > 0
        rows, cols = np.nonzero(m)
        if len(rows) == 0:
            return None
        x, y = src.xy(rows.mean(), cols.mean())
        tf = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        clon, clat = tf.transform(x, y)
    d = (lat - clat) ** 2 + ((lon - clon) * np.cos(np.radians(clat))) ** 2
    iy, ix = np.unravel_index(np.argmin(d), d.shape)
    return int(iy), int(ix), float(clat), float(clon)


def fill_hours(arr, times, full, name, rain=False):
    """Reindex (nt, ...) array onto the full hourly index; interpolate gaps."""
    nt_full = len(full)
    shp = arr.shape[1:]
    df = pd.DataFrame(arr.reshape(len(times), -1), index=times).reindex(full)
    n_missing = int(df.iloc[:, 0].isna().sum())
    if rain:
        df = df.fillna(0.0)
    else:
        df = df.interpolate(method="time", limit_direction="both")
    return df.values.reshape((nt_full,) + shp), n_missing


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fire-id", required=True, type=int)
    ap.add_argument("--spinup-days", type=float, default=7.0)
    ap.add_argument("--classes", default="1,10,100,1000")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="run Nelson on only the N cells nearest ignition (testing)")
    ap.add_argument("--all-cells", action="store_true",
                    help="also run ocean cells (default: land cells only)")
    ap.add_argument("--offline", action="store_true",
                    help="use cached downloads only")
    args = ap.parse_args()
    classes = [int(c) for c in args.classes.split(",")]

    try:
        import nfdrs4
    except ImportError:
        sys.exit("Package nfdrs4 is not installed in this environment. See the install steps.")

    f = args.fire_id
    fire_dir = PROC / str(f)
    nc08 = fire_dir / "hrrr_hourly.nc"
    if not nc08.exists():
        sys.exit(f"{nc08} not found. Run script 08 first.")
    OUT.mkdir(exist_ok=True)

    ds08, lat, lon, t08 = load_hrrr08(nc08)
    ny, nx = lat.shape
    spin0 = t08.min() - pd.Timedelta(days=args.spinup_days)
    full = pd.date_range(spin0, t08.max(), freq="h")
    keep = full >= t08.min()
    print(f"HRRR grid {ny} x {nx}; script 08 hours {len(t08)} ({t08.min()} to {t08.max()} UTC)")
    print(f"Nelson run {len(full)} hours from {spin0} (spin-up {args.spinup_days:g} days)")

    # ---- temperature / RH: script 08 + spin-up downloads
    fetch = HrrrFetcher(lat, lon, ROOT / "data" / "raw" / "hrrr_fm" / str(f),
                        ROOT / "data" / "raw" / "hrrr" / "_tmp")
    spin_hours = full[full < t08.min()]
    print(f"\nSpin-up temperature/RH: {len(spin_hours)} HRRR analyses")
    Ts, Rs, ts_ok = [], [], []
    for i, t in enumerate(spin_hours):
        r = fetch.get("f00", t, args.offline)
        if r is not None:
            Ts.append(r["t2m"]); Rs.append(r["rh"]); ts_ok.append(t)
        if (i + 1) % 24 == 0:
            print(f"  {i + 1}/{len(spin_hours)}")
    T_all = np.concatenate([np.array(Ts).reshape(-1, ny, nx), ds08["t2m_c"].values], axis=0) \
        if Ts else ds08["t2m_c"].values
    R_all = np.concatenate([np.array(Rs).reshape(-1, ny, nx), ds08["rh2"].values], axis=0) \
        if Rs else ds08["rh2"].values
    t_all = pd.DatetimeIndex(list(ts_ok) + list(t08))
    T, miss_T = fill_hours(T_all, t_all, full, "t2m")
    RH, _ = fill_hours(R_all, t_all, full, "rh")

    # ---- solar radiation and rain from 1-h forecasts
    print(f"\nSolar radiation and 1-h rain: {len(full)} HRRR 1-hour forecasts")
    S, P, tp_ok = [], [], []
    for i, t in enumerate(full):
        r = fetch.get("f01", t, args.offline)
        if r is not None:
            S.append(r["dswrf"]); P.append(r["apcp"]); tp_ok.append(t)
        if (i + 1) % 48 == 0:
            print(f"  {i + 1}/{len(full)}")
    if not tp_ok:
        sys.exit("No DSWRF/APCP fields could be read.")
    SW, miss_S = fill_hours(np.array(S), pd.DatetimeIndex(tp_ok), full, "dswrf")
    PR, miss_P = fill_hours(np.array(P), pd.DatetimeIndex(tp_ok), full, "apcp", rain=True)
    land = fetch.land(t08.min(), args.offline)

    # ---- clamp to the ranges NFDRS4 accepts (it rejects updates outside them)
    T = np.clip(T, -59.9, 59.9)
    RHf = np.clip(RH / 100.0, 0.001, 1.0)
    SW = np.clip(SW, 0.0, 1999.0)
    PR = np.clip(PR, 0.0, None)
    Pcm = PR / 10.0

    # ---- gridMET
    print("\ngridMET 100-h / 1000-h")
    bbox = (lon.min() - 0.05, lat.min() - 0.05, lon.max() + 0.05, lat.max() + 0.05)
    years = sorted({spin0.year, t08.max().year})
    gcache = ROOT / "data" / "raw" / "gridmet"
    g = {}
    for key in ["fm100", "fm1000"]:
        da = gridmet_fetch(key, years, bbox, spin0, t08.max(), gcache)
        g[key] = (pd.DatetimeIndex(da["day"].values), gridmet_to_cells(da, lat, lon))
        v = g[key][1]
        print(f"  {key}: {v.shape[0]} days, range {np.nanmin(v):.1f}-{np.nanmax(v):.1f} %")

    def gm_start(key):
        days, v = g[key]
        i = int(np.searchsorted(days, spin0.normalize(), side="right")) - 1
        w = v[max(i, 0)] / 100.0
        return np.where(np.isfinite(w), w, 0.15)

    w_start = {1: gm_start("fm100"), 10: gm_start("fm100"),
               100: gm_start("fm100"), 1000: gm_start("fm1000")}

    # ---- Nelson
    ign = ignition_cell(fire_dir, lat, lon)
    cells = [(iy, ix) for iy in range(ny) for ix in range(nx)
             if args.all_cells or land is None or land[iy, ix]]
    if land is not None and not args.all_cells:
        print(f"\nSkipping {ny * nx - len(cells)} ocean cells (HRRR land mask); use --all-cells to keep them")
    if args.max_cells and ign:
        cells.sort(key=lambda c: (c[0] - ign[0]) ** 2 + (c[1] - ign[1]) ** 2)
        cells = cells[:args.max_cells]
    nt = len(full)
    res = {c: np.full((nt, ny, nx), np.nan, dtype="float32") for c in classes}
    bad_total = 0
    print(f"\nNelson model: {len(cells)} cells x {len(classes)} classes x {nt} hours")
    t_start = time.time()
    for n, (iy, ix) in enumerate(cells, 1):
        for c in classes:
            out, bad = run_nelson(nfdrs4, c, T[:, iy, ix], RHf[:, iy, ix], SW[:, iy, ix],
                                  Pcm[:, iy, ix], w_start[c][iy, ix])
            res[c][:, iy, ix] = out
            bad_total += bad
        if n in (5, 50) or n % 200 == 0 or n == len(cells):
            el = time.time() - t_start
            print(f"  {n}/{len(cells)} cells, {el:.0f} s elapsed, "
                  f"~{el / n * (len(cells) - n):.0f} s left")

    # ---- write NetCDF (script 08 period only)
    tk = full[keep]
    out = xr.Dataset(coords={"time": tk, "latitude": (("y", "x"), lat),
                             "longitude": (("y", "x"), lon)})
    for c in classes:
        out[f"fm{c}"] = (("time", "y", "x"), res[c][keep])
        out[f"fm{c}"].attrs = {"units": "%", "long_name": f"{c}-h dead fuel moisture, Nelson model (NFDRS4)"}
    for key in ["fm100", "fm1000"]:
        days, v = g[key]
        out = out.assign_coords({f"day": days}) if "day" not in out.coords else out
        out[f"{key}_gridmet"] = (("day", "y", "x"), v.astype("float32"))
        out[f"{key}_gridmet"].attrs = {"units": "%", "long_name": f"gridMET {key[2:]}-h dead fuel moisture (daily)"}
    out["dswrf"] = (("time", "y", "x"), SW[keep].astype("float32"))
    out["dswrf"].attrs = {"units": "W m-2", "long_name": "HRRR DSWRF, 1-h forecast valid at time"}
    out["apcp"] = (("time", "y", "x"), PR[keep].astype("float32"))
    out["apcp"].attrs = {"units": "mm", "long_name": "HRRR 1-h precipitation ending at time"}
    if land is not None:
        out["land"] = (("y", "x"), land.astype("int8"))
    out.attrs = {"source": "Nelson (2000) dead fuel moisture model via NFDRS4 (USFS Missoula Fire Sciences Lab)",
                 "nfdrs4_version": getattr(nfdrs4, "__version__", "unknown"),
                 "spinup_days": args.spinup_days,
                 "initialization": "gridMET fm100 (1/10/100-h) and fm1000 (1000-h) at spin-up start"}
    fo = fire_dir / ("fuel_moisture_hourly_TEST.nc" if args.max_cells else "fuel_moisture_hourly.nc")
    out.to_netcdf(fo)
    print(f"\nWrote {fo.relative_to(ROOT)}")

    # ---- summary
    mask = np.ones((ny, nx), bool) if land is None else land
    done = np.isfinite(res[classes[0]][-1])
    mask = mask & done
    L = [f"Dead fuel moisture, fire {f}", "=" * 50,
         f"Model: Nelson (2000) via NFDRS4 {getattr(nfdrs4, '__version__', '')}; "
         f"hourly, {len(cells)} HRRR cells; spin-up {args.spinup_days:g} days from {spin0} UTC",
         f"Initial moisture: gridMET fm100 (1/10/100-h) and fm1000 (1000-h) on {spin0:%Y-%m-%d}",
         f"Gap-filled hours: temperature/RH {miss_T}, solar {miss_S}, rain {miss_P} (rain set to 0)",
         f"Rejected NFDRS4 updates: {bad_total}",
         f"Land cells summarized: {int(mask.sum())} of {ny * nx}"
         + ("" if land is not None else " (no land mask)"), "",
         "Moisture over land cells, script 08 period (%)", "-" * 50]
    rows = []
    for c in classes:
        v = res[c][keep][:, mask]
        rows.append(dict(cls=f"{c}-h", min=np.nanmin(v), p10=np.nanpercentile(v, 10),
                         median=np.nanmedian(v), p90=np.nanpercentile(v, 90), max=np.nanmax(v)))
    L.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    L += ["", f"Nelson vs gridMET (Nelson at {LOCAL_AFTERNOON_UTC}Z ~ 13:00 PST vs gridMET same day)",
          "-" * 50]
    rows = []
    for c, key in [(100, "fm100"), (1000, "fm1000")]:
        if c not in classes:
            continue
        days, gv = g[key]
        sel = (full.hour == LOCAL_AFTERNOON_UTC) & keep
        nel_days = full[sel].normalize()
        nel = res[c][sel]
        gi = [days.get_loc(d) if d in days else -1 for d in nel_days]
        ok = np.array(gi) >= 0
        a = nel[ok][:, mask].ravel()
        b = gv[np.array(gi)[ok]][:, mask].ravel()
        m = np.isfinite(a) & np.isfinite(b)
        rows.append(dict(cls=f"{c}-h", n=int(m.sum()), nelson=a[m].mean(), gridmet=b[m].mean(),
                         bias=(a[m] - b[m]).mean(), rmse=np.sqrt(((a[m] - b[m]) ** 2).mean()),
                         r=np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan))
    if rows:
        L.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    obs_times = []
    ot = fire_dir / "obs_times.csv"
    if ot.exists():
        obs_times = pd.to_datetime(pd.read_csv(ot)["obs_time_utc"], utc=True).dt.tz_localize(None)
    if ign:
        iy, ix, clat, clon = ign
        L += ["", f"Cell nearest ignition ({lat[iy, ix]:.3f}, {lon[iy, ix]:.3f}); "
              f"values at the hour nearest each of the first 8 observations", "-" * 50]
        rows = []
        for t in list(obs_times)[:8]:
            k = int(np.argmin(np.abs(full - t)))
            d = {"obs_time_utc": t.strftime("%m-%d %H:%M"), "T_C": T[k, iy, ix],
                 "RH_%": RH[k, iy, ix]}
            for c in classes:
                d[f"fm{c}"] = res[c][k, iy, ix]
            rows.append(d)
        L.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    L += ["", "Notes",
          "- Nelson 1-h and 10-h respond within hours and are insensitive to the start value",
          "  after a few days of spin-up. 100-h and 1000-h carry their gridMET start value",
          "  for weeks, so treat their early hours as gridMET-anchored.",
          "- HRRR solar and rain are 1-h forecasts; HRRR temperature and RH are analyses.",
          "- gridMET values are daily and represent afternoon conditions."]
    txt = "\n".join(L)
    (OUT / f"fire_{f}_fuel_moisture_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)

    # ---- figures
    if ign:
        iy, ix = ign[0], ign[1]
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        cols = {1: "tab:red", 10: "tab:orange", 100: "tab:green", 1000: "tab:blue"}
        for c in classes:
            axes[0].plot(tk, res[c][keep][:, iy, ix], lw=1, color=cols[c], label=f"{c}-h Nelson")
        for key, c in [("fm100", 100), ("fm1000", 1000)]:
            days, gv = g[key]
            sel = (days >= tk.min().normalize())
            axes[0].plot(days[sel] + pd.Timedelta(hours=LOCAL_AFTERNOON_UTC), gv[sel, iy, ix],
                         "o", ms=3, color=cols[c], label=f"{c}-h gridMET")
        axes[0].set_ylabel("dead fuel moisture (%)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(tk, RH[keep][:, iy, ix], color="tab:green", lw=0.9)
        axes[1].set_ylabel("RH 2 m (%)")
        ax2 = axes[2]
        ax2.bar(tk, PR[keep][:, iy, ix], width=1 / 24, color="tab:blue", label="rain (mm/h)")
        ax2.set_ylabel("rain (mm/h)")
        ax3 = ax2.twinx()
        ax3.plot(tk, SW[keep][:, iy, ix], color="goldenrod", lw=0.7)
        ax3.set_ylabel("solar (W/m2)")
        for ax in axes:
            for t in obs_times:
                ax.axvline(t, color="grey", lw=0.3, alpha=0.5)
        axes[0].set_title(f"Fire {f}: Nelson dead fuel moisture at the HRRR cell nearest "
                          f"ignition ({lat[iy, ix]:.3f}, {lon[iy, ix]:.3f}); grey = VIIRS times (UTC)")
        fig.tight_layout()
        fig.savefig(OUT / f"fire_{f}_fuel_moisture_timeseries.png", dpi=150)
        plt.close(fig)

    if len(obs_times):
        k = int(np.argmin(np.abs(full - obs_times.iloc[0])))
        show = [c for c in (1, 10) if c in classes]
        fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 5), squeeze=False)
        for ax, c in zip(axes[0], show):
            v = res[c][k].copy()
            if land is not None:
                v[~land] = np.nan
            pm = ax.pcolormesh(lon, lat, v, cmap="YlOrBr_r", shading="nearest")
            fig.colorbar(pm, ax=ax, label="%")
            if ign:
                ax.plot(ign[3], ign[2], "k*", ms=12, label="first perimeter centroid")
                ax.legend(loc="lower left", fontsize=8)
            ax.set_title(f"{c}-h fuel moisture, {full[k]:%Y-%m-%d %H}Z (first observation)")
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
        fig.tight_layout()
        fig.savefig(OUT / f"fire_{f}_fuel_moisture_map.png", dpi=150)
        plt.close(fig)
    print(f"\nWrote outputs/fire_{f}_fuel_moisture_* (summary, time series, map)")


if __name__ == "__main__":
    main()

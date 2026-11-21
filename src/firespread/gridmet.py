"""gridMET daily data (Abatzoglou 2013), ~4 km, via the THREDDS NetCDF Subset Service.

One cached NetCDF per variable, year and box under data/raw/gridmet/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from . import http, manifest
from .config import Config

log = logging.getLogger(__name__)

# file-name key -> variable name inside the per-year files
VARIABLES = {
    "fm100": "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "tmmx": "air_temperature",  # daily max, K
    "tmmn": "air_temperature",  # daily min, K
    "rmax": "relative_humidity",  # daily max, %
    "rmin": "relative_humidity",  # daily min, %
    "pr": "precipitation_amount",  # daily total, mm
    "vs": "wind_speed",  # daily mean 10 m, m/s
}


def _is_netcdf(content: bytes) -> bool:
    return content[:3] in (b"CDF", b"\x89HD")


def fetch(cfg: Config, key: str, years, bbox) -> xr.DataArray:
    """Daily values for whole years over bbox (lonmin, latmin, lonmax, latmax): DataArray (day, lat, lon)."""
    cache = cfg.raw_dir / "gridmet"
    cache.mkdir(parents=True, exist_ok=True)
    lonmin, latmin, lonmax, latmax = bbox
    frames = []
    for y in sorted(set(years)):
        f = cache / f"{key}_{y}_{latmin:.2f}_{latmax:.2f}_{lonmin:.2f}_{lonmax:.2f}.nc"
        if not f.exists():
            _download(cfg, key, y, bbox, f)
        with xr.open_dataset(f) as ds:
            name = VARIABLES[key] if VARIABLES[key] in ds else next(iter(ds.data_vars))
            da = ds[name].load()
        tdim = next(d for d in da.dims if d not in ("lat", "lon"))
        frames.append(da.rename({tdim: "day"}))
    da = xr.concat(frames, dim="day").sortby("day")
    da.name = key
    return da


def _download(cfg: Config, key: str, year: int, bbox, path) -> None:
    lonmin, latmin, lonmax, latmax = bbox
    base = {
        "north": latmax,
        "south": latmin,
        "west": lonmin,
        "east": lonmax,
        "horizStride": 1,
        "time_start": f"{year}-01-01T00:00:00Z",
        "time_end": f"{year}-12-31T23:59:59Z",
        "accept": "netcdf",
    }
    last = None
    for tmpl in cfg.fuel_moisture.gridmet_urls:
        url = tmpl.format(var=key, year=year)
        for var in (VARIABLES[key], "all"):  # "all" if the server names the variable differently
            try:
                log.info("  downloading gridMET %s %s (%s)", key, year, url.split("/")[2])
                r = http.get(url, params={"var": var, **base}, timeout=300)
                if _is_netcdf(r.content):
                    path.write_bytes(r.content)
                    manifest.record(
                        cfg.data_dir,
                        source="gridMET (Abatzoglou 2013) NCSS",
                        url=r.url,
                        path=path,
                        step="gridmet",
                    )
                    return
                last = f"not NetCDF: {r.content[:120]!r}"
            except http.ClientError as e:  # wrong variable name: try "all"
                last = str(e)
                continue
            except Exception as e:  # server down: try the next mirror
                last = str(e)
                break
        log.warning("    %s failed: %s", url.split("/")[2], last)
    raise RuntimeError(f"gridMET {key} {year} could not be downloaded: {last}")


def nearest_index(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gridMET (row, col) nearest to each point of the lat/lon arrays."""
    glat, glon = da["lat"].values, da["lon"].values
    iy = np.abs(glat[:, None] - np.ravel(lat)[None, :]).argmin(axis=0)
    ix = np.abs(glon[:, None] - np.ravel(lon)[None, :]).argmin(axis=0)
    return iy.reshape(np.shape(lat)), ix.reshape(np.shape(lon))


def to_cells(da: xr.DataArray, lat, lon) -> np.ndarray:
    """Nearest gridMET value for every target cell -> array (day, *lat.shape)."""
    iy, ix = nearest_index(da, lat, lon)
    return da.values[:, iy.ravel(), ix.ravel()].reshape((da.sizes["day"], *np.shape(lat))).astype(float)


def local_day(times_utc) -> pd.DatetimeIndex:
    """gridMET day for each UTC time (gridMET days follow local standard time, UTC-8 in California)."""
    return (pd.DatetimeIndex(times_utc) - pd.Timedelta(hours=8)).normalize()

"""Shared data conventions: the one place that reads and writes pipeline data.

Conventions (every step goes through these functions):
  * Rasters: Cloud-Optimized GeoTIFF (COG), deflate, with units, description and
    scale factors stored in the file; categorical layers use nearest overviews.
  * Gridded time series ("cubes"): Zarr (or NetCDF by suffix) following CF-1.10.
    Coordinates are always `time` (UTC), `latitude`, `longitude` (degrees_east
    in -180..180) on dims (`time`, `y`, `x`). Variable names, units and
    standard names come from the VARIABLES registry; unknown names are refused.
  * Times: tz-aware UTC in memory (pandas), naive UTC datetime64 inside xarray
    (CF attribute time_zone = "UTC"), ISO 8601 with a trailing "Z" in CSV files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import Affine, from_origin

from . import __version__

CONVENTIONS = "CF-1.10"

# ------------------------------------------------------------------ registry
# name -> CF attributes. Add new variables here before writing them.
VARIABLES: dict[str, dict[str, str]] = {
    # weather (HRRR)
    "u10": {
        "standard_name": "eastward_wind",
        "units": "m s-1",
        "long_name": "10 m eastward wind (earth-relative)",
    },
    "v10": {
        "standard_name": "northward_wind",
        "units": "m s-1",
        "long_name": "10 m northward wind (earth-relative)",
    },
    "wspd10": {"standard_name": "wind_speed", "units": "m s-1", "long_name": "10 m wind speed"},
    "wdir10": {
        "standard_name": "wind_from_direction",
        "units": "degree",
        "long_name": "10 m wind direction (from)",
    },
    "gust": {"standard_name": "wind_speed_of_gust", "units": "m s-1", "long_name": "surface wind gust"},
    "t2m": {"standard_name": "air_temperature", "units": "degC", "long_name": "2 m air temperature"},
    "rh2": {"standard_name": "relative_humidity", "units": "%", "long_name": "2 m relative humidity"},
    "dswrf": {
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "units": "W m-2",
        "long_name": "downward shortwave radiation at the surface",
    },
    "apcp": {
        "standard_name": "precipitation_amount",
        "units": "kg m-2",
        "long_name": "precipitation in the hour ending at time (mm)",
    },
    "land": {"standard_name": "land_binary_mask", "units": "1", "long_name": "land mask (1 = land)"},
    "qc_filled_f00": {
        "units": "1",
        "long_name": "1 = analysis fields missing in archive, filled by time interpolation",
    },
    "qc_filled_f01": {
        "units": "1",
        "long_name": "1 = 1-h forecast fields missing in archive, filled (rain = 0)",
    },
    # fuel moisture
    "fm1": {"units": "%", "long_name": "1-h dead fuel moisture (Nelson model, NFDRS4)"},
    "fm10": {"units": "%", "long_name": "10-h dead fuel moisture (Nelson model, NFDRS4)"},
    "fm100": {"units": "%", "long_name": "100-h dead fuel moisture (Nelson model, NFDRS4)"},
    "fm1000": {"units": "%", "long_name": "1000-h dead fuel moisture (Nelson model, NFDRS4)"},
    "fm100_gridmet": {"units": "%", "long_name": "gridMET 100-h dead fuel moisture (daily)"},
    "fm1000_gridmet": {"units": "%", "long_name": "gridMET 1000-h dead fuel moisture (daily)"},
}

COORD_ATTRS = {
    "time": {"standard_name": "time", "long_name": "time (UTC)", "time_zone": "UTC"},
    "latitude": {"standard_name": "latitude", "units": "degrees_north"},
    "longitude": {"standard_name": "longitude", "units": "degrees_east"},
    "x_utm": {"long_name": "UTM 11N easting", "units": "m"},
    "y_utm": {"long_name": "UTM 11N northing", "units": "m"},
    "day": {"long_name": "date (gridMET day)"},
}

# HRRR native grid (Lambert conformal conic, sphere R = 6371229 m)
HRRR_GRID_MAPPING = {
    "grid_mapping_name": "lambert_conformal_conic",
    "standard_parallel": 38.5,
    "longitude_of_central_meridian": -97.5,
    "latitude_of_projection_origin": 38.5,
    "earth_radius": 6371229.0,
}


# --------------------------------------------------------------------- times
def to_utc(values) -> pd.DatetimeIndex:
    """Anything datetime-like -> tz-aware UTC DatetimeIndex (naive input is taken as UTC)."""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def utc_naive(values) -> pd.DatetimeIndex:
    """UTC DatetimeIndex without tz, nanosecond precision (what xarray stores)."""
    return to_utc(values).tz_convert(None).astype("datetime64[ns]")


def iso_z(ts) -> str:
    return to_utc([ts])[0].strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    crs: str
    res: float
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return from_origin(self.bounds[0], self.bounds[3], self.res, self.res)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def to_json(self, path: Path, **extra) -> None:
        d = {
            "crs": self.crs,
            "res": self.res,
            "bounds": list(self.bounds),
            "width": self.width,
            "height": self.height,
            "transform": list(self.transform)[:6],
            **extra,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Grid:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["crs"], float(d["res"]), tuple(d["bounds"]), int(d["width"]), int(d["height"]))

    def matches(self, path: Path) -> bool:
        with rasterio.open(path) as src:
            return (
                src.width == self.width
                and src.height == self.height
                and src.crs == rasterio.crs.CRS.from_string(self.crs)
                and src.transform.almost_equals(self.transform)
            )


# ------------------------------------------------------------------- rasters
def write_raster(
    path: Path,
    arr: np.ndarray,
    grid: Grid,
    *,
    nodata,
    dtype: str,
    units: str,
    description: str,
    scale: float | None = None,
    categorical: bool = False,
) -> Path:
    """Write one band as a Cloud-Optimized GeoTIFF on `grid`."""
    if arr.shape != grid.shape:
        raise ValueError(f"{path.name}: array {arr.shape} != grid {grid.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(
        driver="COG",
        width=grid.width,
        height=grid.height,
        count=1,
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        overview_resampling="nearest" if categorical else "average",
    )
    data = arr.astype(dtype)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data, 1)
        dst.units = (units,)
        if scale is not None:
            dst.scales = (scale,)
        dst.set_band_description(1, description)
        dst.update_tags(1, units=units, description=description, categorical=str(categorical).lower())
        dst.update_tags(created_utc=now_utc(), software=f"firespread {__version__}")
    return path


def read_raster(path: Path, *, physical: bool = True) -> tuple[np.ndarray, dict]:
    """Read band 1. physical=True returns float64 with NaN for nodata and the
    scale factor applied; physical=False returns the stored integers/floats."""
    with rasterio.open(path) as src:
        a = src.read(1)
        meta = {
            "nodata": src.nodata,
            "scale": src.scales[0],
            "units": src.units[0],
            "crs": src.crs.to_string(),
            "transform": src.transform,
            "tags": src.tags(1),
        }
    if not physical:
        return a, meta
    out = a.astype("float64")
    if meta["nodata"] is not None:
        out[a == meta["nodata"]] = np.nan
    if meta["scale"] not in (None, 1.0):
        out *= meta["scale"]
    return out, meta


# --------------------------------------------------------------------- cubes
def lon180(lon):
    return ((np.asarray(lon, dtype="float64") + 180.0) % 360.0) - 180.0


def new_cube(times, latitude, longitude, **coords) -> xr.Dataset:
    """Empty CF cube on (time, y, x) with the standard coordinates."""
    ds = xr.Dataset(
        coords={
            "time": ("time", utc_naive(times)),
            "latitude": (("y", "x"), np.asarray(latitude, dtype="float64")),
            "longitude": (("y", "x"), lon180(longitude)),
            **coords,
        }
    )
    return ds


def _apply_attrs(ds: xr.Dataset) -> None:
    for name in ds.data_vars:
        if name == "crs":
            continue
        if name not in VARIABLES:
            raise KeyError(f"Variable '{name}' is not in io.VARIABLES; register it first")
        ds[name].attrs.update(VARIABLES[name])
        if "crs" in ds.data_vars and {"y", "x"} <= set(ds[name].dims):
            ds[name].attrs["grid_mapping"] = "crs"
        ds[name].attrs["coordinates"] = "latitude longitude"
    for name, attrs in COORD_ATTRS.items():
        if name in ds.coords:
            ds[name].attrs.update(attrs)


def write_cube(ds: xr.Dataset, path: Path, *, title: str, source: str, **attrs) -> Path:
    """Validate against the conventions and write Zarr (.zarr) or NetCDF (.nc)."""
    for c in ("time", "latitude", "longitude"):
        if c not in ds.coords:
            raise KeyError(f"cube lacks coordinate '{c}'")
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        raise TypeError("time must be datetime64 (naive UTC)")
    ds = ds.copy()
    _apply_attrs(ds)
    ds.attrs.update(
        {
            "Conventions": CONVENTIONS,
            "title": title,
            "source": source,
            "history": f"{now_utc()} written by firespread {__version__}",
            "institution": "University of California, Riverside (MS capstone)",
            **attrs,
        }
    )
    enc = {"time": {"units": "hours since 1970-01-01 00:00:00", "calendar": "standard"}}
    if "day" in ds.coords:
        enc["day"] = {"units": "days since 1970-01-01", "calendar": "standard"}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr":
        import shutil

        if path.exists():
            shutil.rmtree(path)
        ds.to_zarr(path, mode="w", encoding=enc, zarr_format=3, consolidated=False)
    else:
        ds.to_netcdf(path, encoding=enc)
    return path


def read_cube(path: Path) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=False) if path.suffix == ".zarr" else xr.open_dataset(path)
    for c in ("time", "latitude", "longitude"):
        if c not in ds.coords:
            raise KeyError(f"{path} does not follow the cube conventions (no '{c}')")
    return ds


def hrrr_projected_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-D Lambert x/y (m) of the HRRR rows/columns, for CF projection coordinates."""
    from pyproj import CRS, Transformer

    lcc = CRS.from_cf(HRRR_GRID_MAPPING)
    tr = Transformer.from_crs("EPSG:4326", lcc, always_xy=True)
    x, y = tr.transform(lon180(lon), lat)
    return x.mean(axis=0), y.mean(axis=1)


# -------------------------------------------------------------------- tables
def write_table(df: pd.DataFrame, path: Path) -> Path:
    """CSV with every datetime column as ISO 8601 UTC ('...Z')."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = to_utc(out[c]).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def read_table(path: Path, time_cols: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in time_cols:
        df[c] = pd.Series(to_utc(df[c]), index=df.index)
    return df

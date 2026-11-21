"""Offline stand-ins for Herbie, NFDRS4 and the web services the pipeline calls.

They return data in the same shapes and formats as the real services so every
code path runs without network access (locally and in CI).
"""

from __future__ import annotations

import json
import math
import tempfile
import types
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

# ------------------------------------------------------------------ HRRR grid
_LCC = CRS.from_cf(
    {
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": 38.5,
        "longitude_of_central_meridian": -97.5,
        "latitude_of_projection_origin": 38.5,
        "earth_radius": 6371229.0,
    }
)
_FULL = None
MISSING_F00 = {"2017-12-03 05:00", "2017-12-05 20:00"}  # absent from the fake archive
MISSING_F01 = {"2017-12-04 11:00"}  # valid time of a missing f01


def hrrr_full_grid():
    """A 3 km Lambert grid (like HRRR) covering coastal Southern California."""
    global _FULL
    if _FULL is None:
        tr = Transformer.from_crs("EPSG:4326", _LCC, always_xy=True)
        x0, y0 = tr.transform(-120.6, 33.4)
        xs = x0 + 3000.0 * np.arange(90)
        ys = y0 + 3000.0 * np.arange(80)
        X, Y = np.meshgrid(xs, ys)
        lon, lat = Transformer.from_crs(_LCC, "EPSG:4326", always_xy=True).transform(X, Y)
        _FULL = (lat, lon % 360.0)  # cfgrib reports 0..360
    return _FULL


class FakeHerbie:
    calls: ClassVar[list] = []

    def __init__(self, date, model="hrrr", product="sfc", fxx=0, priority=None, save_dir=None, verbose=False):
        self.init = pd.Timestamp(date)
        self.fxx = fxx
        self.valid = self.init + pd.Timedelta(hours=fxx)
        key = self.valid.strftime("%Y-%m-%d %H:%M")
        missing = (fxx == 0 and key in MISSING_F00) or (fxx == 1 and key in MISSING_F01)
        self.grib = (
            None
            if missing
            else f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/fake/{self.init:%Y%m%d%H}/f{fxx:02d}"
        )
        FakeHerbie.calls.append((self.init, fxx))

    def inventory(self, search):
        return pd.DataFrame({"search_this": [f":FAKE:{search}:{self.fxx} hour fcst"]})

    def xarray(self, search, remove_grib=True):
        lat, lon = hrrr_full_grid()
        c = {"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)}
        h = self.valid.hour + self.valid.minute / 60
        day = self.valid.day
        if "UGRD" in search:
            offshore = day <= 8
            spd = (9.0 if offshore else 3.0) + 2 * np.sin(h / 24 * 2 * np.pi)
            # wind from the NE (offshore) or SW (onshore), given grid-relative (rotated back)
            wdir = 45.0 if offshore else 225.0
            ue = -spd * math.sin(math.radians(wdir))
            ve = -spd * math.cos(math.radians(wdir))
            alpha = np.radians(math.sin(math.radians(38.5)) * (((lon + 180) % 360 - 180) + 97.5))
            ug = np.cos(alpha) * ue - np.sin(alpha) * ve
            vg = np.sin(alpha) * ue + np.cos(alpha) * ve
            rh = np.full(lat.shape, 8.0 if offshore else 45.0) + 10 * np.cos(h / 24 * 2 * np.pi)
            t = 288.0 + 6 * np.sin((h - 15) / 24 * 2 * np.pi) + 0 * lat

            def mk(v, s):
                attrs = {"GRIB_shortName": s, "GRIB_LoVInDegrees": 262.5, "GRIB_Latin1InDegrees": 38.5}
                return xr.DataArray(v.astype("float32"), dims=("y", "x"), attrs=attrs)

            a = xr.Dataset({"u10": mk(ug, "10u"), "v10": mk(vg, "10v")}, coords=c)
            b = xr.Dataset({"t2m": mk(t, "2t"), "r2": mk(rh, "2r")}, coords=c)
            g = xr.Dataset({"gust": mk(np.full(lat.shape, spd * 1.5), "gust")}, coords=c)
            return [a, b, g]
        if "DSWRF" in search:
            sw = max(0.0, 700 * math.sin((h - 15) / 24 * 2 * math.pi)) if 15 <= h or h <= 1 else 0.0
            rain = 1.5 if (self.valid.day == 6 and 3 <= self.valid.hour <= 6) else 0.0
            a = xr.Dataset(
                {"sdswrf": (("y", "x"), np.full(lat.shape, sw, "float32"))},
                coords=c,
                attrs={"GRIB_stepType": "instant"},
            )
            a["sdswrf"].attrs = {"GRIB_shortName": "dswrf", "GRIB_stepType": "instant"}
            b = xr.Dataset(
                {"tp": (("y", "x"), np.full(lat.shape, rain, "float32"))},
                coords=c,
                attrs={"GRIB_stepType": "accum"},
            )
            b["tp"].attrs = {"GRIB_shortName": "tp", "GRIB_stepType": "accum"}
            return [a, b]
        if "LAND" in search:
            land = (((lon + 180) % 360 - 180) > -119.6) & (lat > 33.9)
            return xr.Dataset({"lsm": (("y", "x"), land.astype("float32"))}, coords=c)
        raise ValueError(search)


herbie_module = types.ModuleType("herbie")
herbie_module.Herbie = FakeHerbie


# --------------------------------------------------------------------- NFDRS4
class FakeDeadFuelMoisture:
    """Same calls as nfdrs4.DeadFuelMoisture; simple time-lag physics."""

    def __init__(self, tau):
        self.tau, self.w, self.ready = tau, None, False

    @staticmethod
    def createDeadFuelMoisture1(name=""):
        return FakeDeadFuelMoisture(1.0)

    @staticmethod
    def createDeadFuelMoisture10(name=""):
        return FakeDeadFuelMoisture(10.0)

    @staticmethod
    def createDeadFuelMoisture100(name=""):
        return FakeDeadFuelMoisture(100.0)

    @staticmethod
    def createDeadFuelMoisture1000(name=""):
        return FakeDeadFuelMoisture(1000.0)

    def _init(self):
        self.ready = True

    initDeadFuelMoisture1 = initDeadFuelMoisture10 = initDeadFuelMoisture100 = initDeadFuelMoisture1000 = (
        _init
    )

    def initializeEnvironment(self, ta, ha, sr, rc, ti, hi, wi, bp=0.0218):
        assert self.ready and 0 < wi < 1 and 0 < ha <= 1
        self.w = wi

    def update(self, et, at, rh, sW, rcum, bpr=0.0218, prcpAsAmnt=False):
        if not (0.001 <= rh <= 1 and -60 <= at <= 60 and 0 <= sW <= 2000):
            return False
        emc = 0.01 * (2.0 + 25 * rh**2) - 0.00002 * sW
        self.w += (emc - self.w) * (1 - math.exp(-et / self.tau)) + 0.3 * rcum
        self.w = min(max(self.w, 0.01), 0.6)
        return True

    def medianRadialMoisture(self):
        return self.w


nfdrs4_module = types.ModuleType("nfdrs4")
nfdrs4_module.DeadFuelMoisture = FakeDeadFuelMoisture
nfdrs4_module.__version__ = "fake"


# ----------------------------------------------------------------------- HTTP
class FakeResponse:
    def __init__(self, url, params, content: bytes, status=200):
        self.status_code = status
        self.content = content
        q = urlencode(params, doseq=True) if params else ""
        self.url = f"{url}?{q}" if q else url

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


def _tiff(bbox, size, epsg):
    xmin, ymin, xmax, ymax = (float(v) for v in bbox.split(","))
    w, h = (int(v) for v in size.split(","))
    xs = np.linspace(xmin, xmax, w)
    ys = np.linspace(ymax, ymin, h)
    X, Y = np.meshgrid(xs, ys)
    z = (600 + 500 * np.sin(X / 4000.0) * np.cos(Y / 5000.0)).astype("float32")
    z[X < 270000] = 0.0  # "coast"
    with MemoryFile() as mf:
        with mf.open(
            driver="GTiff",
            width=w,
            height=h,
            count=1,
            dtype="float32",
            crs=f"EPSG:{epsg}",
            transform=from_bounds(xmin, ymin, xmax, ymax, w, h),
        ) as d:
            d.write(z, 1)
        return mf.read()


ASOS_STATIONS = [
    ("OXR", "OXNARD", 34.2008, -119.2072, 13.0),
    ("CMA", "CAMARILLO", 34.2137, -119.0943, 23.0),
    ("SZP", "SANTA PAULA", 34.3472, -119.0611, 74.0),
    ("LAX", "LOS ANGELES", 33.94, -118.40, 30.0),
]
RAWS_STATIONS = [
    ("WLYC1", "WILEY RIDGE", 34.45, -119.03, 2700.0),
    ("ROSC1", "ROSE VALLEY", 34.53, -119.18, 3400.0),
]


def _asos_csv(stid, sts, ets):
    t = pd.date_range(
        pd.Timestamp(sts.rstrip("Z")).ceil("h"), pd.Timestamp(ets.rstrip("Z")), freq="h"
    ) - pd.Timedelta(minutes=7)
    n = len(t)
    rng = np.random.default_rng(abs(hash(stid)) % 2**32)
    offshore = t.day <= 8
    sknt = np.where(offshore, 20, 6) + rng.normal(0, 2, n)
    df = pd.DataFrame(
        {
            "station": stid,
            "valid": t.strftime("%Y-%m-%d %H:%M"),
            "sknt": np.clip(sknt, 0, None).round(),
            "drct": np.where(offshore, 50, 230),
            "gust": np.where(sknt > 15, sknt * 1.5, np.nan).round(),
            "tmpf": (59 + rng.normal(0, 3, n)).round(1),
            "relh": np.clip(np.where(offshore, 10, 50) + rng.normal(0, 4, n), 2, 100).round(1),
        }
    )
    return df.to_csv(index=False, na_rep="").encode()


def _synoptic(params):
    t0 = pd.Timestamp(params["start"])
    t1 = pd.Timestamp(params["end"])
    t = pd.date_range(t0, t1, freq="h") + pd.Timedelta(minutes=13)
    off = t.day <= 8
    st = []
    for sid, name, la, lo, ft in RAWS_STATIONS:
        st.append(
            {
                "STID": sid,
                "NAME": name,
                "LATITUDE": str(la),
                "LONGITUDE": str(lo),
                "ELEVATION": str(ft),
                "OBSERVATIONS": {
                    "date_time": t.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                    "wind_speed_set_1": np.where(off, 12.0, 3.0).tolist(),
                    "wind_direction_set_1": np.where(off, 40.0, 220.0).tolist(),
                    "wind_gust_set_1": np.where(off, 20.0, 6.0).tolist(),
                    "air_temp_set_1": [14.0] * len(t),
                    "relative_humidity_set_1": np.where(off, 6.0, 40.0).tolist(),
                },
            }
        )
    return json.dumps({"SUMMARY": {"RESPONSE_CODE": 1}, "STATION": st}).encode()


def _gridmet(params, var):
    year = int(params["time_start"][:4])
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    lat = np.arange(params["north"], params["south"], -1 / 24)
    lon = np.arange(params["west"], params["east"], 1 / 24)
    base = 7.0 if "100hr" in var else 10.0
    v = base + 2 * np.sin(np.arange(len(days)) / 30)[:, None, None] + 0 * lat[None, :, None] + 0 * lon
    ds = xr.Dataset(
        {var: (("day", "lat", "lon"), v.astype("float32"))}, coords={"day": days, "lat": lat, "lon": lon}
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.nc"
        ds.to_netcdf(p)
        return p.read_bytes()


class FakeWeb:
    """Routes http.get(url, params) to synthetic responses; records every call."""

    def __init__(self, synoptic_status: int = 200):
        self.calls: list[str] = []
        self.synoptic_status = synoptic_status

    def __call__(self, url, params=None, timeout=None):
        from firespread import http

        self.calls.append(url)
        p = dict(params) if isinstance(params, dict) else {}
        if "3DEPElevation" in url:
            return FakeResponse(url, params, _tiff(p["bbox"], p["size"], p["imageSR"]))
        if "geojson/network" in url:
            feats = [
                {
                    "type": "Feature",
                    "id": s,
                    "geometry": {"type": "Point", "coordinates": [lo, la]},
                    "properties": {"sid": s, "sname": n, "elevation": e},
                }
                for s, n, la, lo, e in ASOS_STATIONS
            ]
            return FakeResponse(
                url, params, json.dumps({"type": "FeatureCollection", "features": feats}).encode()
            )
        if "asos.py" in url:
            d = dict(params)
            return FakeResponse(url, params, _asos_csv(d["station"], d["sts"], d["ets"]))
        if "synopticdata" in url:
            if self.synoptic_status != 200:
                raise http.ClientError(
                    self.synoptic_status, '{"SUMMARY": {"RESPONSE_MESSAGE": "Invalid token"}}', url
                )
            return FakeResponse(url, params, _synoptic(p))
        if "reacchpna" in url:
            raise http.ClientError(503, "simulated outage of the first mirror", url)
        if "thredds" in url:
            return FakeResponse(url, params, _gridmet(p, p["var"]))
        raise AssertionError(f"unexpected URL {url}")

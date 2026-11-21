"""VIIRS fire records (Chen et al. 2022 Largefire / FEDS) and their real observation times.

The Largefire `time` field is a LABEL, not a UTC timestamp (FEDS convention):
00:00 = the ~01:30 local-solar-time VIIRS overpass, 12:00 = the ~13:30 overpass,
and the date is the UTC date. Real UTC time = date + (1.5 or 13.5 h) - clon/15 h,
about +9.4 h in Southern California.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

LAYER = "perimeter"


def label_to_utc(label_time: pd.Series, clon: pd.Series) -> pd.Series:
    """Largefire half-day labels (naive) + centroid longitude -> tz-aware UTC observation times."""
    lab = pd.to_datetime(label_time)
    if lab.dt.tz is not None:
        lab = lab.dt.tz_localize(None)
    hrs = lab.dt.hour
    if not hrs.isin([0, 12]).all():
        raise ValueError(f"Unexpected label hours {sorted(hrs.unique())}; expected only 0 and 12")
    solar = np.where(hrs == 0, 1.5, 13.5)
    offset = -clon.to_numpy(dtype=float) / 15.0
    obs = lab.dt.normalize() + pd.to_timedelta(solar + offset, unit="h")
    return obs.dt.tz_localize("UTC")


def _layer(gpkg: Path) -> str:
    names = [row[0] for row in pyogrio.list_layers(gpkg)]
    return LAYER if LAYER in names else names[0]


def load_records(gpkg: Path, fire_id: int, geometry: bool = True) -> gpd.GeoDataFrame | pd.DataFrame:
    """All 12-hour records of one fire, sorted, with `label_time` and `obs_time_utc`."""
    layer = _layer(gpkg)
    if geometry:
        df = gpd.read_file(gpkg, layer=layer, where=f"fireID = {int(fire_id)}")
    else:
        df = pyogrio.read_dataframe(gpkg, layer=layer, read_geometry=False, where=f"fireID = {int(fire_id)}")
    if df.empty:
        raise KeyError(f"fireID {fire_id} not found in {gpkg.name} (layer '{layer}')")
    lab = pd.to_datetime(df["time"])
    df["label_time"] = lab.dt.tz_localize(None) if lab.dt.tz is not None else lab
    df["obs_time_utc"] = label_to_utc(df["label_time"], df["clon"])
    return df.sort_values("obs_time_utc").reset_index(drop=True)


def obs_table(records: pd.DataFrame) -> pd.DataFrame:
    """Per-record table written as perimeters/obs_times.csv."""
    out = records[["label_time", "obs_time_utc", "farea", "n_newpixels", "clat", "clon"]].copy()
    out.insert(0, "record", range(len(out)))
    out["label_time"] = out["label_time"].dt.strftime("%Y-%m-%d %H:%M (label)")
    return out

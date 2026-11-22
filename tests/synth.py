"""Builds a small synthetic study area with the same file formats as the real inputs."""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
UTM = "EPSG:32611"

# fireID: (lon, lat, first label, number of records, final half-size m)
FIRES = {
    4299: (-119.10, 34.40, "2017-12-05 00:00", 5, 4500),  # Thomas-like test fire
    3922: (-118.75, 34.10, "2018-11-09 00:00", 3, 4000),  # Woolsey-like test fire
    5001: (-117.20, 33.80, "2015-07-01 00:00", 6, 4000),
    5002: (-116.80, 34.30, "2016-08-10 00:00", 6, 4200),
    5003: (-117.60, 33.20, "2017-06-15 00:00", 6, 3500),
    5004: (-118.20, 34.60, "2018-07-20 00:00", 6, 4600),
    5005: (-120.10, 34.90, "2019-09-01 00:00", 6, 3800),
    5006: (-116.20, 33.50, "2016-05-05 00:00", 3, 1500),
    5101: (-117.00, 33.40, "2016-06-01 00:00", 1, 1000),  # single record
    5102: (-117.30, 34.00, "2013-08-01 00:00", 4, 2000),  # before HRRR
    5103: (-117.00, 32.90, "2017-12-06 00:00", 3, 2000),  # concurrent with the Thomas-like fire
    5104: (-119.80, 36.70, "2016-07-01 00:00", 4, 2000),  # outside SoCal (Fresno)
}
LONG_FIRE = 5105  # two records 70 days apart


def _records(fid, lon, lat, first, n, half):
    tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    x, y = tr.transform(lon, lat)
    t0 = pd.Timestamp(first)
    rows = []
    for k in range(n):
        h = half * (k + 1) / n
        poly = (
            gpd.GeoSeries([shapely.box(x - h, y - h * 0.8, x + h, y + h * 0.8)], crs=UTM).to_crs(4326).iloc[0]
        )
        rows.append(
            {
                "fireID": fid,
                "time": (t0 + pd.Timedelta(hours=12 * k)).strftime("%Y-%m-%dT%H:%M:%S"),
                "clat": poly.centroid.y,
                "clon": poly.centroid.x,
                "farea": (2 * h) * (1.6 * h) / 1e6,
                "fperim": 7.2 * h / 1e3,
                "n_pixels": 10 * (k + 1),
                "n_newpixels": 10,
                "meanFRP": 25.0,
                "year": t0.year,
                "geometry": poly,
            }
        )
    return rows


def build(root: Path) -> Path:
    """Create config + raw inputs under root; returns the config path."""
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text())
    cfg["fires"] = [4299]
    cfg["hrrr"]["workers"] = 1  # fakes live in this process; the parallel path has its own test
    (root / "config").mkdir()
    (root / "config" / "pipeline.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Largefire GeoPackage in a tar.gz
    rows = []
    for fid, (lon, lat, first, n, half) in FIRES.items():
        rows += _records(fid, lon, lat, first, n, half)
    long = _records(LONG_FIRE, -116.9, 34.2, "2016-01-01 00:00", 2, 2000)
    long[1]["time"] = "2016-03-12T00:00:00"
    rows += long
    lf = gpd.GeoDataFrame(rows, crs=4326)
    work = root / "_build" / "Largefire"
    work.mkdir(parents=True)
    lf.to_file(work / "LargeFires_2012-2020.gpkg", layer="perimeter", driver="GPKG")
    with tarfile.open(raw / "Largefire.tar.gz", "w:gz") as t:
        t.add(work, arcname="Largefire")

    # counties (10 SoCal boxes + Fresno), EPSG:3857 shapefile in a zip
    names = [
        "San Luis Obispo",
        "Kern",
        "San Bernardino",
        "Riverside",
        "Imperial",
        "Santa Barbara",
        "Ventura",
        "Los Angeles",
        "Orange",
        "San Diego",
    ]
    polys = []
    for i, nm in enumerate(names):
        col, row = i % 5, i // 5
        polys.append(
            {
                "NAME": nm,
                "geometry": shapely.box(
                    -121 + 1.4 * col, 34.15 - 1.65 * row, -121 + 1.4 * (col + 1), 35.8 - 1.65 * row
                ),
            }
        )
    polys.append({"NAME": "Fresno", "geometry": shapely.box(-121, 35.8, -118, 37.5)})
    cdir = root / "_build" / "counties"
    cdir.mkdir()
    gpd.GeoDataFrame(polys, crs=4326).to_crs(3857).to_file(cdir / "CA_Counties.shp")
    with zipfile.ZipFile(raw / "ca_counties.zip", "w") as z:
        for f in cdir.iterdir():
            z.write(f, f"CA_Counties/{f.name}")

    # FRAP FileGDB in a zip
    def final(fid):
        r = [x for x in rows if x["fireID"] == fid][-1]["geometry"]
        return gpd.GeoSeries([r], crs=4326).to_crs(3310).buffer(300).iloc[0]

    frap = gpd.GeoDataFrame(
        [
            {
                "FIRE_NAME": "THOMAS",
                "YEAR_": 2017,
                "ALARM_DATE": pd.Timestamp("2017-12-04"),
                "GIS_ACRES": 5000.0,
                "geometry": final(4299),
            },
            {
                "FIRE_NAME": "WOOLSEY",
                "YEAR_": 2018,
                "ALARM_DATE": pd.Timestamp("2018-11-08"),
                "GIS_ACRES": 4000.0,
                "geometry": final(3922),
            },
            {
                "FIRE_NAME": "OTHER",
                "YEAR_": 2016,
                "ALARM_DATE": pd.Timestamp("2016-08-10"),
                "GIS_ACRES": 100.0,
                "geometry": final(5002),
            },
        ],
        crs=3310,
    )
    gdb = root / "_build" / "fire25_1.gdb"
    frap.to_file(
        gdb,
        layer="firep25_1",
        driver="OpenFileGDB",
        layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},
    )
    with zipfile.ZipFile(raw / "fire251gdb.zip", "w") as z:
        for f in gdb.rglob("*"):
            z.write(f, f"fire25_1.gdb/{f.relative_to(gdb).as_posix()}")

    # LANDFIRE-like rasters (EPSG:5070, 90 m, ocean fill -10000)
    lfd = raw / "landfire" / "LF2016_Remap"
    lfd.mkdir(parents=True)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = tr.transform([-119.5, -118.6, -119.5, -118.6], [34.0, 34.0, 34.8, 34.8])
    x0, y1 = min(xs), max(ys)
    w, h = int((max(xs) - x0) / 90) + 1, int((y1 - min(ys)) / 90) + 1
    yy, xx = np.mgrid[0:h, 0:w]
    ocean = xx < w * 0.37
    layers = {
        "FBFM40": np.where((xx // 40 + yy // 40) % 3 == 0, 145, np.where((xx // 40) % 2, 122, 102)),
        "CC": (xx % 60),
        "CH": (xx % 200),
        "CBH": (yy % 50),
        "CBD": (yy % 30),
    }
    for key, a in layers.items():
        a = np.where(ocean, -10000, a).astype("int16")
        with rasterio.open(
            lfd / f"LF2016_{key}_CONUS.tif",
            "w",
            driver="GTiff",
            width=w,
            height=h,
            count=1,
            dtype="int16",
            crs="EPSG:5070",
            transform=from_origin(x0, y1, 90, 90),
            nodata=32767,
        ) as d:
            d.write(a, 1)
    # Globe-LFMC-like workbook (chamise near the Thomas-like fire + other rows)
    lfd = raw / "lfmc"
    lfd.mkdir()
    dates = pd.date_range("2017-01-10", "2017-12-20", freq="14D")
    recs = []
    for site, la, lo in (("Ventura Hills", 34.45, -119.15), ("Ojai Valley", 34.50, -119.25)):
        for d in dates:
            for k in range(2):
                recs.append(
                    {
                        "Site name": site,
                        "Country": "United States",
                        "State/Region": "CA",
                        "Latitude (WGS84, EPSG:4326)": la,
                        "Longitude (WGS84, EPSG:4326)": lo,
                        "Sampling date (YYYYMMDD)": d,
                        "LFMC value (%)": 60 + 60 * (d.month in (3, 4, 5)) + k,
                        "Species collected": "Adenostoma fasciculatum",
                        "Species functional type": "Shrub",
                        "Individual sample or mean value": "Individual sample",
                        "Elevation (m.a.s.l)": 300.0,
                    }
                )
    recs.append({**recs[0], "Site name": "Far away", "Latitude (WGS84, EPSG:4326)": 40.0})
    recs.append({**recs[0], "Site name": "Grass", "Species collected": "Bromus"})
    recs.append({**recs[0], "Site name": "Abroad", "Country": "Spain"})
    with pd.ExcelWriter(lfd / "Globe-LFMC-2.0 final.xlsx") as xw:
        pd.DataFrame({"Contact": ["x"]}).to_excel(xw, sheet_name="Contact", index=False)
        pd.DataFrame(recs).to_excel(xw, sheet_name="LFMC data", index=False)
        pd.DataFrame({"Protocol code": [1]}).to_excel(xw, sheet_name="Protocol", index=False)
    shutil.rmtree(root / "_build")
    return root / "config" / "pipeline.yaml"

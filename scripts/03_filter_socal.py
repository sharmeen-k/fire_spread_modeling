"""
Filter Largefire to Southern California (10 counties) by ignition location.
Writes outputs/socal_fires.csv (one row per fire) and
data/processed/socal_perimeters.gpkg (all 12-hour records for those fires).
Usage: python scripts/03_filter_socal.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

SOCAL = ["Imperial", "Kern", "Los Angeles", "Orange", "Riverside",
         "San Bernardino", "San Diego", "San Luis Obispo", "Santa Barbara", "Ventura"]
HRRR_START = pd.Timestamp("2014-07-01")  # approximate; verify the real HRRR archive start

gpkg = next(Path("data/extracted/Largefire").rglob("LargeFires_2012-2020.gpkg"))
shp = next(Path("data/extracted/counties").rglob("*.shp"))

counties = gpd.read_file(shp).to_crs(4326)
socal = counties[counties["NAME"].isin(SOCAL)]
assert len(socal) == 10, f"expected 10 counties, got {len(socal)}"
socal_union = socal.union_all()

# fire-level table from attributes only (no geometry, fast)
attrs = pyogrio.read_dataframe(
    gpkg, layer="perimeter", read_geometry=False,
    columns=["fireID", "time", "clat", "clon", "farea", "year"])
attrs["time"] = pd.to_datetime(attrs["time"])
attrs = attrs.sort_values(["fireID", "time"])

fires = attrs.groupby("fireID").agg(
    start=("time", "min"), end=("time", "max"), n_records=("time", "size"),
    max_farea=("farea", "max"),
    start_clat=("clat", "first"), start_clon=("clon", "first")).reset_index()

pts = gpd.GeoSeries(gpd.points_from_xy(fires["start_clon"], fires["start_clat"]), crs=4326)
fires["socal"] = pts.within(socal_union).values
socal_fires = fires[fires["socal"]].drop(columns="socal")

# all 12-hour records for SoCal fires
peri = gpd.read_file(gpkg, layer="perimeter")
peri = peri[peri["fireID"].isin(socal_fires["fireID"])]
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
peri.to_file("data/processed/socal_perimeters.gpkg", layer="perimeter", driver="GPKG")
socal_fires.to_csv("outputs/socal_fires.csv", index=False)

print(f"All Largefire fires:        {len(fires)}  ({len(attrs)} records)")
print(f"SoCal fires (10 counties):  {len(socal_fires)}  ({len(peri)} records)")
print(f"  starting on/after {HRRR_START.date()} (HRRR coverage): {(socal_fires['start'] >= HRRR_START).sum()}")
print(f"  max_farea range: {socal_fires['max_farea'].min():.2f} to {socal_fires['max_farea'].max():.1f}")
print("\nSoCal fires per start year:")
print(socal_fires["start"].dt.year.value_counts().sort_index().to_string())
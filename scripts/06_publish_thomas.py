"""
Publish the observed Thomas fire (fireID 4299) to ArcGIS Online as a
time-enabled hosted feature layer, plus a web map.
Usage: python scripts/06_publish_thomas.py --url https://<org>.maps.arcgis.com --user <username>
Run inside .venv-arcgis. 
"""
import argparse
import getpass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from arcgis.features import GeoAccessor
from arcgis.gis import GIS
    
FIRE_ID = 4299
TITLE = "Thomas_2017_observed_VIIRS"

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--user", required=True)
args = ap.parse_args()

# ---- prepare data (runs before any login) ----
g = gpd.read_file("data/processed/socal_perimeters.gpkg", layer="perimeter")
g = g[g["fireID"] == FIRE_ID].copy()
g["obs_time"] = pd.to_datetime(g["time"])
g = g.sort_values("obs_time").reset_index(drop=True)
g["end_time"] = g["obs_time"] + pd.Timedelta(hours=12)   # each perimeter shows for its 12 h window
g["step"] = range(len(g))
bad = ~g.is_valid
g.loc[bad, "geometry"] = g.loc[bad, "geometry"].make_valid()
assert set(g.geom_type) <= {"Polygon", "MultiPolygon"}, set(g.geom_type)
g = g[["fireID", "step", "obs_time", "end_time", "farea", "fperim",
       "n_pixels", "n_newpixels", "meanFRP", "geometry"]]

Path("data/processed").mkdir(parents=True, exist_ok=True)
g.to_file("data/processed/thomas_4299.gpkg", layer="perimeter", driver="GPKG")  # for manual upload
print(f"{len(g)} perimeters, {g['obs_time'].min()} to {g['obs_time'].max()}, CRS {g.crs}")

# ---- sign in ----
gis = GIS(args.url, args.user, getpass.getpass("ArcGIS password: "))
print("Signed in as", gis.users.me.username)

existing = gis.content.search(f'title:"{TITLE}" AND owner:{gis.users.me.username}')
if existing:
    if input(f"{len(existing)} item(s) named {TITLE} exist. Delete and recreate? [y/N] ").lower() != "y":
        raise SystemExit("Stopped; nothing changed.")
    for it in existing:
        it.delete()

# ---- publish hosted feature layer ----
sdf = GeoAccessor.from_geodataframe(g)
item = sdf.spatial.to_featurelayer(TITLE, gis=gis)
item.update(item_properties={
    "tags": "wildfire, Thomas fire, VIIRS, capstone",
    "snippet": "Observed Thomas fire (Dec 2017) perimeters at 12-hour steps from VIIRS (Chen et al. 2022)."})

# ---- enable time (start and end fields) ----
layer = item.layers[0]
layer.manager.update_definition({"timeInfo": {
    "startTimeField": "obs_time",
    "endTimeField": "end_time",
    "timeReference": {"timeZone": "UTC"},
    "timeInterval": 12,
    "timeIntervalUnits": "esriTimeUnitsHours"}})

print("Layer item:", item.homepage)
print("Next: open the item, click Open in Map Viewer, and add a time slider.")
"""
Find Thomas (2017) and Woolsey (2018) in Largefire by matching FRAP perimeters.
Writes outputs/test_fire_candidates.csv. 
Usage: python scripts/04_find_test_fires.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

TARGETS = {"THOMAS": 2017, "WOOLSEY": 2018}
WINDOW = pd.Timedelta(days=7)

fires = pd.read_csv("outputs/socal_fires.csv", parse_dates=["start", "end"])
peri = gpd.read_file("data/processed/socal_perimeters.gpkg", layer="perimeter")
peri["time"] = pd.to_datetime(peri["time"])

# final perimeter per fire, in California Albers (EPSG:3310) for area math
# (assumes each perimeter is cumulative; check that farea grows over time)
last = (peri.sort_values("time").groupby("fireID").tail(1)
        .set_index("fireID").to_crs(3310))
last["geometry"] = last.geometry.make_valid()

gdb = next(Path("data/extracted/frap").rglob("*.gdb"))
layer = next(n for n, _ in pyogrio.list_layers(gdb) if n.startswith("firep"))
frap = gpd.read_file(gdb, layer=layer).to_crs(3310)

rows = []
for name, year in TARGETS.items():
    cand = frap[(frap["FIRE_NAME"].str.upper() == name) & (frap["YEAR_"] == year)]
    if cand.empty:
        print(f"No FRAP match for {name} {year}")
        continue
    fp = cand.sort_values("GIS_ACRES", ascending=False).iloc[0]
    fgeom = gpd.GeoSeries([fp.geometry], crs=3310).make_valid().iloc[0]
    alarm = fp["ALARM_DATE"].tz_localize(None) if fp["ALARM_DATE"].tzinfo else fp["ALARM_DATE"]
    print(f"\nFRAP {name} {year}: alarm {alarm.date()}, {fp['GIS_ACRES']:.0f} acres")

    near = fires[(fires["start"] >= alarm - WINDOW) & (fires["start"] <= alarm + WINDOW)]
    for fid in near["fireID"]:
        g = last.loc[fid, "geometry"]
        inter = g.intersection(fgeom).area
        iou = inter / g.union(fgeom).area if g.union(fgeom).area else 0
        rows.append({"target": name, "fireID": fid, "iou": round(iou, 3),
                     "start": near.loc[near["fireID"] == fid, "start"].iloc[0],
                     "n_records": int(near.loc[near["fireID"] == fid, "n_records"].iloc[0])})

out = pd.DataFrame(rows).sort_values(["target", "iou"], ascending=[True, False])
out.to_csv("outputs/test_fire_candidates.csv", index=False)
print("\nTop candidates:")
print(out.groupby("target").head(3).to_string(index=False))
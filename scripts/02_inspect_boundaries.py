"""
Extract the county shapefile and FRAP geodatabase, then list layers,
columns, CRS and row counts. Writes outputs/boundaries_schema.txt.
Usage: python scripts/02_inspect_boundaries.py
"""
import zipfile
from pathlib import Path

import geopandas as gpd
import pyogrio

RAW = Path("data/raw")
EXT = Path("data/extracted")
ZIPS = {"counties": RAW / "ca_counties.zip", "frap": RAW / "fire251gdb.zip"}  # adjust names if needed

lines = []
for key, zpath in ZIPS.items():
    out = EXT / key
    if not out.exists():
        out.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)

    # shapefiles are files; a .gdb is a folder, so rglob finds both
    sources = sorted(out.rglob("*.shp")) + sorted(out.rglob("*.gdb"))
    lines.append(f"\n##### {key}: {[s.name for s in sources]}")

    for src in sources:
        for name, geom_type in pyogrio.list_layers(src):
            info = pyogrio.read_info(src, layer=name)
            lines.append(f"\n--- {src.name} | layer: {name} | geometry: {geom_type} | rows: {info['features']} ---")
            lines.append(f"CRS: {info['crs']}")
            sample = gpd.read_file(src, layer=name, rows=3)
            for col, dtype in sample.dtypes.items():
                ex = "" if col == "geometry" else f"  e.g. {sample[col].iloc[0] if len(sample) else None}"
                lines.append(f"  {col:30s} {dtype}{ex}")

report = "\n".join(lines)
print(report)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/boundaries_schema.txt").write_text(report)
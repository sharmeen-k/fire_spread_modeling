"""
Extract Largefire.tar.gz, list every GeoPackage, its layers,
and each layer's columns, CRS and row count.
Run as: python scripts/01_inspect_largefire.py data/raw/Largefire.tar.gz
"""
import sys
import tarfile
from pathlib import Path

import geopandas as gpd
import pyogrio

tar_path = Path(sys.argv[1])
out_dir = Path("data/extracted") / tar_path.name.replace(".tar.gz", "")
report_path = Path("outputs/largefire_schema.txt")
report_path.parent.mkdir(exist_ok=True)

if not out_dir.exists():
    out_dir.mkdir(parents=True)
    with tarfile.open(tar_path) as t:
        t.extractall(out_dir)

gpkgs = sorted(out_dir.rglob("*.gpkg"))
lines = [f"Archive: {tar_path.name}", f"GeoPackage files found: {len(gpkgs)}"]
lines += [f"  {p.relative_to(out_dir)}" for p in gpkgs]

# Inspect the first file fully; for the rest, just count layers.
first = gpkgs[0]
layers = pyogrio.list_layers(first)  # array of [name, geometry type]
lines.append(f"\n=== {first.name}: {len(layers)} layer(s) ===")

for name, geom_type in layers:
    info = pyogrio.read_info(first, layer=name)
    lines.append(f"\n--- layer: {name} | geometry: {geom_type} | rows: {info['features']} ---")
    lines.append(f"CRS: {info['crs']}")
    sample = gpd.read_file(first, layer=name, rows=3)
    for col, dtype in sample.dtypes.items():
        example = "" if col == "geometry" else f"  e.g. {sample[col].iloc[0] if len(sample) else None}"
        lines.append(f"  {col:30s} {dtype}{example}")

for p in gpkgs[1:]:
    lines.append(f"{p.name}: {len(pyogrio.list_layers(p))} layer(s)")

report = "\n".join(lines)
print(report)
report_path.write_text(report)
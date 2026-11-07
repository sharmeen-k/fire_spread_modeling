"""
07_build_fire_stack.py
Build the 30 m model grid for one fire and put terrain and LANDFIRE fuels on it.

Phase 1 (always runs):
    - 30 m grid in UTM 11N (EPSG:32611) around the fire's full extent + 5 km buffer
    - 3DEP DEM resampled onto that grid (USGS ImageServer, no API key)
    - slope and aspect (Horn method)
    - perimeter masks: first record (ignition), last record, union of all records
Phase 2 (runs when LANDFIRE GeoTIFFs are found in --landfire-dir):
    - FBFM40, CC, CH, CBH, CBD reprojected onto the same grid (nearest neighbour)

Outputs:
    data/processed/fires/<fireID>/*.tif and grid.json
    outputs/fire_<fireID>_stack_summary.txt
    outputs/fire_<fireID>_alignment.png

Usage (repo root, .venv active):
    python scripts/07_build_fire_stack.py --fire-id 4299 --landfire-dir data/raw/landfire/LF2016_Remap
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds, Resampling

REPO = Path(__file__).resolve().parents[1]
PERIM_GPKG = REPO / "data" / "processed" / "socal_perimeters.gpkg"
OUT_DIR = REPO / "outputs"

CRS = "EPSG:32611"
RES = 30.0
# Grid edges snapped to 750 m (LCM of 30 m and 375 m) so a 375 m VIIRS
# scoring grid can share the same origin. 375/30 = 12.5, so aggregation
# to 375 m must still be area-weighted (Resampling.average), not block sums.
SNAP = 750.0
BUFFER_M = 5000.0

DEM_URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
           "3DEPElevation/ImageServer/exportImage")
DEM_TILE = 2000  # pixels per request side

LF_NODATA = -9999
# Filename tokens -> layer key. Handles old (LC16_F40_200) and new
# (LF2016_FBFM40_CONUS) LANDFIRE naming.
LF_TOKENS = {"F40": "fbfm40", "FBFM40": "fbfm40", "CC": "cc", "CH": "ch",
             "CBH": "cbh", "CBD": "cbd"}
LF_LAYERS = ["fbfm40", "cc", "ch", "cbh", "cbd"]
FBFM40_GROUPS = [  # (label, lo, hi, colour)
    ("NB non-burnable", 91, 99, "#bdbdbd"),
    ("GR grass", 101, 109, "#f2e394"),
    ("GS grass-shrub", 121, 124, "#d9b26f"),
    ("SH shrub", 141, 149, "#8c6d31"),
    ("TU timber-understory", 161, 165, "#78a35a"),
    ("TL timber litter", 181, 189, "#2f6b3a"),
    ("SB slash-blowdown", 201, 204, "#a14b3a"),
]


# ---------------------------------------------------------------- inputs
def load_fire(fire_id):
    names = [row[0] for row in pyogrio.list_layers(PERIM_GPKG)]
    layer = "perimeter" if "perimeter" in names else names[0]
    gdf = gpd.read_file(PERIM_GPKG, layer=layer)
    gdf = gdf[gdf["fireID"] == fire_id].copy()
    if gdf.empty:
        sys.exit(f"fireID {fire_id} not found in {PERIM_GPKG.name} (layer '{layer}')")
    gdf["time"] = pd.to_datetime(gdf["time"], utc=True)
    gdf = gdf.sort_values("time").reset_index(drop=True).to_crs(CRS)
    return gdf, layer


# ------------------------------------------------------------------ grid
def make_grid(gdf):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin = np.floor((xmin - BUFFER_M) / SNAP) * SNAP
    ymin = np.floor((ymin - BUFFER_M) / SNAP) * SNAP
    xmax = np.ceil((xmax + BUFFER_M) / SNAP) * SNAP
    ymax = np.ceil((ymax + BUFFER_M) / SNAP) * SNAP
    width = int(round((xmax - xmin) / RES))
    height = int(round((ymax - ymin) / RES))
    return {
        "crs": CRS, "res": RES,
        "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "width": width, "height": height,
        "transform": from_origin(xmin, ymax, RES, RES),
    }


def write_tif(path, arr, grid, nodata, dtype):
    profile = dict(driver="GTiff", height=grid["height"], width=grid["width"],
                   count=1, dtype=dtype, crs=CRS, transform=grid["transform"],
                   nodata=nodata, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def grid_matches(path, grid):
    with rasterio.open(path) as src:
        return (src.width == grid["width"] and src.height == grid["height"]
                and src.crs.to_string() == CRS
                and src.transform.almost_equals(grid["transform"]))


# --------------------------------------------------------------- terrain
def fetch_dem(grid):
    xmin, ymin, xmax, ymax = grid["bounds"]
    H, W = grid["height"], grid["width"]
    dem = np.full((H, W), np.nan, dtype="float32")
    n_tiles = -(-H // DEM_TILE) * -(-W // DEM_TILE)
    k = 0
    for r0 in range(0, H, DEM_TILE):
        for c0 in range(0, W, DEM_TILE):
            k += 1
            h, w = min(DEM_TILE, H - r0), min(DEM_TILE, W - c0)
            txmin = xmin + c0 * RES
            tymax = ymax - r0 * RES
            params = {
                "bbox": f"{txmin},{tymax - h * RES},{txmin + w * RES},{tymax}",
                "bboxSR": 32611, "imageSR": 32611, "size": f"{w},{h}",
                "format": "tiff", "pixelType": "F32",
                "interpolation": "RSP_BilinearInterpolation",
                "noDataInterpretation": "esriNoDataMatchAny", "f": "image",
            }
            r = requests.get(DEM_URL, params=params, timeout=300)
            r.raise_for_status()
            if r.content[:4] not in (b"II*\x00", b"MM\x00*"):
                sys.exit("3DEP did not return a GeoTIFF. Start of response:\n"
                         f"{r.content[:400]!r}")
            with MemoryFile(r.content) as mf, mf.open() as src:
                tile = src.read(1, masked=True).filled(np.nan).astype("float32")
            if tile.shape != (h, w):
                sys.exit(f"3DEP tile shape {tile.shape} != requested {(h, w)}")
            dem[r0:r0 + h, c0:c0 + w] = tile
            print(f"  DEM tile {k}/{n_tiles} ok ({h}x{w})")
    dem[(dem < -500) | (dem > 5000)] = np.nan  # catch fill values
    return dem


def slope_aspect(dem):
    """Horn (1981). Slope in degrees. Aspect = downslope direction,
    degrees clockwise from north; -1 where flat."""
    z = np.pad(dem, 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * RES)   # + = rises east
    dzdn = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * RES)   # + = rises north
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdn)) + 360.0) % 360.0
    aspect[(dzdx == 0) & (dzdn == 0)] = -1.0
    return slope.astype("float32"), aspect.astype("float32")


def hillshade(slope, aspect, az=315.0, alt=45.0):
    s, a = np.radians(slope), np.radians(aspect)
    hs = (np.sin(np.radians(alt)) * np.cos(s)
          + np.cos(np.radians(alt)) * np.sin(s) * np.cos(np.radians(az) - a))
    return np.clip(hs, 0, 1)


# ------------------------------------------------------------ perimeters
def burn(geoms, grid):
    return rasterize(((g, 1) for g in geoms if g is not None and not g.is_empty),
                     out_shape=(grid["height"], grid["width"]),
                     transform=grid["transform"], fill=0, dtype="uint8")


# -------------------------------------------------------------- landfire
def find_landfire(lf_dir):
    found, unmatched = {}, []
    if lf_dir is None or not lf_dir.exists():
        return found, unmatched
    for p in sorted(lf_dir.rglob("*.tif")):
        toks = p.stem.upper().replace("-", "_").split("_")
        hits = {LF_TOKENS[t] for t in toks if t in LF_TOKENS}
        if len(hits) == 1:
            key = hits.pop()
            if key in found:
                print(f"  WARNING: second {key} file ignored: {p.name}")
            else:
                found[key] = p
        else:
            unmatched.append(p.name)
    return found, unmatched


def landfire_to_grid(path, grid):
    out = np.full((grid["height"], grid["width"]), LF_NODATA, dtype="int16")
    with rasterio.open(path) as src:
        if src.count != 1:
            sys.exit(f"{path.name} has {src.count} bands; expected one product per file")
        info = {"file": path.name, "crs": src.crs.to_string(), "dtype": src.dtypes[0],
                "nodata": src.nodata, "res": list(src.res)}
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata if src.nodata is not None else LF_NODATA,
                  dst_transform=grid["transform"], dst_crs=CRS,
                  dst_nodata=LF_NODATA, resampling=Resampling.nearest)
    out[out < 0] = LF_NODATA   # LANDFIRE fill values (e.g. -10000 over ocean) are not valid codes
    return out, info


def fbfm40_groups(fb):
    cls = np.full(fb.shape, -1, dtype="int8")
    for k, (_, lo, hi, _) in enumerate(FBFM40_GROUPS):
        cls[(fb >= lo) & (fb <= hi)] = k
    return cls


# ------------------------------------------------------------------ plot
def plot_alignment(path, grid, hill, gdf, fb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    xmin, ymin, xmax, ymax = grid["bounds"]
    ext = (xmin, xmax, ymin, ymax)
    n = 2 if fb is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), squeeze=False)
    first, last = gdf.iloc[[0]], gdf.iloc[[-1]]

    ax = axes[0, 0]
    ax.imshow(hill, cmap="gray", extent=ext)
    ax.set_title("3DEP hillshade + perimeters")

    if fb is not None:
        ax2 = axes[0, 1]
        cls = np.ma.masked_less(fbfm40_groups(fb), 0)
        cmap = ListedColormap([g[3] for g in FBFM40_GROUPS])
        ax2.imshow(hill, cmap="gray", extent=ext)
        ax2.imshow(cls, cmap=cmap, vmin=0, vmax=len(FBFM40_GROUPS) - 1,
                   extent=ext, alpha=0.75, interpolation="nearest")
        ax2.set_title("LANDFIRE FBFM40 groups + perimeters")
        ax2.legend(handles=[Patch(color=g[3], label=g[0]) for g in FBFM40_GROUPS],
                   loc="lower left", fontsize=7)

    for ax in axes[0]:
        last.boundary.plot(ax=ax, color="yellow", linewidth=1.2)
        first.boundary.plot(ax=ax, color="red", linewidth=1.5)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("UTM 11N easting (m)"); ax.set_ylabel("northing (m)")
    fig.suptitle(f"fireID {int(gdf['fireID'].iloc[0])}: red = first record "
                 f"({gdf['time'].iloc[0]:%Y-%m-%d %H:%M} UTC), yellow = last "
                 f"({gdf['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def stats_line(name, arr, nodata=None):
    a = arr.astype("float64")
    bad = np.isnan(a) if nodata is None else (a == nodata)
    v = a[~bad]
    if v.size == 0:
        return f"{name:<14} ALL NODATA"
    return (f"{name:<14} min {v.min():9.2f}  max {v.max():9.2f}  "
            f"mean {v.mean():9.2f}  nodata {bad.mean():6.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire-id", type=int, default=4299)
    ap.add_argument("--landfire-dir", type=Path,
                    default=REPO / "data" / "raw" / "landfire" / "LF2016_Remap")
    args = ap.parse_args()
    fid = args.fire_id
    fdir = REPO / "data" / "processed" / "fires" / str(fid)
    fdir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # --- fire + grid
    gdf, layer = load_fire(fid)
    grid = make_grid(gdf)
    log(f"fireID {fid}: {len(gdf)} records from layer '{layer}', "
        f"{gdf['time'].iloc[0]} -> {gdf['time'].iloc[-1]}")
    log(f"Grid {CRS} @ {RES:.0f} m: {grid['width']} x {grid['height']} cells, "
        f"bounds {grid['bounds']}")
    lon_lat = transform_bounds(CRS, "EPSG:4326", *grid["bounds"], densify_pts=21)
    log("LANDFIRE download box (WGS84, already padded 0.01 deg): "
        f"W {lon_lat[0] - 0.01:.4f}  S {lon_lat[1] - 0.01:.4f}  "
        f"E {lon_lat[2] + 0.01:.4f}  N {lon_lat[3] + 0.01:.4f}")

    gj = {k: v for k, v in grid.items() if k != "transform"}
    gj["transform"] = list(grid["transform"])[:6]
    gj.update(fireID=fid, t_first=str(gdf["time"].iloc[0]),
              t_last=str(gdf["time"].iloc[-1]), n_records=len(gdf))
    (fdir / "grid.json").write_text(json.dumps(gj, indent=2))

    # --- DEM (cached)
    dem_path = fdir / "dem.tif"
    if dem_path.exists() and grid_matches(dem_path, grid):
        log("\nDEM: using cached dem.tif")
        with rasterio.open(dem_path) as src:
            dem = src.read(1, masked=True).filled(np.nan).astype("float32")
    else:
        log("\nDEM: downloading 3DEP onto grid ...")
        dem = fetch_dem(grid)
        write_tif(dem_path, np.nan_to_num(dem, nan=-9999.0), grid, -9999.0, "float32")
        dem[dem == -9999.0] = np.nan
    slope, aspect = slope_aspect(dem)
    write_tif(fdir / "slope.tif", np.nan_to_num(slope, nan=-9999.0), grid, -9999.0, "float32")
    write_tif(fdir / "aspect.tif", np.nan_to_num(aspect, nan=-9999.0), grid, -9999.0, "float32")
    log(stats_line("dem (m)", dem))
    log(stats_line("slope (deg)", slope))
    log(stats_line("aspect (deg)", np.where(aspect < 0, np.nan, aspect)))

    # --- perimeter masks
    m_first = burn([gdf.geometry.iloc[0]], grid)
    m_last = burn([gdf.geometry.iloc[-1]], grid)
    m_union = burn([gdf.geometry.union_all()], grid)
    for name, m in [("perim_first", m_first), ("perim_last", m_last), ("perim_union", m_union)]:
        write_tif(fdir / f"{name}.tif", m, grid, 0, "uint8")
    cell_km2 = RES * RES / 1e6
    inter = np.logical_and(m_last, m_union).sum()
    uni = np.logical_or(m_last, m_union).sum()
    log("\nPerimeters on grid:")
    log(f"  first record area  {m_first.sum() * cell_km2:9.2f} km2")
    log(f"  last record area   {m_last.sum() * cell_km2:9.2f} km2   "
        f"(farea last = {gdf['farea'].iloc[-1]:.2f}, farea max = {gdf['farea'].max():.2f})")
    log(f"  union of records   {m_union.sum() * cell_km2:9.2f} km2")
    log(f"  IoU(last, union) = {inter / uni:.3f}   "
        "(~1.0 means perimeters are cumulative)")
    log(f"  farea monotonic non-decreasing: {bool(gdf['farea'].is_monotonic_increasing)}")

    # --- LANDFIRE
    found, unmatched = find_landfire(args.landfire_dir)
    fb = None
    log(f"\nLANDFIRE dir: {args.landfire_dir}")
    if unmatched:
        log(f"  unmatched .tif files (ignored): {unmatched}")
    missing = [k for k in LF_LAYERS if k not in found]
    if missing:
        log(f"  missing layers: {missing}. Phase 2 skipped; "
            "download them for the box above and re-run.")
    else:
        for key in LF_LAYERS:
            arr, info = landfire_to_grid(found[key], grid)
            write_tif(fdir / f"{key}.tif", arr, grid, LF_NODATA, "int16")
            log(f"  {key:<7} <- {info['file']}  src CRS {info['crs'][:40]}  "
                f"dtype {info['dtype']}  nodata {info['nodata']}  res {info['res']}")
            log("  " + stats_line(key, arr, LF_NODATA))
            if key == "fbfm40":
                fb = arr
        inside = fb[m_union == 1]
        inside = inside[inside != LF_NODATA]
        vals, cnts = np.unique(inside, return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:10]
        log("  FBFM40 inside burned union (top 10 codes): "
            + ", ".join(f"{v}:{c / inside.size:.1%}" for c, v in top))
        nb = ((inside >= 91) & (inside <= 99)).mean() if inside.size else float("nan")
        log(f"  non-burnable fraction inside burned area: {nb:.1%}")
        log("  NOTE: LANDFIRE stores CH and CBH as m*10 and CBD as kg/m3*100; "
            "confirm in the product metadata before using them in Rothermel.")

    # --- plot + summary
    png = OUT_DIR / f"fire_{fid}_alignment.png"
    plot_alignment(png, grid, hillshade(np.nan_to_num(slope), aspect), gdf, fb)
    log(f"\nWrote rasters to {fdir.relative_to(REPO)}")
    log(f"Wrote {png.relative_to(REPO)}")
    (OUT_DIR / f"fire_{fid}_stack_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

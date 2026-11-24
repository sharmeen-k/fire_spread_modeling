"""30 m model grid, terrain, perimeter masks and LANDFIRE fuels for one fire (formerly script 07).

Outputs (data/processed/fires/<id>/):
  grid.json
  static/dem.tif slope.tif aspect.tif fbfm40.tif cc.tif ch.tif cbh.tif cbd.tif   (COG)
  perimeters/perim_first.tif perim_last.tif perim_union.tif obs_times.csv
  outputs/fire_<id>_stack_summary.txt, outputs/fire_<id>_alignment.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject, transform_bounds

from . import http, io, logs, manifest
from .config import Config
from .perimeters import load_records, obs_table

log = logging.getLogger(__name__)

DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
DEM_TILE = 2000  # pixels per request side
DEM_NODATA = -9999.0
LF_NODATA = -9999
# filename tokens -> layer key; handles old (LC16_F40_200) and new (LF2016_FBFM40_CONUS) naming
LF_TOKENS = {"F40": "fbfm40", "FBFM40": "fbfm40", "CC": "cc", "CH": "ch", "CBH": "cbh", "CBD": "cbd"}
# key -> (units, description, scale factor of the stored integers, categorical)
LF_LAYERS = {
    "fbfm40": ("1", "Scott & Burgan 40 fire behavior fuel model code (LANDFIRE)", None, True),
    "cc": ("%", "forest canopy cover (LANDFIRE)", None, False),
    "ch": ("m", "forest canopy height (LANDFIRE, stored as m x 10)", 0.1, False),
    "cbh": ("m", "canopy base height (LANDFIRE, stored as m x 10)", 0.1, False),
    "cbd": ("kg m-3", "canopy bulk density (LANDFIRE, stored as kg/m3 x 100)", 0.01, False),
}
FBFM40_GROUPS = [  # (label, lo, hi, colour)
    ("NB non-burnable", 91, 99, "#bdbdbd"),
    ("GR grass", 101, 109, "#f2e394"),
    ("GS grass-shrub", 121, 124, "#d9b26f"),
    ("SH shrub", 141, 149, "#8c6d31"),
    ("TU timber-understory", 161, 165, "#78a35a"),
    ("TL timber litter", 181, 189, "#2f6b3a"),
    ("SB slash-blowdown", 201, 204, "#a14b3a"),
]


# --------------------------------------------------------------------- grid
def make_grid(gdf: gpd.GeoDataFrame, cfg: Config) -> io.Grid:
    p = cfg.project
    xmin, ymin, xmax, ymax = gdf.to_crs(p.crs).total_bounds
    xmin = np.floor((xmin - p.buffer_m) / p.snap_m) * p.snap_m
    ymin = np.floor((ymin - p.buffer_m) / p.snap_m) * p.snap_m
    xmax = np.ceil((xmax + p.buffer_m) / p.snap_m) * p.snap_m
    ymax = np.ceil((ymax + p.buffer_m) / p.snap_m) * p.snap_m
    return io.Grid(
        p.crs,
        p.resolution_m,
        (float(xmin), float(ymin), float(xmax), float(ymax)),
        int(round((xmax - xmin) / p.resolution_m)),
        int(round((ymax - ymin) / p.resolution_m)),
    )


# ------------------------------------------------------------------ terrain
def fetch_dem(grid: io.Grid) -> np.ndarray:
    """3DEP elevation resampled server-side onto the grid, in tiles."""
    xmin, _, _, ymax = grid.bounds
    H, W, res = grid.height, grid.width, grid.res
    epsg = rasterio.crs.CRS.from_string(grid.crs).to_epsg()
    dem = np.full((H, W), np.nan, dtype="float32")
    n_tiles = -(-H // DEM_TILE) * -(-W // DEM_TILE)
    k = 0
    for r0 in range(0, H, DEM_TILE):
        for c0 in range(0, W, DEM_TILE):
            k += 1
            h, w = min(DEM_TILE, H - r0), min(DEM_TILE, W - c0)
            tx, ty = xmin + c0 * res, ymax - r0 * res
            params = {
                "bbox": f"{tx},{ty - h * res},{tx + w * res},{ty}",
                "bboxSR": epsg,
                "imageSR": epsg,
                "size": f"{w},{h}",
                "format": "tiff",
                "pixelType": "F32",
                "interpolation": "RSP_BilinearInterpolation",
                "noDataInterpretation": "esriNoDataMatchAny",
                "f": "image",
            }
            r = http.get(DEM_URL, params=params, timeout=300)
            if r.content[:4] not in (b"II*\x00", b"MM\x00*"):
                raise RuntimeError(f"3DEP did not return a GeoTIFF: {r.content[:300]!r}")
            with MemoryFile(r.content) as mf, mf.open() as src:
                tile = src.read(1, masked=True).filled(np.nan).astype("float32")
            if tile.shape != (h, w):
                raise RuntimeError(f"3DEP tile shape {tile.shape} != requested {(h, w)}")
            dem[r0 : r0 + h, c0 : c0 + w] = tile
            log.info("  DEM tile %d/%d ok (%dx%d)", k, n_tiles, h, w)
    dem[(dem < -500) | (dem > 5000)] = np.nan
    return dem


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981). Slope in degrees; aspect = downslope direction, degrees
    clockwise from north, -1 where flat."""
    z = np.pad(dem, 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    dzdn = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdn)) + 360.0) % 360.0
    aspect[(dzdx == 0) & (dzdn == 0)] = -1.0
    return slope.astype("float32"), aspect.astype("float32")


def hillshade(slope, aspect, az=315.0, alt=45.0):
    s, a = np.radians(slope), np.radians(aspect)
    hs = np.sin(np.radians(alt)) * np.cos(s) + np.cos(np.radians(alt)) * np.sin(s) * np.cos(
        np.radians(az) - a
    )
    return np.clip(hs, 0, 1)


# --------------------------------------------------------------- landfire
def find_landfire(lf_dir: Path | None) -> tuple[dict[str, Path], list[str]]:
    found, unmatched = {}, []
    if lf_dir is None or not lf_dir.exists():
        return found, unmatched
    for p in sorted(lf_dir.rglob("*.tif")):
        toks = p.stem.upper().replace("-", "_").split("_")
        hits = {LF_TOKENS[t] for t in toks if t in LF_TOKENS}
        if len(hits) == 1:
            key = hits.pop()
            if key in found:
                log.warning("second %s file ignored: %s", key, p.name)
            else:
                found[key] = p
        else:
            unmatched.append(p.name)
    return found, unmatched


def landfire_to_grid(path: Path, grid: io.Grid) -> tuple[np.ndarray, dict]:
    out = np.full(grid.shape, LF_NODATA, dtype="int16")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path.name} has {src.count} bands; expected one product per file")
        info = {
            "file": path.name,
            "crs": src.crs.to_string(),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "res": list(src.res),
        }
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else LF_NODATA,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=LF_NODATA,
            resampling=Resampling.nearest,
        )
    out[out < 0] = LF_NODATA  # LANDFIRE fill values (e.g. -10000 over the ocean) are not valid codes
    return out, info


def fbfm40_groups(fb):
    cls = np.full(fb.shape, -1, dtype="int8")
    for k, (_, lo, hi, _) in enumerate(FBFM40_GROUPS):
        cls[(fb >= lo) & (fb <= hi)] = k
    return cls


# --------------------------------------------------------------------- plot
def plot_alignment(path, grid, hill, gdf, fb):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    xmin, ymin, xmax, ymax = grid.bounds
    ext = (xmin, xmax, ymin, ymax)
    n = 2 if fb is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), squeeze=False)
    g = gdf.to_crs(grid.crs)
    first, last = g.iloc[[0]], g.iloc[[-1]]
    axes[0, 0].imshow(hill, cmap="gray", extent=ext)
    axes[0, 0].set_title("3DEP hillshade + perimeters")
    if fb is not None:
        ax2 = axes[0, 1]
        cls = np.ma.masked_less(fbfm40_groups(fb), 0)
        cmap = ListedColormap([x[3] for x in FBFM40_GROUPS])
        ax2.imshow(hill, cmap="gray", extent=ext)
        ax2.imshow(
            cls,
            cmap=cmap,
            vmin=0,
            vmax=len(FBFM40_GROUPS) - 1,
            extent=ext,
            alpha=0.75,
            interpolation="nearest",
        )
        ax2.set_title("LANDFIRE FBFM40 groups + perimeters")
        ax2.legend(
            handles=[Patch(color=x[3], label=x[0]) for x in FBFM40_GROUPS], loc="lower left", fontsize=7
        )
    for ax in axes[0]:
        last.boundary.plot(ax=ax, color="yellow", linewidth=1.2)
        first.boundary.plot(ax=ax, color="red", linewidth=1.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("UTM 11N easting (m)")
        ax.set_ylabel("northing (m)")
    fig.suptitle(
        f"fireID {int(gdf['fireID'].iloc[0])}: red = first record "
        f"({gdf['obs_time_utc'].iloc[0]:%Y-%m-%d %H:%M} UTC), yellow = last "
        f"({gdf['obs_time_utc'].iloc[-1]:%Y-%m-%d %H:%M} UTC)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _stats(name, arr):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return f"{name:<14} ALL NODATA"
    return (
        f"{name:<14} min {v.min():9.2f}  max {v.max():9.2f}  mean {v.mean():9.2f}  "
        f"nodata {1 - v.size / arr.size:6.2%}"
    )


# --------------------------------------------------------------------- main
def run(cfg: Config, fire_id: int, landfire_dir: Path | None = None) -> None:
    log_ = logs.setup(cfg.outputs_dir / "logs", "stack", fire_id)
    rep = logs.Report(log_)
    fdir = cfg.fire_dir(fire_id)
    sdir, pdir = fdir / "static", fdir / "perimeters"

    gdf = load_records(cfg.socal_perimeters, fire_id)
    grid = make_grid(gdf, cfg)
    year = int(gdf["obs_time_utc"].iloc[0].year)
    lf_dir = landfire_dir or cfg.landfire_dir(year)
    rep(
        f"fireID {fire_id}: {len(gdf)} records, {gdf['obs_time_utc'].iloc[0]} -> {gdf['obs_time_utc'].iloc[-1]}"
    )
    rep(f"Grid {grid.crs} @ {grid.res:.0f} m: {grid.width} x {grid.height} cells, bounds {list(grid.bounds)}")
    ll = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    rep(
        f"LANDFIRE download box (WGS84, padded 0.01 deg): W {ll[0] - 0.01:.4f}  S {ll[1] - 0.01:.4f}  "
        f"E {ll[2] + 0.01:.4f}  N {ll[3] + 0.01:.4f}"
    )
    grid.to_json(
        fdir / "grid.json",
        fireID=fire_id,
        t_first_utc=io.iso_z(gdf["obs_time_utc"].iloc[0]),
        t_last_utc=io.iso_z(gdf["obs_time_utc"].iloc[-1]),
        n_records=len(gdf),
        landfire_dir=str(lf_dir),
    )

    # observation times
    io.write_table(obs_table(gdf), pdir / "obs_times.csv")

    # DEM (cached; also reuses a DEM from the old script-07 layout)
    dem_path = sdir / "dem.tif"
    reuse = next((p for p in (dem_path, fdir / "dem.tif") if p.exists() and grid.matches(p)), None)
    if reuse is not None:
        rep(f"\nDEM: reusing {reuse.relative_to(cfg.root)}")
        dem = io.read_raster(reuse)[0].astype("float32")
    else:
        rep("\nDEM: downloading 3DEP onto grid ...")
        dem = fetch_dem(grid)
    if reuse != dem_path:
        io.write_raster(
            dem_path,
            np.nan_to_num(dem, nan=DEM_NODATA),
            grid,
            nodata=DEM_NODATA,
            dtype="float32",
            units="m",
            description="elevation (USGS 3DEP)",
        )
        if reuse is None:
            manifest.record(
                cfg.data_dir,
                source="USGS 3DEP ImageServer",
                url=DEM_URL,
                path=dem_path,
                step="stack",
                fire_id=fire_id,
            )
    slope, aspect = slope_aspect(dem, grid.res)
    io.write_raster(
        sdir / "slope.tif",
        np.nan_to_num(slope, nan=DEM_NODATA),
        grid,
        nodata=DEM_NODATA,
        dtype="float32",
        units="degree",
        description="slope (Horn 1981)",
    )
    io.write_raster(
        sdir / "aspect.tif",
        np.nan_to_num(aspect, nan=DEM_NODATA),
        grid,
        nodata=DEM_NODATA,
        dtype="float32",
        units="degree",
        description="aspect: downslope direction clockwise from north; -1 = flat",
    )
    rep(_stats("dem (m)", dem))
    rep(_stats("slope (deg)", slope))
    rep(_stats("aspect (deg)", np.where(aspect < 0, np.nan, aspect)))

    # perimeter masks
    g = gdf.to_crs(grid.crs)

    def burn(geoms):
        return rasterize(
            ((x, 1) for x in geoms if x is not None and not x.is_empty),
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0,
            dtype="uint8",
        )

    masks = {
        "perim_first": burn([g.geometry.iloc[0]]),
        "perim_last": burn([g.geometry.iloc[-1]]),
        "perim_union": burn([g.geometry.union_all()]),
    }
    for name, m in masks.items():
        io.write_raster(
            pdir / f"{name}.tif",
            m,
            grid,
            nodata=None,
            dtype="uint8",
            units="1",
            description=f"{name.replace('_', ' ')} mask (1 = burned)",
            categorical=True,
        )
    cell_km2 = grid.res**2 / 1e6
    ml, mu = masks["perim_last"], masks["perim_union"]
    rep("\nPerimeters on grid:")
    rep(f"  first record area  {masks['perim_first'].sum() * cell_km2:9.2f} km2")
    rep(
        f"  last record area   {ml.sum() * cell_km2:9.2f} km2   (farea last = {gdf['farea'].iloc[-1]:.2f}, "
        f"max = {gdf['farea'].max():.2f})"
    )
    rep(f"  union of records   {mu.sum() * cell_km2:9.2f} km2")
    rep(
        f"  IoU(last, union) = {np.logical_and(ml, mu).sum() / np.logical_or(ml, mu).sum():.3f}   "
        "(~1.0 means perimeters are cumulative)"
    )
    rep(f"  farea monotonic non-decreasing: {bool(gdf['farea'].is_monotonic_increasing)}")

    # LANDFIRE
    found, unmatched = find_landfire(lf_dir)
    fb = None
    rep(f"\nLANDFIRE dir: {lf_dir}")
    if unmatched:
        rep(f"  unmatched .tif files (ignored): {unmatched}")
    missing = [k for k in LF_LAYERS if k not in found]
    if missing:
        rep(f"  missing layers: {missing}. Fuels skipped; download them for the box above and re-run.")
    else:
        for key, (units, desc, scale, categorical) in LF_LAYERS.items():
            arr, info = landfire_to_grid(found[key], grid)
            io.write_raster(
                sdir / f"{key}.tif",
                arr,
                grid,
                nodata=LF_NODATA,
                dtype="int16",
                units=units,
                description=desc,
                scale=scale,
                categorical=categorical,
            )
            phys = np.where(arr == LF_NODATA, np.nan, arr * (scale or 1.0))
            rep(f"  {key:<7} <- {info['file']}  src CRS {info['crs'][:40]}  nodata {info['nodata']}")
            rep("  " + _stats(f"{key} ({units})", phys))
            if key == "fbfm40":
                fb = arr
        inside = fb[mu == 1]
        missing_share = float((inside == LF_NODATA).mean()) if inside.size else 1.0
        if missing_share > 0.5:
            raise RuntimeError(
                f"LANDFIRE has no data for {missing_share:.0%} of fire {fire_id}'s burned area; the files in "
                f"{lf_dir} do not cover this fire (a clip for another area?). Use full-extent CONUS files."
            )
        inside = inside[inside != LF_NODATA]
        vals, cnts = np.unique(inside, return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:10]
        rep(
            "  FBFM40 inside burned union (top 10 codes): "
            + ", ".join(f"{v}:{c / inside.size:.1%}" for c, v in top)
        )
        nb = ((inside >= 91) & (inside <= 99)).mean() if inside.size else float("nan")
        rep(f"  non-burnable fraction inside burned area: {nb:.1%}")

    png = cfg.outputs_dir / f"fire_{fire_id}_alignment.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    plot_alignment(png, grid, hillshade(np.nan_to_num(slope), aspect), gdf, fb)
    rep(f"\nWrote rasters to {fdir.relative_to(cfg.root)}; figure {png.relative_to(cfg.root)}")
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_stack_summary.txt")

"""Source inspection, SoCal filtering and test-fire matching (formerly scripts 01-04)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from . import io, logs
from .config import Config
from .perimeters import label_to_utc


# ------------------------------------------------------------------ inspect
def _describe(src: Path, rep: logs.Report) -> None:
    for name, geom_type in pyogrio.list_layers(src):
        info = pyogrio.read_info(src, layer=name)
        rep(f"\n--- {src.name} | layer: {name} | geometry: {geom_type} | rows: {info['features']} ---")
        rep(f"CRS: {info['crs']}")
        sample = gpd.read_file(src, layer=name, rows=3)
        for col, dtype in sample.dtypes.items():
            ex = "" if col == "geometry" else f"  e.g. {sample[col].iloc[0] if len(sample) else None}"
            rep(f"  {col:30s} {dtype}{ex}")


def _safe_extract_tar(tar_path: Path, out: Path) -> None:
    with tarfile.open(tar_path) as t:
        t.extractall(out, filter="data")


def _safe_extract_zip(zip_path: Path, out: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            if Path(m).is_absolute() or ".." in Path(m).parts:
                raise ValueError(f"unsafe path in {zip_path.name}: {m}")
        z.extractall(out)


def extract_sources(cfg: Config) -> dict[str, Path]:
    """Extract the manually obtained archives once into data/extracted/."""
    out = {}
    tar_path = cfg.path(cfg.inputs.largefire_archive)
    d = cfg.extracted_dir / "Largefire"
    if not d.exists():
        d.mkdir(parents=True)
        _safe_extract_tar(tar_path, d)
    out["largefire"] = d
    for key, zp in (("counties", cfg.inputs.counties_zip), ("frap", cfg.inputs.frap_zip)):
        d = cfg.extracted_dir / key
        if not d.exists():
            d.mkdir(parents=True)
            _safe_extract_zip(cfg.path(zp), d)
        out[key] = d
    return out


def largefire_gpkg(cfg: Config) -> Path:
    return next((cfg.extracted_dir / "Largefire").rglob("LargeFires_2012-2020.gpkg"))


def inspect_sources(cfg: Config) -> None:
    log = logs.setup(cfg.outputs_dir / "logs", "inspect")
    dirs = extract_sources(cfg)

    rep = logs.Report(log)
    gpkgs = sorted(dirs["largefire"].rglob("*.gpkg"))
    rep(f"Archive: {cfg.inputs.largefire_archive}")
    rep(f"GeoPackage files found: {len(gpkgs)}")
    for p in gpkgs:
        rep(f"  {p.relative_to(dirs['largefire'])}")
    _describe(gpkgs[0], rep)
    rep.save(cfg.outputs_dir / "largefire_schema.txt")

    rep = logs.Report(log)
    for key in ("counties", "frap"):
        sources = sorted(dirs[key].rglob("*.shp")) + sorted(dirs[key].rglob("*.gdb"))
        rep(f"\n##### {key}: {[s.name for s in sources]}")
        for src in sources:
            _describe(src, rep)
    rep.save(cfg.outputs_dir / "boundaries_schema.txt")


# ------------------------------------------------------------- SoCal filter
def filter_socal(cfg: Config) -> None:
    log = logs.setup(cfg.outputs_dir / "logs", "socal")
    extract_sources(cfg)
    gpkg = largefire_gpkg(cfg)
    shp = next((cfg.extracted_dir / "counties").rglob("*.shp"))

    counties = gpd.read_file(shp).to_crs(4326)
    socal = counties[counties["NAME"].isin(cfg.study.counties)]
    if len(socal) != len(cfg.study.counties):
        raise ValueError(f"expected {len(cfg.study.counties)} counties, got {len(socal)}")
    socal_union = socal.union_all()

    attrs = pyogrio.read_dataframe(
        gpkg,
        layer="perimeter",
        read_geometry=False,
        columns=["fireID", "time", "clat", "clon", "farea", "year"],
    )
    attrs["label_time"] = pd.to_datetime(attrs["time"])
    attrs["obs_time_utc"] = label_to_utc(attrs["label_time"], attrs["clon"])
    attrs = attrs.sort_values(["fireID", "obs_time_utc"])

    fires = (
        attrs.groupby("fireID")
        .agg(
            start_utc=("obs_time_utc", "min"),
            end_utc=("obs_time_utc", "max"),
            n_records=("obs_time_utc", "size"),
            max_farea=("farea", "max"),
            start_clat=("clat", "first"),
            start_clon=("clon", "first"),
        )
        .reset_index()
    )
    pts = gpd.GeoSeries(gpd.points_from_xy(fires["start_clon"], fires["start_clat"]), crs=4326)
    socal_fires = fires[pts.within(socal_union).to_numpy()].reset_index(drop=True)

    peri = gpd.read_file(gpkg, layer="perimeter")
    peri = peri[peri["fireID"].isin(socal_fires["fireID"])]
    cfg.socal_perimeters.parent.mkdir(parents=True, exist_ok=True)
    peri.to_file(cfg.socal_perimeters, layer="perimeter", driver="GPKG")
    io.write_table(socal_fires, cfg.outputs_dir / "socal_fires.csv")

    hrrr0 = pd.Timestamp(cfg.study.hrrr_start, tz="UTC")
    log.info("All Largefire fires:        %d  (%d records)", len(fires), len(attrs))
    log.info("SoCal fires (%d counties):  %d  (%d records)", len(socal), len(socal_fires), len(peri))
    log.info(
        "  starting on/after %s (HRRR archive): %d", hrrr0.date(), (socal_fires["start_utc"] >= hrrr0).sum()
    )
    log.info(
        "SoCal fires per start year:\n%s",
        socal_fires["start_utc"].dt.year.value_counts().sort_index().to_string(),
    )


# --------------------------------------------------------------- test fires
def find_test_fires(cfg: Config) -> None:
    log = logs.setup(cfg.outputs_dir / "logs", "test_fires")
    extract_sources(cfg)
    window = pd.Timedelta(days=cfg.study.match_window_days)
    fires = io.read_table(cfg.outputs_dir / "socal_fires.csv", ["start_utc", "end_utc"])
    peri = gpd.read_file(cfg.socal_perimeters, layer="perimeter")
    peri["label_time"] = pd.to_datetime(peri["time"])
    # final (cumulative) perimeter per fire, California Albers for area math
    last = peri.sort_values("label_time").groupby("fireID").tail(1).set_index("fireID").to_crs(3310)
    last["geometry"] = last.geometry.make_valid()

    gdb = next((cfg.extracted_dir / "frap").rglob("*.gdb"))
    layer = next(n for n, _ in pyogrio.list_layers(gdb) if n.startswith("firep"))
    frap = gpd.read_file(gdb, layer=layer).to_crs(3310)

    rows = []
    for name, year in cfg.study.frap_targets.items():
        cand = frap[(frap["FIRE_NAME"].str.upper() == name) & (frap["YEAR_"] == year)]
        if cand.empty:
            log.warning("No FRAP match for %s %s", name, year)
            continue
        fp = cand.sort_values("GIS_ACRES", ascending=False).iloc[0]
        fgeom = gpd.GeoSeries([fp.geometry], crs=3310).make_valid().iloc[0]
        alarm = io.to_utc([fp["ALARM_DATE"]])[0]
        log.info("FRAP %s %s: alarm %s, %.0f acres", name, year, alarm.date(), fp["GIS_ACRES"])
        near = fires[(fires["start_utc"] >= alarm - window) & (fires["start_utc"] <= alarm + window)]
        for r in near.itertuples():
            g = last.loc[r.fireID, "geometry"]
            union = g.union(fgeom).area
            iou = g.intersection(fgeom).area / union if union else 0.0
            rows.append(
                {
                    "target": name,
                    "fireID": r.fireID,
                    "iou": round(iou, 3),
                    "start_utc": r.start_utc,
                    "n_records": int(r.n_records),
                }
            )

    out = pd.DataFrame(rows, columns=["target", "fireID", "iou", "start_utc", "n_records"])
    out = out.sort_values(["target", "iou"], ascending=[True, False])
    io.write_table(out, cfg.outputs_dir / "test_fire_candidates.csv")
    log.info("Top candidates:\n%s", out.groupby("target").head(3).to_string(index=False))

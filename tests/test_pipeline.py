"""End-to-end run of the whole pipeline on the synthetic study area (offline)."""

import numpy as np
import pandas as pd
import pytest

from firespread import cli, io, manifest
from firespread.fuel_moisture import cube_path as fm_path
from firespread.weather import cube_path as wx_path

from . import fakes


def run(*args):
    assert cli.main(list(args)) == 0, f"firespread {' '.join(args)} failed"


@pytest.fixture
def catalog(cfg, web):
    run("inspect")
    run("socal")
    run("test-fires")
    run("split")
    return cfg


def test_catalog_and_split(catalog):
    cfg = catalog
    out = cfg.outputs_dir
    assert "perimeter" in (out / "largefire_schema.txt").read_text()
    assert "firep25_1" in (out / "boundaries_schema.txt").read_text()
    fires = io.read_table(out / "socal_fires.csv", ["start_utc"])
    assert 5104 not in set(fires.fireID)  # Fresno fire filtered out
    assert str(fires["start_utc"].dt.tz) == "UTC"
    cand = pd.read_csv(out / "test_fire_candidates.csv")
    best = cand.sort_values("iou", ascending=False).groupby("target").head(1).set_index("target")
    assert best.loc["THOMAS", "fireID"] == 4299 and best.loc["THOMAS", "iou"] > 0.8
    assert best.loc["WOOLSEY", "fireID"] == 3922
    split = pd.read_csv(out / "split.csv").set_index("fireID")
    assert split.loc[4299, "split"] == "test" and split.loc[3922, "split"] == "test"
    assert split.loc[5101, "reason"] == "single_record"
    assert split.loc[5102, "reason"] == "pre_HRRR"
    assert split.loc[5103, "reason"] == "concurrent_with_test"
    assert split.loc[5105, "reason"] == "long_duration"
    assert (split["split"] == "val").sum() == 3


def test_fire_pipeline(catalog, web, fake_herbie, fake_nfdrs4):
    cfg = catalog
    fid = 4299
    run("stack", "--fire-id", str(fid))
    fdir = cfg.fire_dir(fid)
    import rasterio

    for name in ["dem", "slope", "aspect", "fbfm40", "cc", "ch", "cbh", "cbd"]:
        with rasterio.open(fdir / "static" / f"{name}.tif") as src:
            assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
    ch, meta = io.read_raster(fdir / "static" / "ch.tif")
    assert meta["scale"] == pytest.approx(0.1) and np.nanmax(ch) < 20.0  # metres, not m x 10
    fb, _ = io.read_raster(fdir / "static" / "fbfm40.tif", physical=False)
    assert (fb == -9999).any() and not (fb == -10000).any()  # ocean fill masked
    obs = io.read_table(fdir / "perimeters" / "obs_times.csv", ["obs_time_utc"])
    assert obs["obs_time_utc"].iloc[0] == pd.Timestamp("2017-12-05 09:26", tz="UTC").round("min") or abs(
        obs["obs_time_utc"].iloc[0] - pd.Timestamp("2017-12-05 09:26", tz="UTC")
    ) < pd.Timedelta(minutes=2)

    # weather: all hours, spin-up, rotation, gap filling and QC flags
    run("weather", "--fire-id", str(fid))
    wx = io.read_cube(wx_path(cfg, fid))
    assert wx.attrs["Conventions"] == "CF-1.10"
    t = pd.DatetimeIndex(wx["time"].values)
    assert (t[1:] - t[:-1] == pd.Timedelta(hours=1)).all()  # continuous
    assert pd.Timestamp(wx.attrs["spinup_start_utc"]) < pd.Timestamp(wx.attrs["analysis_start_utc"])
    filled00 = {str(x)[:16].replace("T", " ") for x in t[wx["qc_filled_f00"].values == 1]}
    assert filled00 == fakes.MISSING_F00
    assert int(wx["qc_filled_f01"].sum()) == len(fakes.MISSING_F01)
    off = wx.sel(time="2017-12-06 12:00")
    assert float(np.median(off["wdir10"])) == pytest.approx(45.0, abs=0.5)  # rotated back to NE
    for v in ["u10", "v10", "t2m", "rh2", "dswrf", "apcp"]:
        assert np.isfinite(wx[v].values).all(), v
    assert float(wx["apcp"].sum()) > 0 and int(wx["land"].sum()) > 0

    # stations: ASOS + RAWS, filled hours excluded
    import os

    os.environ["SYNOPTIC_TOKEN"] = "test-token"
    try:
        run("stations", "--fire-id", str(fid))
    finally:
        del os.environ["SYNOPTIC_TOKEN"]
    met = pd.read_csv(cfg.outputs_dir / f"fire_{fid}_hrrr_station_metrics.csv")
    assert {"ASOS", "RAWS"} <= set(met.network)
    pairs = io.read_table(cfg.outputs_dir / f"fire_{fid}_hrrr_station_pairs.csv", ["time"])
    assert pd.Timestamp("2017-12-05 20:00", tz="UTC") not in set(pairs["time"])
    st = pd.read_csv(cfg.outputs_dir / f"fire_{fid}_stations.csv").set_index("stid")
    assert "LAX" not in st.index or st.loc["LAX", "n_hours"] == 0
    assert st.loc["WLYC1", "elev_m"] == pytest.approx(2700 * 0.3048)
    assert set(st["terrain_src"].dropna()) <= {"grid", "3dep", ""}
    srcs = {m["source"] for m in manifest.read(cfg.data_dir)}
    assert {"NOAA HRRR (Herbie)", "IEM ASOS", "Synoptic RAWS", "USGS 3DEP ImageServer"} <= srcs
    assert all("test-token" not in m["url"] for m in manifest.read(cfg.data_dir))  # secret redacted

    # fuel moisture (fake NFDRS4 physics; the wiring is what is tested here)
    run("fuel-moisture", "--fire-id", str(fid))
    fm = io.read_cube(fm_path(cfg, fid))
    assert pd.Timestamp(fm["time"].values[0]) == pd.Timestamp(wx.attrs["analysis_start_utc"]).tz_convert(None)
    land = wx["land"].values.astype(bool)
    for c in (1, 10, 100, 1000):
        v = fm[f"fm{c}"].values
        assert np.isfinite(v[:, land]).all() and np.isnan(v[:, ~land]).all()
    dry = float(np.nanmedian(fm["fm1"].sel(time="2017-12-05 21:00").values))
    assert dry < 6.0
    assert "fm100_gridmet" in fm and fm["fm100_gridmet"].dims == ("day", "y", "x")
    assert (
        "Rejected NFDRS4 updates: 0"
        in (cfg.outputs_dir / f"fire_{fid}_fuel_moisture_summary.txt").read_text()
    )
    assert any("reacchpna" in c for c in web.calls) and any("northwestknowledge" in c for c in web.calls)

    # re-running weather is fully cached (no new HRRR requests)
    fakes.FakeHerbie.calls.clear()
    run("weather", "--fire-id", str(fid))
    assert fakes.FakeHerbie.calls == []

    # publishing prepares real UTC times (no ArcGIS sign-in in tests)
    run("publish", "--fire-id", str(fid), "--dry-run")
    import geopandas as gpd

    pub = gpd.read_file(fdir / "publish" / f"fire_{fid}_observed.gpkg")
    assert (pd.to_datetime(pub["end_time"]) > pd.to_datetime(pub["obs_time"])).all()


def test_raws_bad_token_is_skipped(catalog, web, fake_herbie, cfg, monkeypatch):
    from firespread import stack, stations, weather

    fid = 4299
    if not (cfg.fire_dir(fid) / "weather" / "hrrr_hourly.zarr").exists():
        stack.run(cfg, fid)
        weather.run(cfg, fid)
    web.synoptic_status = 401
    monkeypatch.setenv("SYNOPTIC_TOKEN", "bad")
    met = stations.run(cfg, fid, networks=("asos", "raws"), refresh=True)
    assert set(met.network) == {"ASOS"}


def test_legacy_caches_are_reused(cfg, fake_herbie, tmp_path):
    """A script-08 hourly cache file (old layout) is read without re-downloading."""
    from firespread.weather import HrrrCache

    box = (-119.6, 34.0, -118.6, 34.8)
    c = HrrrCache(cfg, 9999, box)
    t = pd.Timestamp("2017-12-05 12:00")
    f, st = c.get("f00", t)
    assert st == "downloaded" and f.name == "2017120512.nc"
    f2, st2 = c.get("f00", t)
    assert st2 == "cached" and f2 == f


def test_cli_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "fuel-moisture" in capsys.readouterr().out

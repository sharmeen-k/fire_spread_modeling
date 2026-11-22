"""End-to-end run of the whole pipeline on the synthetic study area (offline)."""

import sys

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

    # the spread model's 100-h source is a config switch
    from firespread.fuel_moisture import surface_dead_moisture

    nel = surface_dead_moisture(cfg, fid)
    assert set(nel.data_vars) == {"fm1", "fm10", "fm100"} and nel.attrs["fm100_source"] == "nelson"
    np.testing.assert_array_equal(nel["fm100"].values, fm["fm100"].values)
    cfg.fuel_moisture.fm100_source = "gridmet"
    try:
        gm = surface_dead_moisture(cfg, fid)
    finally:
        cfg.fuel_moisture.fm100_source = "nelson"
    assert gm.attrs["fm100_source"] == "gridmet"
    np.testing.assert_array_equal(gm["fm1"].values, nel["fm1"].values)
    t = pd.Timestamp("2017-12-06 05:00")  # 21:00 PST on Dec 5 -> gridMET day Dec 5
    day_val = fm["fm100_gridmet"].sel(day="2017-12-05").values
    np.testing.assert_allclose(gm["fm100"].sel(time=t).values, day_val)

    # live fuel moisture (fake NFDRS4 physics; the wiring is tested)
    run("live-fuel-moisture", "--fire-id", str(fid))
    from firespread.live_fuel_moisture import cube_path as lfm_path

    lf = io.read_cube(lfm_path(cfg, fid))
    lt = pd.DatetimeIndex(lf["time"].values)
    assert (lt.hour == 21).all() and (lt[1:] - lt[:-1] == pd.Timedelta(days=1)).all()
    assert len(lt) >= 366 and lt[-1] >= pd.Timestamp("2017-12-06")
    herb, wood = lf["lfm_herb"].values[:, land], lf["lfm_woody"].values[:, land]
    assert np.isfinite(herb).all() and np.isfinite(wood).all()
    assert herb.min() >= 30 and wood.min() >= 60 and np.isnan(lf["lfm_herb"].values[:, ~land]).all()
    # wet winter > dry autumn in the synthetic climate
    assert (
        lf["lfm_woody"].sel(time="2017-02-15").values[0][land].mean()
        > lf["lfm_woody"].sel(time="2017-11-15").values[0][land].mean()
    )
    assert (
        "On the first fire day"
        in (cfg.outputs_dir / f"fire_{fid}_live_fuel_moisture_summary.txt").read_text()
    )

    # live fuel moisture vs (synthetic) Globe-LFMC chamise samples
    run("lfmc-check", "--fire-id", str(fid))
    chk = io.read_table(cfg.outputs_dir / f"fire_{fid}_lfmc_check.csv", ["date"])
    assert set(chk["site"]) == {"Ventura Hills", "Ojai Valley"}  # far, wrong-species and foreign rows dropped
    assert (chk["n_samples"] == 2).all() and chk["model_woody"].notna().all()
    assert (cfg.processed_dir / "lfmc" / "globe_lfmc_us.csv").exists()
    txt = (cfg.outputs_dir / f"fire_{fid}_lfmc_check_summary.txt").read_text()
    assert "By site" in txt and "Mar-May" in txt

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


@pytest.mark.skipif(sys.platform == "win32", reason="worker processes can't see the fake Herbie on Windows")
def test_parallel_downloads(cfg, fake_herbie):
    """The process-pool path fetches every hour and the parent writes the manifest."""
    from firespread.weather import HrrrCache, download

    cfg.hrrr.workers = 3
    box = (-119.6, 34.0, -118.6, 34.8)
    hours = pd.date_range("2017-12-05 00:00", periods=8, freq="h")
    cache = HrrrCache(cfg, 9998, box, record=False)
    before = len(manifest.read(cfg.data_dir))
    got, status = download(cfg, 9998, box, cache, hours, retry_missing=False)
    assert len(got["f00"]) == 8 and len(got["f01"]) == 8
    assert all(v == "downloaded" for v in status["f00"].values())
    assert len(manifest.read(cfg.data_dir)) - before == 16
    _, status2 = download(cfg, 9998, box, cache, hours, retry_missing=False)
    assert all(v == "cached" for v in status2["f01"].values())


def test_cli_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "fuel-moisture" in capsys.readouterr().out

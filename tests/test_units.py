"""Unit tests for the shared conventions and the physics helpers."""

import numpy as np
import pandas as pd
import pytest

from firespread import io
from firespread.perimeters import label_to_utc
from firespread.stack import slope_aspect
from firespread.stations import circ_diff, metrics
from firespread.weather import rotate_to_earth, wind_dir_from


def test_label_to_utc_thomas_first_record():
    # Thomas: first record labelled 2017-12-05 00:00 at clon -119.1 -> ~09:26 UTC
    obs = label_to_utc(
        pd.Series(pd.to_datetime(["2017-12-05 00:00", "2017-12-05 12:00"])), pd.Series([-119.1, -119.1])
    )
    assert str(obs.dt.tz) == "UTC"
    assert obs.iloc[0] == pd.Timestamp("2017-12-05 09:26:24", tz="UTC")
    assert (obs.iloc[1] - obs.iloc[0]) == pd.Timedelta(hours=12)


def test_label_to_utc_rejects_other_hours():
    with pytest.raises(ValueError):
        label_to_utc(pd.Series(pd.to_datetime(["2017-12-05 06:00"])), pd.Series([-119.0]))


def test_rotation_angle_matches_lambert_formula():
    ue, ve, alpha = rotate_to_earth(np.array([1.0]), np.array([0.0]), np.array([-119.1]), 262.5, 38.5)
    assert alpha[0] == pytest.approx(np.sin(np.radians(38.5)) * (-119.1 + 97.5))
    assert np.hypot(ue, ve)[0] == pytest.approx(1.0)


def test_wind_dir_from_conventions():
    assert wind_dir_from(np.array(0.0), np.array(-1.0)) == pytest.approx(0.0)  # from the north
    assert wind_dir_from(np.array(-1.0), np.array(0.0)) == pytest.approx(90.0)  # from the east
    assert circ_diff(350, 10) == pytest.approx(-20)


def test_horn_slope_aspect_on_a_plane():
    _y, x = np.mgrid[0:20, 0:20]
    dem = (x * 30.0 * 0.1).astype("float32")  # rises 10% to the east
    slope, aspect = slope_aspect(dem, 30.0)
    assert slope[10, 10] == pytest.approx(np.degrees(np.arctan(0.1)), rel=1e-4)
    assert aspect[10, 10] == pytest.approx(270.0)  # faces (drains) west


def test_metrics_bias_and_direction():
    m = metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0]), "wspd")
    assert m["bias"] == pytest.approx(1.0) and m["r"] == pytest.approx(1.0)
    d = metrics(np.array([350.0, 355.0, 5.0]), np.array([10.0, 15.0, 25.0]), "wdir")
    assert d["bias"] == pytest.approx(20.0)


def test_cog_roundtrip_with_scale(tmp_path):
    grid = io.Grid("EPSG:32611", 30.0, (300000.0, 3800000.0, 306000.0, 3806000.0), 200, 200)
    a = np.full(grid.shape, 125, dtype="int16")
    a[0, 0] = -9999
    p = io.write_raster(
        tmp_path / "ch.tif",
        a,
        grid,
        nodata=-9999,
        dtype="int16",
        units="m",
        description="canopy height",
        scale=0.1,
    )
    phys, meta = io.read_raster(p)
    assert np.isnan(phys[0, 0]) and phys[1, 1] == pytest.approx(12.5)
    assert meta["units"] == "m" and grid.matches(p)
    import rasterio

    with rasterio.open(p) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"


def test_cube_conventions(tmp_path):
    t = pd.date_range("2017-12-05", periods=3, freq="h", tz="UTC")
    lat, lon = np.full((2, 2), 34.0), np.full((2, 2), 241.0)
    ds = io.new_cube(t, lat, lon)
    ds["t2m"] = (("time", "y", "x"), np.zeros((3, 2, 2), "float32"))
    p = io.write_cube(ds, tmp_path / "c.zarr", title="t", source="s")
    back = io.read_cube(p)
    assert back.attrs["Conventions"] == "CF-1.10"
    assert back["t2m"].attrs["units"] == "degC"
    assert float(back["longitude"].max()) == pytest.approx(-119.0)
    assert pd.Timestamp(back["time"].values[0]) == pd.Timestamp("2017-12-05 00:00")
    ds["mystery"] = ("time", np.zeros(3))
    with pytest.raises(KeyError):
        io.write_cube(ds, tmp_path / "d.zarr", title="t", source="s")


def test_tables_are_iso_utc(tmp_path):
    df = pd.DataFrame({"t": pd.to_datetime(["2017-12-05 09:26"]).tz_localize("UTC"), "v": [1]})
    p = io.write_table(df, tmp_path / "x.csv")
    assert "2017-12-05T09:26:00Z" in p.read_text()
    back = io.read_table(p, ["t"])
    assert back["t"].iloc[0] == df["t"].iloc[0]

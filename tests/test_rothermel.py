"""Rothermel port vs the Fire Lab behave library's own test values (src/testBehave/testBehave.cpp).

Base scenario (behave setSurfaceInputsForGS4LowMoistureScenario): GS4 (124); moisture 6/7/8/60/90 %;
5 mi/h 20-ft wind; 30 % slope; canopy cover 50 %, canopy height 30 ft, crown ratio 0.5.
"""

import math

import numpy as np
import pytest

from firespread import rothermel as R

CH = R.CHAINS_PER_HOUR_PER_FTMIN
PCT30 = math.degrees(math.atan(0.30))  # 30 % slope in degrees
TOL = 1e-5


def gs4(**kw):
    base = dict(
        fuel_model=124,
        m1=0.06,
        m10=0.07,
        m100=0.08,
        m_herb=0.60,
        m_woody=0.90,
        wind_speed=5.0,
        wind_direction=0.0,
        slope=PCT30,
        aspect=0.0,
        wind_height="20ft",
        orientation="north",
        canopy_cover=0.5,
        canopy_height=30.0,
        crown_ratio=0.5,
    )
    base.update(kw)
    return R.surface_fire(**base)


def close(a, b, tol=TOL):
    assert float(a) == pytest.approx(b, abs=tol), (float(a), b)


def test_fuel_table_from_behave():
    t = R.fuel_table()
    assert t.code[124] == "GS4" and t.code[147] == "SH7" and t.code[4] == "FM4"
    assert t.dynamic[124] and not t.dynamic[147]
    close(t.load[147, 1], 5.3 * 2000 / 43560, 1e-12)  # SH7 10-h load, 5.3 t/ac
    assert not t.burnable(91) and t.burnable(145)


def test_single_fuel_model_north_45_wind_95_aspect_30deg():
    f = gs4(slope=30.0, wind_direction=45.0, aspect=95.0)
    close(f["ros"] * CH, 19.677584)
    close(f["mx_live"] * 100, 137.968551)
    close(f["moisture_live"] * 100, 85.874007)
    close(f["moisture_dead"] * 100, 6.005463)
    close(f["sigma"], 1631.128734, 1e-4)
    close(f["heat_source"], 5177.248579, 1e-4)
    close(f["lwr"], 1.375624)


def test_heat_source_no_wind_direction_no_slope():
    close(gs4(slope=0.0, wind_direction=0.0, aspect=0.0)["heat_source"], 1164.267376, 1e-4)


def test_upslope_mode_rates_and_intensity():
    f = gs4(orientation="upslope")
    close(f["ros"] * CH, 8.876216)
    close(f["backing"] * CH, 2.91614659)
    close(f["flanking"] * CH, 5.08766627)
    close(f["ros"] * CH * 2, 17.7524327)  # 2-h spread distance, chains
    close(f["fireline_intensity"], 598.339039, 1e-4)
    close(f["fireline_intensity"] * R.KW_PER_M_PER_BTU_PER_FT_S, 2072.730450, 1e-3)


@pytest.mark.parametrize(
    "kw, expected",
    [
        (dict(orientation="upslope", wind_direction=90.0), 7.091665),
        (dict(wind_direction=0.0, aspect=0.0), 8.876216),
        (dict(wind_direction=45.0, aspect=215.0), 4.113265),
        (dict(wind_direction=45.0, aspect=5.0), 8.503960),
        (dict(fuel_model=4, wind_direction=90.0, aspect=0.0, slope=30.0, canopy_cover=0.4), 46.631688),
        (dict(fuel_model=91), 0.0),
    ],
)
def test_spread_rate_cases(kw, expected):
    close(gs4(**kw)["ros"] * CH, expected)


def test_length_to_width_ratio():
    close(gs4(wind_speed=0.0, slope=0.0)["lwr"], 1.0)
    close(gs4(wind_speed=5.0, wind_height="midflame", slope=0.0)["lwr"], 1.590064)
    close(gs4(wind_speed=15.0, wind_direction=45.0, aspect=95.0, slope=30.0)["lwr"], 1.519936)


def test_direction_of_interest():
    f = gs4(orientation="upslope", wind_direction=0.0)
    r = R.ros_in_direction(f, 90.0)
    close(r, 5.596433)
    close(R.flame_length(R.fireline_intensity(f, r)), 6.598148)
    f = gs4(orientation="upslope", wind_direction=290.0)
    close(R.ros_in_direction(f, 160.0) * CH, 2.766387)
    f = gs4(orientation="upslope", wind_direction=215.0)
    close(R.ros_in_direction(f, 215.0) * CH, 2.818063)
    f = gs4(wind_direction=280.0, aspect=135.0)
    close(R.ros_in_direction(f, 30.0, mode="ignition") * CH, 4.180938)


def test_vectorized_matches_scalar_and_handles_nonburnable():
    fm = np.array([[124, 91], [147, 102]])
    ws = np.array([[5.0, 5.0], [10.0, 20.0]])
    v = R.surface_fire(fm, 0.06, 0.07, 0.08, 0.6, 0.9, ws, 45.0, 10.0, 95.0)
    for i in range(2):
        for j in range(2):
            s = R.surface_fire(int(fm[i, j]), 0.06, 0.07, 0.08, 0.6, 0.9, float(ws[i, j]), 45.0, 10.0, 95.0)
            for k in ("ros", "direction", "fireline_intensity", "lwr"):
                close(v[k][i, j], float(s[k]), 1e-9)
    assert v["ros"][0, 1] == 0 and not v["burnable"][0, 1]
    assert np.isfinite(v["ros"]).all()


def test_physical_monotonicity():
    dry, wet = gs4(m1=0.03), gs4(m1=0.12)
    assert dry["ros"] > wet["ros"]
    assert gs4(wind_speed=15.0)["ros"] > gs4(wind_speed=5.0)["ros"]
    assert R.surface_fire(147, 0.06, 0.07, 0.08, 0.3, 0.6, 5.0, slope=0.0)["ros"] > 0
    assert R.surface_fire(147, 0.40, 0.40, 0.40, 0.3, 0.6, 5.0, slope=0.0)["ros"] == 0  # above extinction


def test_si_wrapper_matches_imperial():
    si = R.surface_fire_si(147, 0.05, 0.06, 0.08, 0.6, 0.7, 10.0, 30.0, 15.0, 200.0)
    imp = R.surface_fire(
        147, 0.05, 0.06, 0.08, 0.6, 0.7, 10.0 * R.MPH_PER_MPS, 30.0, 15.0, 200.0, wind_height="10m"
    )
    close(si["ros_m_min"], float(imp["ros"]) * 0.3048, 1e-9)


def test_all_landfire_fbfm40_codes_defined():
    codes = [
        91,
        92,
        93,
        98,
        99,
        *range(101, 110),
        *range(121, 125),
        *range(141, 150),
        *range(161, 166),
        *range(181, 190),
        *range(201, 205),
    ]
    t = R.fuel_table()
    assert all(t.defined[c] for c in codes)
    assert not any(t.burnable(c) for c in (91, 92, 93, 98, 99))


def test_grid_helpers():
    from firespread import surface_spread as S

    # UTM 11N west of the central meridian (-117): meridians lean east going north, by ~(dlon * sin lat)
    g = S.true_north_azimuth(np.array([-119.1]), np.array([34.4]), "EPSG:32611")[0]
    assert g == pytest.approx(2.1 * math.sin(math.radians(34.4)), abs=0.05)
    assert S.true_north_azimuth(np.array([-117.0]), np.array([34.4]), "EPSG:32611")[0] == pytest.approx(
        0, abs=1e-3
    )
    # bilinear reproduces a plane, ascending or descending axes
    xs, ys = np.array([0.0, 10, 20]), np.array([100.0, 50, 0])
    f = 2 * xs[None, :] + 3 * ys[:, None]
    b = S.Bilinear(xs, ys, np.array([5.0, 17]), np.array([20.0, 80]))
    assert np.allclose(b(f), [2 * 5 + 3 * 20, 2 * 17 + 3 * 80])
    # nearest-land fill
    land = np.array([[True, False], [False, False]])
    v = np.where(land, 7.0, np.nan)
    assert np.all(S.fill(v, S.nearest_land_index(land)) == 7.0)
    # gaps over land (e.g. live moisture without a gridMET cell) come from the nearest valid cell
    w = np.array([[1.0, np.nan, np.nan], [2.0, 3.0, np.nan]])
    f = S.fill(w, S.nearest_land_index(np.ones((2, 3), bool)))
    assert np.isfinite(f).all() and f[0, 1] in (1.0, 3.0) and f[1, 2] == 3.0
    st = S.error_stats([1, 2, 4], [2, 2, 2])
    assert st["n"] == 3 and st["within_2x"] == 100 and st["under"] == pytest.approx(100 / 3)

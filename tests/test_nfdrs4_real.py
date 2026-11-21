"""Checks our calls against the real Fire Lab NFDRS4 library (skipped if it is not installed)."""

import math

import numpy as np
import pytest

from firespread.fuel_moisture import make_stick, run_nelson

nfdrs4 = pytest.importorskip("nfdrs4")
pytestmark = pytest.mark.nfdrs4


def test_library_reference_value():
    # value published in the NFDRS4 bindings' own README/tests
    d = nfdrs4.DeadFuelMoisture.createDeadFuelMoisture1()
    assert d.update(1, 1, 1, 1, 1)
    assert math.isclose(d.medianRadialMoisture(), 0.5606049792805294, rel_tol=1e-9)


@pytest.mark.parametrize("cls", [1, 10, 100, 1000])
def test_every_stick_class_initializes(cls):
    s = make_stick(nfdrs4, cls)
    s.initializeEnvironment(20.0, 0.3, 0.0, 0.0, 20.0, 0.3, 0.1)
    assert s.update(1.0, 20.0, 0.3, 0.0, 0.0, 0.0218, True)
    assert 0.0 < s.medianRadialMoisture() < 1.0


def test_nelson_dries_under_santa_ana_and_wets_with_rain():
    n = 96
    h = np.arange(n) % 24
    T = 22 + 8 * np.sin((h - 15) / 24 * 2 * np.pi)
    RH = np.full(n, 0.08)
    SW = np.clip(800 * np.sin((h - 15) / 24 * 2 * np.pi), 0, None)
    P = np.zeros(n)
    fm1, bad = run_nelson(nfdrs4, 1, T, RH, SW, P, 0.15)
    fm10, _ = run_nelson(nfdrs4, 10, T, RH, SW, P, 0.15)
    assert bad == 0
    assert fm1[-1] < 6.0  # very dry fine fuels after 4 dry days
    assert fm1[30] < fm10[30]  # 1-h responds faster than 10-h
    P_wet = P.copy()
    P_wet[48:54] = 0.3  # 3 mm/h for 6 h (cm/h)
    RH_wet = RH.copy()
    RH_wet[48:60] = 0.95
    wet, _ = run_nelson(nfdrs4, 1, T, RH_wet, SW, P_wet, 0.15)
    assert wet[54] > fm1[54] + 10.0


def test_live_fuel_moisture_daily_api():
    """NFDRS4.Init + UpdateDaily, as live_fuel_moisture.run_cell calls them, give values within NFDRS limits."""
    import pandas as pd

    from firespread.live_fuel_moisture import run_cell

    days = pd.date_range("2016-12-01", "2017-12-10", freq="D")
    doy = days.day_of_year.to_numpy()
    season = np.cos(2 * np.pi * (doy - 15) / 365.25)  # wet winter, dry autumn
    d = {
        "tmmx": 296.0 - 6 * season,
        "tmmn": 283.0 - 5 * season,
        "rmin": 30.0 + 15 * season,
        "pr": np.where(season > 0.5, 3.0, 0.0),
        "vs": np.full(len(days), 3.0),
        "fm100": 10.0 + 3 * season,
        "fm1000": 12.0 + 3 * season,
    }
    herb, wood, gsi = run_cell(nfdrs4, 34.4, days, d, herbs_annual=True)
    assert np.isfinite(herb).all() and np.isfinite(wood).all() and np.isfinite(gsi).all()
    assert herb.min() >= 30 - 1e-6 and herb.max() <= 250 + 1e-6
    assert wood.min() >= 60 - 1e-6 and wood.max() <= 200 + 1e-6
    assert gsi.min() >= 0 and gsi.max() <= 1

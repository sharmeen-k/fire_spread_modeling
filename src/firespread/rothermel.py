"""Rothermel (1972) surface fire spread, vectorized; a line-by-line port of the Fire Lab `behave` library.

This reproduces the surface fire calculations in firelab/behave (the engine behind BehavePlus 6/7 and the
reference implementation for FlamMap/FARSITE surface spread), including its conventions:
- Scott & Burgan (2005) 40 + Anderson (1982) 13 fuel models; table taken verbatim from behave's
  fuelModels.cpp (data/fuel_models.csv); dynamic herbaceous load transfer for (D) models.
- Albini (1976) live moisture of extinction, size-class weighting of net fuel load, silica damping.
- Wind adjustment factor from canopy cover/height/crown ratio (Albini & Baughman 1979; Andrews 2012),
  20-ft or 10-m (/1.15) wind input; wind and slope combined by vector addition (Rothermel 1983);
  optional effective wind speed limit (off by default, as in behave).
- Elliptical fire shape (Anderson 1983 length-to-width ratio), backing/flanking rates, spread rate in any
  direction (from the perimeter or from the ignition point), Byram fireline intensity and flame length.

Units are behave's base units: fuel moisture as fractions, wind in mi/h, slope and directions in degrees,
canopy height and fuelbed depth in ft, spread rates in ft/min, reaction intensity in Btu/ft^2/min,
fireline intensity in Btu/ft/s. `surface_fire_si` wraps this for SI inputs/outputs.

Directions: in "north" orientation the wind direction is where the wind blows FROM (degrees clockwise
from north) and aspect is the downslope direction; spread directions are where the fire goes TO.
In "upslope" orientation both are relative to the upslope direction.

Reference values for the tests come from behave's own test suite (src/testBehave/testBehave.cpp).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

BEHAVE_COMMIT = "830fbf35dc4b5bd43ec06a18ff2145418bdde6ec"  # firelab/behave, source of table and test values

FT_PER_M = 1 / 0.3048
MPH_PER_MPS = 3600 / 1609.344
FTMIN_PER_MPH = 88.0
CHAINS_PER_HOUR_PER_FTMIN = 60 / 66
KW_PER_M_PER_BTU_PER_FT_S = 3.464140419  # behave behaveUnits.cpp

_N = 5  # size classes per life state (behave MaxParticles)


@dataclass(frozen=True)
class FuelTable:
    """Fuel model parameters as lookup arrays indexed by fuel model number (0-255)."""

    code: np.ndarray
    depth: np.ndarray
    mx_dead: np.ndarray
    heat_dead: np.ndarray
    heat_live: np.ndarray
    load: np.ndarray  # (256, 5): 1h, 10h, 100h, herb, woody (lb/ft^2)
    savr: np.ndarray  # (256, 3): 1h, herb, woody (ft^2/ft^3)
    dynamic: np.ndarray
    defined: np.ndarray

    def burnable(self, fm) -> np.ndarray:
        fm = np.asarray(fm, dtype=np.int64)
        ok = (fm >= 0) & (fm < 256)
        out = np.zeros(fm.shape, bool)
        out[ok] = self.load[fm[ok]].sum(-1) > 0
        return out


@lru_cache(maxsize=1)
def fuel_table() -> FuelTable:
    with resources.files("firespread.data").joinpath("fuel_models.csv").open() as f:
        df = pd.read_csv(f)
    n = 256
    z = lambda: np.zeros(n)  # noqa: E731
    code = np.full(n, "", dtype=object)
    depth, mxd, hd, hl = z(), z(), z(), z()
    load, savr = np.zeros((n, 5)), np.zeros((n, 3))
    dyn, defined = np.zeros(n, bool), np.zeros(n, bool)
    for r in df.itertuples():
        i = int(r.number)
        code[i], depth[i], mxd[i], hd[i], hl[i] = r.code, r.depth_ft, r.mx_dead, r.heat_dead, r.heat_live
        load[i] = (r.load_1h, r.load_10h, r.load_100h, r.load_herb, r.load_woody)
        savr[i] = (r.savr_1h, r.savr_herb, r.savr_woody)
        dyn[i], defined[i] = bool(r.dynamic), True
    return FuelTable(code, depth, mxd, hd, hl, load, savr, dyn, defined)


def _b(*arrays):
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])


def wind_adjustment_factor(canopy_cover, canopy_height_ft, crown_ratio, fuelbed_depth_ft):
    """behave WindAjustmentFactor::calculateWindAdjustmentFactorWithCrownRatio (cover, ratio as fractions)."""
    cc, ch, cr, d = _b(canopy_cover, canopy_height_ft, crown_ratio, fuelbed_depth_ft)
    cf = cr * cc / 3.0
    sheltered = ~((cc < 1e-7) | (cf < 0.05) | (ch < 6.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        unsh = np.where(d > 1e-7, 1.83 / np.log((20.0 + 0.36 * d) / (0.13 * d)), 0.0)
        sh = 0.555 / (np.sqrt(cf * ch) * np.log((20.0 + 0.36 * ch) / (0.13 * ch)))
    return np.where(sheltered, sh, unsh)


def surface_fire(
    fuel_model,
    m1,
    m10,
    m100,
    m_herb,
    m_woody,
    wind_speed,
    wind_direction=0.0,
    slope=0.0,
    aspect=0.0,
    *,
    wind_height: str = "20ft",
    orientation: str = "north",
    canopy_cover=0.0,
    canopy_height=0.0,
    crown_ratio=0.0,
    waf=None,
    wind_limit: bool = False,
) -> dict:
    """Surface fire behavior in the direction of maximum spread (behave doSurfaceRunInDirectionOfMaxSpread).

    wind_height: "20ft", "10m" (divided by 1.15 to 20 ft) or "midflame". waf, if given, replaces the
    computed wind adjustment factor. All inputs broadcast against each other.
    """
    t = fuel_table()
    fm = np.asarray(fuel_model, dtype=np.int64)
    m1, m10, m100, mh, mw, ws, wd, slope, aspect, cc, ch, cr = _b(
        m1,
        m10,
        m100,
        m_herb,
        m_woody,
        wind_speed,
        wind_direction,
        slope,
        aspect,
        canopy_cover,
        canopy_height,
        crown_ratio,
    )
    fm = np.broadcast_to(fm, m1.shape)
    fm = np.where((fm >= 0) & (fm < 256), fm, 0)
    shape = m1.shape
    depth, mxd = t.depth[fm], t.mx_dead[fm]
    hd, hl = t.heat_dead[fm], t.heat_live[fm]
    L, S = t.load[fm], t.savr[fm]
    zero = np.zeros(shape)

    load_d = np.stack([L[..., 0], L[..., 1], L[..., 2], zero, zero], -1)
    load_l = np.stack([L[..., 3], L[..., 4], zero, zero, zero], -1)
    n_d = np.where((load_d[..., :4] != 0).any(-1), 4, 0)  # counted before load transfer, as in behave
    n_l = np.where((load_l != 0).any(-1), 5, 0)
    moist_d = np.stack([m1, m10, m100, m1, zero], -1)
    moist_l = np.stack([mh, mw, zero, zero, zero], -1)
    savr_d = np.stack([S[..., 0], np.full(shape, 109.0), np.full(shape, 30.0), S[..., 1], zero], -1)
    savr_l = np.stack([S[..., 1], S[..., 2], zero, zero, zero], -1)

    # dynamic herbaceous load transfer (Scott & Burgan 2005)
    dyn = t.dynamic[fm]
    herb = load_l[..., 0].copy()
    full = dyn & (mh < 0.30)
    part = dyn & ~full & (mh <= 1.20)
    moved = np.where(full, herb, np.where(part, herb * (1.333 - 1.11 * mh), 0.0))
    load_d[..., 3] = np.where(full | part, moved, load_d[..., 3])
    load_l[..., 0] = np.where(full | part, herb - moved, load_l[..., 0])

    heat_d = np.repeat(hd[..., None], _N, -1)
    heat_l = np.stack([hl, hl, hl, zero, zero], -1)
    idx = np.arange(_N)

    def area_fractions(load, savr, n):
        inc = idx < n[..., None]
        sa = np.where(inc, load * savr / 32.0, 0.0)
        tot = sa.sum(-1)
        frac = np.where(inc & (tot[..., None] > 1e-7), sa / np.where(tot > 0, tot, 1.0)[..., None], 0.0)
        # size-sorted weights (behave sum/assignFractionOfTotalSurfaceAreaBySizeClass)
        bins = np.select([savr >= 1200, savr >= 192, savr >= 96, savr >= 48, savr >= 16], [0, 1, 2, 3, 4], 5)
        summed = np.stack([np.where(bins == k, frac, 0.0).sum(-1) for k in range(6)], -1)
        summed[..., 5] = 0.0
        sorted_ = np.take_along_axis(summed, bins, -1)
        return tot, frac, sorted_

    tsa_d, frac_d, sorted_d = area_fractions(load_d, savr_d, n_d)
    tsa_l, frac_l, sorted_l = area_fractions(load_l, savr_l, n_l)
    tsa = tsa_d + tsa_l
    burnable = tsa > 0
    f_dead = np.where(burnable, tsa_d / np.where(burnable, tsa, 1.0), 0.0)
    f_live = np.where(burnable, 1.0 - f_dead, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # live moisture of extinction (Albini 1976)
        vd, vl = savr_d > 1e-7, savr_l > 1e-7
        wfd = np.where(vd, load_d * np.exp(-138.0 / np.where(vd, savr_d, 1.0)), 0.0)
        fine_dead = wfd.sum(-1)
        fdm = np.where(
            fine_dead > 1e-7, (wfd * moist_d).sum(-1) / np.where(fine_dead > 0, fine_dead, 1.0), 0.0
        )
        incl = vl & (idx < n_l[..., None])
        fine_live = np.where(incl, load_l * np.exp(-500.0 / np.where(vl, savr_l, 1.0)), 0.0).sum(-1)
        ratio = np.where(fine_live > 1e-7, fine_dead / np.where(fine_live > 0, fine_live, 1.0), 0.0)
        mxl = 2.9 * ratio * (1.0 - fdm / np.where(mxd > 0, mxd, 1.0)) - 0.226
        mxl = np.where(n_l > 0, np.maximum(mxl, mxd), 0.0)

        # characteristic values by life state
        silica_d = np.full(load_d.shape, 0.01)
        silica_l = np.where(idx < 2, 0.01, 0.0) * np.ones(load_l.shape)
        st = 0.0555

        def weighted(load, savr, frac, sorted_, moist, heat, silica, valid):
            wn = np.where(valid, load * (1.0 - st), 0.0)
            w = lambda x: np.where(valid, frac * x, 0.0).sum(-1)  # noqa: E731
            return (
                w(heat),
                w(silica),
                w(moist),
                w(savr),
                np.where(valid, load, 0.0).sum(-1),
                (sorted_ * wn).sum(-1),
            )

        wh_d, wsi_d, wm_d, wsv_d, tl_d, wfl_d = weighted(
            load_d, savr_d, frac_d, sorted_d, moist_d, heat_d, silica_d, vd
        )
        wh_l, wsi_l, wm_l, wsv_l, tl_l, wfl_l = weighted(
            load_l, savr_l, frac_l, sorted_l, moist_l, heat_l, silica_l, vl
        )
        sigma = f_dead * wsv_d + f_live * wsv_l
        dsafe = np.where(depth > 0, depth, 1.0)
        bulk = (tl_d + tl_l) / dsafe
        packing = (load_d.sum(-1) + load_l.sum(-1)) / (dsafe * 32.0)
        ssafe = np.where(sigma > 0, sigma, 1.0)
        rpr = packing / (3.348 / ssafe**0.8189)

        qd = np.where(
            vd,
            f_dead[..., None]
            * frac_d
            * (250.0 + 1116.0 * moist_d)
            * np.exp(-138.0 / np.where(vd, savr_d, 1.0)),
            0.0,
        ).sum(-1)
        ql = np.where(
            vl,
            f_live[..., None]
            * frac_l
            * (250.0 + 1116.0 * moist_l)
            * np.exp(-138.0 / np.where(vl, savr_l, 1.0)),
            0.0,
        ).sum(-1)
        heat_sink = bulk * (qd + ql)
        xi = np.where(
            sigma < 1e-7,
            0.0,
            np.exp((0.792 + 0.681 * np.sqrt(ssafe)) * (packing + 0.1)) / (192.0 + 0.2595 * ssafe),
        )

        # reaction intensity
        aa = 133.0 / ssafe**0.7913
        s15 = ssafe**1.5
        gamma = s15 / (495.0 + 0.0594 * s15) * np.where(rpr > 0, rpr, 1.0) ** aa * np.exp(aa * (1.0 - rpr))

        def eta_m(wm, mx):
            rel = np.where(mx > 0, wm / np.where(mx > 0, mx, 1.0), 0.0)
            poly = 1.0 - 2.59 * rel + 5.11 * rel**2 - 3.52 * rel**3
            return np.where((wm >= mx) | (rel > 1.0), 0.0, poly)

        def eta_s(wsi):
            den = np.where(wsi > 0, wsi, 0.0) ** 0.19
            return np.where(den < 1e-6, 0.0, np.minimum(0.174 / np.where(den > 0, den, 1.0), 1.0))

        ir_d = gamma * wfl_d * wh_d * eta_m(wm_d, mxd) * eta_s(wsi_d)
        ir_l = gamma * wfl_l * wh_l * eta_m(wm_l, mxl) * eta_s(wsi_l)
        ir = np.where(burnable, ir_d + ir_l, 0.0)
        r0 = np.where(heat_sink < 1e-7, 0.0, ir * xi / np.where(heat_sink > 0, heat_sink, 1.0))

        # midflame wind
        u = ws * FTMIN_PER_MPH
        if wind_height == "midflame":
            waf_ = np.ones(shape)
        elif wind_height in ("20ft", "10m"):
            if wind_height == "10m":
                u = u / 1.15
            waf_ = (
                np.broadcast_to(np.asarray(waf, float), shape)
                if waf is not None
                else wind_adjustment_factor(cc, ch, cr, depth)
            )
            u = waf_ * u
        else:
            raise ValueError(f"wind_height must be '20ft', '10m' or 'midflame', not {wind_height!r}")
        wc = 7.47 * np.exp(-0.133 * ssafe**0.55)
        wb = 0.02526 * ssafe**0.54
        we = 0.715 * np.exp(-0.000359 * ssafe)
        rprs = np.where(rpr > 0, rpr, 1.0)
        phi_w = np.where(u < 1e-7, 0.0, np.where(u > 0, u, 1.0) ** wb * wc * rprs ** (-we))
        tan_s = np.tan(np.deg2rad(slope))
        phi_s = 5.275 * np.where(packing > 0, packing, 1.0) ** -0.3 * tan_s**2
        limit = 0.9 * ir
        phi_s = np.where((phi_s > 0) & (phi_s > limit), limit, phi_s)  # as in behave calculateWindSpeedLimit

        # vector sum of wind and slope (Rothermel 1983)
        if orientation == "north":
            wrel = wd - aspect
        elif orientation == "upslope":
            wrel = wd
        else:
            raise ValueError("orientation must be 'north' or 'upslope'")
        rad = np.deg2rad(wrel)
        x = r0 * phi_s + r0 * phi_w * np.cos(rad)
        y = r0 * phi_w * np.sin(rad)
        forward = r0 + np.hypot(x, y)
        az = np.rad2deg(np.arctan2(y, x))
        az = np.where(az < -1e-20, az + 360.0, az)
        az = np.where(np.abs(az) < 0.5, 0.0, az)
        if orientation == "north":
            az = np.mod(az + aspect + 180.0, 360.0)

        r0s = np.where(r0 > 0, r0, 1.0)
        phi_ew = np.where(r0 > 0, forward / r0s - 1.0, 0.0)
        ew = np.where(phi_ew > 0, (phi_ew * rprs**we / wc) ** (1.0 / wb), 0.0)
        limited = np.zeros(shape, bool)
        if wind_limit:
            limited = ew > limit
            phi_l = wc * np.where(limit > 0, limit, 1.0) ** wb * rprs ** (-we)
            forward = np.where(limited, r0 * (1.0 + phi_l), forward)
            ew = np.where(limited, limit, ew)
        ew_mph = ew / FTMIN_PER_MPH

        lwr = np.where(
            ew_mph > 1e-7,
            np.minimum(0.936 * np.exp(0.1147 * ew_mph) + 0.461 * np.exp(-0.0692 * ew_mph) - 0.397, 8.0),
            1.0,
        )
        ecc = np.where(lwr**2 - 1 > 0, np.sqrt(np.maximum(lwr**2 - 1, 0)) / lwr, 0.0)
        backing = forward * (1.0 - ecc) / (1.0 + ecc)
        flanking = (backing + forward) / lwr * 0.5

        tr = np.where(sigma < 1e-7, 0.0, 384.0 / ssafe)
        fli = forward * ir * tr / 60.0

    burn = burnable & (r0 > 0)
    out = dict(
        ros=np.where(burn, forward, 0.0),
        backing=np.where(burn, backing, 0.0),
        flanking=np.where(burn, flanking, 0.0),
        direction=np.where(burn, az, 0.0),
        lwr=np.where(burn, lwr, 1.0),
        eccentricity=np.where(burn, ecc, 0.0),
        effective_wind=np.where(burn, ew_mph, 0.0),
        midflame_wind=u / FTMIN_PER_MPH,
        waf=waf_,
        reaction_intensity=ir,
        residence_time=tr,
        heat_per_unit_area=ir * tr,
        fireline_intensity=np.where(burn, fli, 0.0),
        flame_length=flame_length(np.where(burn, fli, 0.0)),
        r0=r0,
        phi_w=phi_w,
        phi_s=phi_s,
        heat_source=ir * xi * (1.0 + phi_s + phi_w),
        sigma=sigma,
        packing_ratio=packing,
        relative_packing_ratio=rpr,
        mx_live=mxl,
        mx_dead=mxd,
        moisture_dead=wm_d,
        moisture_live=wm_l,
        wind_limited=limited,
        burnable=burnable,
    )
    return out


def flame_length(fireline_intensity):
    """Byram (1959) flame length (ft) from fireline intensity (Btu/ft/s)."""
    i = np.asarray(fireline_intensity, float)
    return np.where(i < 1e-7, 0.0, 0.45 * np.where(i > 0, i, 1.0) ** 0.46)


def ros_in_direction(fire: dict, direction, mode: str = "perimeter"):
    """Spread rate toward `direction` (same orientation as the run), behave calculateSpreadRateAtVector.

    mode "perimeter": normal to the elliptical perimeter (FromPerimeter, BehavePlus default);
    mode "ignition": along a ray from the ignition point (FromIgnitionPoint).
    """
    fwd, back, flank, ecc, dmax = (
        np.asarray(fire[k]) for k in ("ros", "backing", "flanking", "eccentricity", "direction")
    )
    d = np.mod(np.asarray(direction, float), 360.0)
    beta = np.abs(dmax - d)
    beta = np.where(beta > 180.0, 360.0 - beta, beta)
    rad = np.deg2rad(beta)
    if mode == "perimeter":
        f = (fwd + back) / 2.0
        g = fwd - f
        r = g * np.cos(rad) + np.sqrt(f**2 * np.cos(rad) ** 2 + flank**2 * np.sin(rad) ** 2)
    elif mode == "ignition":
        r = fwd * (1.0 - ecc) / (1.0 - ecc * np.cos(rad))
    else:
        raise ValueError("mode must be 'perimeter' or 'ignition'")
    return np.where(fwd != 0, r, fwd)


def fireline_intensity(fire: dict, ros):
    """Byram fireline intensity (Btu/ft/s) for any spread rate (ft/min) of a run."""
    return np.asarray(ros) * fire["reaction_intensity"] * fire["residence_time"] / 60.0


def surface_fire_si(
    fuel_model,
    m1,
    m10,
    m100,
    m_herb,
    m_woody,
    wind_10m_ms,
    wind_from_deg,
    slope_deg,
    aspect_deg,
    canopy_cover=0.0,
    canopy_height_m=0.0,
    crown_ratio=0.0,
    **kw,
) -> dict:
    """SI wrapper: 10-m wind (m/s, from), moisture fractions, canopy height (m); returns m/min, kW/m, m."""
    out = surface_fire(
        fuel_model,
        m1,
        m10,
        m100,
        m_herb,
        m_woody,
        np.asarray(wind_10m_ms) * MPH_PER_MPS,
        wind_from_deg,
        slope_deg,
        aspect_deg,
        wind_height="10m",
        orientation="north",
        canopy_cover=canopy_cover,
        canopy_height=np.asarray(canopy_height_m) * FT_PER_M,
        crown_ratio=crown_ratio,
        **kw,
    )
    for k in ("ros", "backing", "flanking"):
        out[k + "_m_min"] = out[k] * 0.3048
    out["fireline_intensity_kw_m"] = out["fireline_intensity"] * KW_PER_M_PER_BTU_PER_FT_S
    out["flame_length_m"] = out["flame_length"] * 0.3048
    out["midflame_wind_ms"] = out["midflame_wind"] / MPH_PER_MPS
    return out

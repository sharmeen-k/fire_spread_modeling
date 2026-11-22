"""Check modeled live fuel moisture against field samples (Globe-LFMC 2.0).

Globe-LFMC 2.0 (Yebra et al. 2024, Scientific Data 11:332) compiles >280,000
field LFMC measurements, including the US National Fuel Moisture Database.
For one fire, samples of the configured species (default chamise, Adenostoma
fasciculatum) within `radius_km` of the first perimeter centroid and inside the
live-fuel-moisture period are averaged per site and date, and compared with the
NFDRS4 woody moisture at the nearest HRRR land cell on the same day.

The US subset of the workbook is cached as CSV (data/processed/lfmc/), since
reading the full workbook takes minutes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io, logs
from .config import Config
from .live_fuel_moisture import cube_path as live_path
from .stations import haversine_km

log = logging.getLogger(__name__)

SHEET = "LFMC data"
COLS = {
    "Site name": "site",
    "Country": "country",
    "State/Region": "state",
    "Latitude (WGS84, EPSG:4326)": "lat",
    "Longitude (WGS84, EPSG:4326)": "lon",
    "Sampling date (YYYYMMDD)": "date",
    "LFMC value (%)": "lfmc",
    "Species collected": "species",
    "Species functional type": "functional_type",
    "Individual sample or mean value": "sample_kind",
    "Elevation (m.a.s.l)": "elev_m",
}


def us_subset(cfg: Config) -> pd.DataFrame:
    """US rows of Globe-LFMC 2.0 with short column names (cached)."""
    src = cfg.path(cfg.lfmc_validation.file)
    cache = cfg.processed_dir / "lfmc" / "globe_lfmc_us.csv"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        df = pd.read_csv(cache, parse_dates=["date"])
    else:
        log.info("Reading %s (sheet '%s'); this takes a few minutes once", src.name, SHEET)
        raw = pd.read_excel(src, sheet_name=SHEET, usecols=list(COLS))
        df = raw.rename(columns=COLS)
        df = df[df["country"].astype(str).str.contains("United States|USA", case=False, na=False)].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        for c in ("site", "species", "functional_type"):
            df[c] = df[c].astype(str).str.strip()
        df = df.dropna(subset=["lat", "lon", "date", "lfmc"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        log.info("Cached %d US rows -> %s", len(df), cache.relative_to(cfg.root))
    return df


def nearby_samples(
    df: pd.DataFrame, lat0: float, lon0: float, radius_km: float, species, t0, t1
) -> pd.DataFrame:
    sel = df[df["species"].isin(species) & df["date"].between(t0, t1)].copy()
    sel["dist_km"] = haversine_km(lat0, lon0, sel["lat"].to_numpy(), sel["lon"].to_numpy())
    sel = sel[sel["dist_km"] <= radius_km]
    if sel.empty:
        return sel
    g = sel.groupby(["site", "date"], as_index=False).agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        dist_km=("dist_km", "first"),
        species=("species", "first"),
        obs_lfmc=("lfmc", "mean"),
        n_samples=("lfmc", "size"),
    )
    return g.sort_values(["site", "date"]).reset_index(drop=True)


def stats(o, m) -> dict:
    o, m = np.asarray(o, float), np.asarray(m, float)
    ok = np.isfinite(o) & np.isfinite(m)
    o, m = o[ok], m[ok]
    if len(o) < 2:
        return dict(
            n=len(o), obs_mean=np.nan, model_mean=np.nan, bias=np.nan, mae=np.nan, rmse=np.nan, r=np.nan
        )
    d = m - o
    r = np.corrcoef(o, m)[0, 1] if o.std() > 0 and m.std() > 0 else np.nan
    return dict(
        n=len(o),
        obs_mean=o.mean(),
        model_mean=m.mean(),
        bias=d.mean(),
        mae=np.abs(d).mean(),
        rmse=np.sqrt((d**2).mean()),
        r=r,
    )


def run(cfg: Config, fire_id: int) -> pd.DataFrame:
    log_ = logs.setup(cfg.outputs_dir / "logs", "lfmc_check", fire_id)
    vc = cfg.lfmc_validation
    lf = io.read_cube(live_path(cfg, fire_id)).load()
    lat, lon = lf["latitude"].values, lf["longitude"].values
    land = lf["land"].values.astype(bool)
    days = pd.DatetimeIndex(lf["time"].values).normalize()  # local (PST) days at the NFDRS obs hour
    rec = io.read_table(cfg.fire_dir(fire_id) / "perimeters" / "obs_times.csv")
    lat0, lon0 = float(rec["clat"].iloc[0]), float(rec["clon"].iloc[0])
    first_day = pd.Timestamp(lf.attrs["fire_start_day"])

    df = us_subset(cfg)
    s = nearby_samples(df, lat0, lon0, vc.radius_km, vc.species, days[0], days[-1])
    log_.info(
        "%d site-dates of %s within %g km, %s to %s",
        len(s),
        ", ".join(vc.species),
        vc.radius_km,
        days[0].date(),
        days[-1].date(),
    )
    if s.empty:
        raise RuntimeError("No field samples found; widen lfmc_validation.radius_km or check species names")

    # nearest HRRR land cell per site; model value on the sampling day
    pos = {d: i for i, d in enumerate(days)}
    rows = []
    for r in s.itertuples():
        d2 = haversine_km(r.lat, r.lon, lat, lon)
        d2 = np.where(land, d2, np.inf)
        iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
        k = pos.get(r.date)
        rows.append(
            {
                **r._asdict(),
                "cell_dist_km": float(d2[iy, ix]),
                "model_woody": float(lf["lfm_woody"].values[k, iy, ix]) if k is not None else np.nan,
                "model_herb": float(lf["lfm_herb"].values[k, iy, ix]) if k is not None else np.nan,
                "model_gsi": float(lf["gsi"].values[k, iy, ix]) if k is not None else np.nan,
            }
        )
    pairs = pd.DataFrame(rows).drop(columns=["Index"])
    pairs["period"] = np.where(
        pairs["date"] >= first_day - pd.Timedelta(days=vc.near_fire_days), "near_fire", "before"
    )
    io.write_table(pairs, cfg.outputs_dir / f"fire_{fire_id}_lfmc_check.csv")

    rep = logs.Report(log_)
    fmt = lambda x: f"{x:.1f}"  # noqa: E731
    rep(f"Live fuel moisture vs field samples, fire {fire_id}")
    rep("=" * 50)
    rep(
        f"Field data: Globe-LFMC 2.0 (Yebra et al. 2024); species {', '.join(vc.species)}; "
        f"sites within {vc.radius_km:g} km of the first perimeter centroid ({lat0:.3f}, {lon0:.3f})"
    )
    rep(
        f"Model: NFDRS4 GSI woody moisture at the nearest HRRR land cell, same day; "
        f"period {days[0].date()} to {days[-1].date()}"
    )
    rep(f"Site-dates: {len(pairs)} at {pairs['site'].nunique()} sites\n")
    rep("Overall (model - observed, %)\n" + "-" * 50)
    tab = [dict(subset="all", **stats(pairs.obs_lfmc, pairs.model_woody))]
    tab.append(
        dict(
            subset=f"last {vc.near_fire_days} days before + fire",
            **stats(
                pairs[pairs.period == "near_fire"].obs_lfmc, pairs[pairs.period == "near_fire"].model_woody
            ),
        )
    )
    for season, months in (
        ("Dec-Feb", (12, 1, 2)),
        ("Mar-May", (3, 4, 5)),
        ("Jun-Aug", (6, 7, 8)),
        ("Sep-Nov", (9, 10, 11)),
    ):
        m = pairs["date"].dt.month.isin(months)
        tab.append(dict(subset=season, **stats(pairs[m].obs_lfmc, pairs[m].model_woody)))
    rep.block(pd.DataFrame(tab).to_string(index=False, float_format=fmt))
    rep("\nBy site\n" + "-" * 50)
    per = []
    for site, d in pairs.groupby("site"):
        per.append(dict(site=site, dist_km=d.dist_km.iloc[0], **stats(d.obs_lfmc, d.model_woody)))
    rep.block(pd.DataFrame(per).sort_values("dist_km").to_string(index=False, float_format=fmt))
    near = pairs[pairs.period == "near_fire"].sort_values("date")
    if len(near):
        rep(f"\nSamples in the last {vc.near_fire_days} days before the fire and during it\n" + "-" * 50)
        rep.block(
            near[["date", "site", "dist_km", "n_samples", "obs_lfmc", "model_woody", "model_gsi"]].to_string(
                index=False, float_format=fmt
            )
        )
    rep("\nNotes")
    rep("- NFDRS4 woody moisture is bounded to 60-200%; chamise is routinely sampled below 60% in drought,")
    rep("  so the model cannot reach the driest observed values.")
    rep("- Field samples are site means of chamise new growth/foliage per Globe-LFMC protocols.")
    rep.save(cfg.outputs_dir / f"fire_{fire_id}_lfmc_check_summary.txt")

    # figure: up to 6 closest sites, model line vs samples
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = pairs.groupby("site")["dist_km"].first().sort_values().index[:6]
    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 2.3 * len(sites)), sharex=True, squeeze=False)
    for ax, site in zip(axes[:, 0], sites):
        d = pairs[pairs.site == site]
        r0 = d.iloc[0]
        d2 = np.where(land, haversine_km(r0.lat, r0.lon, lat, lon), np.inf)
        iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
        ax.plot(days, lf["lfm_woody"].values[:, iy, ix], color="tab:brown", lw=1, label="NFDRS4 woody")
        ax.plot(d.date, d.obs_lfmc, "o", color="tab:green", ms=4, label=f"{site} ({r0.dist_km:.0f} km)")
        ax.axvline(first_day, color="red", lw=1)
        ax.set_ylabel("LFMC (%)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0, 0].set_title(
        f"Fire {fire_id}: chamise field samples (Globe-LFMC 2.0) vs NFDRS4 woody moisture; red = fire start"
    )
    fig.tight_layout()
    fig.savefig(cfg.outputs_dir / f"fire_{fire_id}_lfmc_check.png", dpi=150)
    plt.close(fig)
    return pairs

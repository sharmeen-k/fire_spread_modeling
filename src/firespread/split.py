"""Train/validation/test split by fire (formerly script 05)."""

from __future__ import annotations

import pandas as pd
import pyogrio

from . import io, logs
from .config import Config
from .perimeters import label_to_utc


def exclusion_reason(r, cfg: Config, test_windows) -> str:
    s = cfg.study.split
    if r.fireID in cfg.study.test_fires:
        return ""
    if r.n_records < s.min_records:
        return "single_record"
    if r.start_utc < pd.Timestamp(cfg.study.hrrr_start, tz="UTC"):
        return "pre_HRRR"
    if r.days > s.max_days:
        return "long_duration"
    if s.exclude_concurrent and any(r.start_utc <= e and r.end_utc >= b for b, e in test_windows):
        return "concurrent_with_test"
    return ""


def make_split(cfg: Config) -> pd.DataFrame:
    log = logs.setup(cfg.outputs_dir / "logs", "split")
    s, test = cfg.study.split, cfg.study.test_fires
    fires = io.read_table(cfg.outputs_dir / "socal_fires.csv", ["start_utc", "end_utc"])
    fires["days"] = (fires["end_utc"] - fires["start_utc"]).dt.days
    windows = [(r.start_utc, r.end_utc) for r in fires[fires["fireID"].isin(test)].itertuples()]

    fires["reason"] = [exclusion_reason(r, cfg, windows) for r in fires.itertuples()]
    ok = fires["reason"] == ""
    fires["split"] = "excluded"
    fires.loc[ok, "split"] = "train"
    fires.loc[fires["fireID"].isin(test), "split"] = "test"

    pool = fires[
        ok
        & ~fires["fireID"].isin(test)
        & (fires["n_records"] >= s.min_records_val)
        & (fires["max_farea"] >= s.min_area_val_km2)
    ]
    log.info("Validation pool: %d fires", len(pool))
    val_ids = pool.sample(s.n_val, random_state=s.seed)["fireID"]
    fires.loc[fires["fireID"].isin(val_ids), "split"] = "val"
    fires["note"] = fires["fireID"].map(test).fillna("")

    cols = ["fireID", "split", "reason", "note", "start_utc", "end_utc", "n_records", "max_farea"]
    out = fires[cols].sort_values(["split", "start_utc"])
    io.write_table(out, cfg.outputs_dir / "split.csv")

    fires["transitions"] = fires["n_records"] - 1
    log.info(
        "\n%s",
        fires.groupby("split")
        .agg(fires=("fireID", "size"), records=("n_records", "sum"), samples=("transitions", "sum"))
        .to_string(),
    )
    log.info(
        "Excluded by reason:\n%s", fires[fires["split"] == "excluded"]["reason"].value_counts().to_string()
    )

    peri = pyogrio.read_dataframe(
        cfg.socal_perimeters,
        layer="perimeter",
        read_geometry=False,
        columns=["fireID", "time", "farea", "clon"],
    )
    peri["obs"] = label_to_utc(pd.to_datetime(peri["time"]), peri["clon"])
    for fid, label in test.items():
        d = peri[peri["fireID"] == fid].sort_values("obs")
        if d.empty:
            continue
        gaps = d["obs"].diff().dt.total_seconds().div(3600).dropna()
        log.info(
            "%s: %d records, farea monotonic: %s, step hours (min/max): %.1f/%.1f",
            label,
            len(d),
            d["farea"].is_monotonic_increasing,
            gaps.min(),
            gaps.max(),
        )
    log.info(
        "Validation fires:\n%s",
        fires[fires["split"] == "val"][["fireID", "start_utc", "n_records", "max_farea"]].to_string(
            index=False
        ),
    )
    return out

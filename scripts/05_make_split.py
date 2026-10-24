"""
Fix the train/val/test split BY FIRE. Writes outputs/split.csv.
Usage: python scripts/05_make_split.py
"""
import pandas as pd
import pyogrio

TEST = {4299: "Thomas 2017", 3922: "Woolsey 2018"}
N_VAL, SEED = 3, 42
HRRR_START = pd.Timestamp("2014-07-01")   # keep in sync with 03; verify the real date
MAX_DAYS = 60                              # drop multi-year merged objects
MIN_RECORDS = 2                            # need >= 2 records for one 12-hour sample
MIN_RECORDS_VAL, MIN_AREA_VAL = 6, 20      # validation fires: enough steps, km2
EXCLUDE_CONCURRENT = True                  # drop fires burning during a test fire

fires = pd.read_csv("outputs/socal_fires.csv", parse_dates=["start", "end"])
fires["days"] = (fires["end"] - fires["start"]).dt.days
windows = [(r.start, r.end) for r in fires[fires["fireID"].isin(TEST)].itertuples()]

def reason(r):
    if r.fireID in TEST:
        return ""
    if r.n_records < MIN_RECORDS:
        return "single_record"
    if r.start < HRRR_START:
        return "pre_HRRR"
    if r.days > MAX_DAYS:
        return "long_duration"
    if EXCLUDE_CONCURRENT and any(r.start <= e and r.end >= s for s, e in windows):
        return "concurrent_with_test"
    return ""

fires["reason"] = [reason(r) for r in fires.itertuples()]
ok = fires["reason"] == ""
fires["split"] = "excluded"
fires.loc[ok, "split"] = "train"
fires.loc[fires["fireID"].isin(TEST), "split"] = "test"

pool = fires[ok & ~fires["fireID"].isin(TEST)
             & (fires["n_records"] >= MIN_RECORDS_VAL) & (fires["max_farea"] >= MIN_AREA_VAL)]
print(f"Validation pool: {len(pool)} fires")
val_ids = pool.sample(N_VAL, random_state=SEED)["fireID"]
fires.loc[fires["fireID"].isin(val_ids), "split"] = "val"

fires["note"] = fires["fireID"].map(TEST).fillna("")
cols = ["fireID", "split", "reason", "note", "start", "end", "n_records", "max_farea"]
fires[cols].sort_values(["split", "start"]).to_csv("outputs/split.csv", index=False)

fires["transitions"] = fires["n_records"] - 1   # 12-hour samples per fire
print(fires.groupby("split").agg(fires=("fireID", "size"), records=("n_records", "sum"),
                                 samples=("transitions", "sum")))
print("\nExcluded by reason:")
print(fires[fires["split"] == "excluded"]["reason"].value_counts().to_string())

# sanity checks on the test fires: is farea cumulative, and are steps 12 h apart?
peri = pyogrio.read_dataframe("data/processed/socal_perimeters.gpkg", layer="perimeter",
                              read_geometry=False, columns=["fireID", "time", "farea"])
peri["time"] = pd.to_datetime(peri["time"])
for fid, label in TEST.items():
    d = peri[peri["fireID"] == fid].sort_values("time")
    gaps = d["time"].diff().dt.total_seconds().div(3600).dropna()
    print(f"\n{label}: {len(d)} records, farea monotonic: {d['farea'].is_monotonic_increasing}, "
          f"step hours (min/max): {gaps.min():.0f}/{gaps.max():.0f}")
print("\nValidation fires:", fires[fires['split'] == 'val'][['fireID', 'start', 'n_records', 'max_farea']].to_string(index=False))
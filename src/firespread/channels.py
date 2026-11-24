"""Sample channel names (dependency-free, so training environments can import them alone)."""

FUEL_GROUPS = {
    "NB": (91, 99),
    "GR": (101, 109),
    "GS": (121, 124),
    "SH": (141, 149),
    "TU": (161, 165),
    "TL": (181, 189),
    "SB": (201, 204),
}
STATIC = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    *[f"fuel_{g}" for g in FUEL_GROUPS],
    "cc",
    "ch",
    "cbh",
    "cbd",
    "nodata",
]
WEATHER = ["u10_mean", "v10_mean", "wspd_max", "gust_max", "t2m_mean", "rh_min", "rain_sum"]
MOISTURE = ["fm1_mean", "fm10_mean", "fm100_mean", "lfm_herb", "lfm_woody"]
PHYSICS = ["ros_mean", "ros_x", "ros_y", "ros_blockmax", "fli_mean"]
STEP = ["hours_since_start", "step_hours", "local_hour"]
CHANNELS = ["burned_t", *STATIC, *WEATHER, *MOISTURE, *PHYSICS, *STEP]

"""Typed pipeline configuration loaded from config/pipeline.yaml."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_ENV = "FIRESPREAD_CONFIG"
DEFAULT_CONFIG = Path("config") / "pipeline.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCfg(_Model):
    crs: str = "EPSG:32611"
    resolution_m: float = 30.0
    snap_m: float = 750.0
    buffer_m: float = 5000.0


class PathsCfg(_Model):
    data: str = "data"
    outputs: str = "outputs"


class InputsCfg(_Model):
    largefire_archive: str
    counties_zip: str
    frap_zip: str
    landfire_root: str


class LandfireRule(_Model):
    max_year: int
    version: str


class SplitCfg(_Model):
    n_val: int = 3
    seed: int = 42
    max_days: int = 60
    min_records: int = 2
    min_records_val: int = 6
    min_area_val_km2: float = 20.0
    exclude_concurrent: bool = True


class StudyCfg(_Model):
    counties: list[str]
    hrrr_start: date
    test_fires: dict[int, str]
    frap_targets: dict[str, int]
    match_window_days: int = 7
    split: SplitCfg = Field(default_factory=SplitCfg)


class HrrrCfg(_Model):
    lead_hours: int = 24
    margin_deg: float = 0.1
    workers: int = Field(default=6, ge=1, le=32)
    priority: list[str] = Field(default_factory=lambda: ["aws", "google"])


class OffshoreCfg(_Model):
    dir_min: float = 0.0
    dir_max: float = 110.0
    wspd_min: float = 5.0
    rh_max: float = 25.0
    cell_fraction: float = 0.20


class StationsCfg(_Model):
    match_tolerance_min: int = 30
    min_wspd_for_dir: float = 2.0
    max_cell_dist_km: float = 3.0
    box_shrink_deg: float = 0.03
    terrain_window_m: float = 3000.0
    raws_network: str = "2"
    offshore: OffshoreCfg = Field(default_factory=OffshoreCfg)


class FuelMoistureCfg(_Model):
    spinup_days: float = 30.0
    fm100_source: Literal["nelson", "gridmet"] = "nelson"
    classes: list[int] = Field(default_factory=lambda: [1, 10, 100, 1000])
    afternoon_utc_hour: int = 21
    gridmet_urls: list[str]


class LiveFuelMoistureCfg(_Model):
    spinup_days: int = Field(default=365, ge=60)
    herbs_annual: bool = True  # SoCal herbaceous fuels are mostly annual grasses
    obs_hour_utc: int = 21  # NFDRS observation time, ~13:00 PST


class LfmcValidationCfg(_Model):
    file: str = "data/raw/lfmc/Globe-LFMC-2.0 final.xlsx"
    species: list[str] = Field(default_factory=lambda: ["Adenostoma fasciculatum"])  # chamise
    radius_km: float = 60.0
    near_fire_days: int = 60


class SurfaceSpreadCfg(_Model):
    roi_buffer_m: float = Field(default=3000.0, ge=0)  # window = final extent + buffer
    hour_stride: int = Field(default=1, ge=1)  # run Rothermel every n-th hour of a step
    min_new_cells: int = Field(default=150, ge=1)  # ~1 VIIRS 375 m pixel at 30 m
    wind_limit: bool = False  # behave default


class SamplesCfg(_Model):
    block: int = Field(default=12, ge=1)  # 30 m cells per ML block side (12 -> 360 m)
    buffer_m: float = Field(default=6000.0, ge=0)  # window = final extent + buffer
    rothermel_hours: int = Field(default=3, ge=1)  # run Rothermel every n-th hour of a step
    rothermel_stride: int = Field(default=3, ge=1)  # ... on every n-th 30 m cell (3 -> 90 m)


class PublishCfg(_Model):
    auth: str = "pro"
    portal_url: str | None = None
    client_id: str | None = None
    title_template: str = "Fire_{fire_id}_observed_VIIRS"


class Config(_Model):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    inputs: InputsCfg
    landfire_rules: list[LandfireRule]
    study: StudyCfg
    fires: list[int]
    hrrr: HrrrCfg = Field(default_factory=HrrrCfg)
    stations: StationsCfg = Field(default_factory=StationsCfg)
    fuel_moisture: FuelMoistureCfg
    live_fuel_moisture: LiveFuelMoistureCfg = Field(default_factory=LiveFuelMoistureCfg)
    lfmc_validation: LfmcValidationCfg = Field(default_factory=LfmcValidationCfg)
    surface_spread: SurfaceSpreadCfg = Field(default_factory=SurfaceSpreadCfg)
    samples: SamplesCfg = Field(default_factory=SamplesCfg)
    publish: PublishCfg = Field(default_factory=PublishCfg)

    # set by load(); not part of the YAML
    root: Path = Field(default=Path("."), exclude=True)

    # ---------------------------------------------------------------- paths
    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self.path(self.paths.data)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.path(self.paths.outputs)

    @property
    def socal_perimeters(self) -> Path:
        return self.processed_dir / "socal_perimeters.gpkg"

    def fire_dir(self, fire_id: int) -> Path:
        return self.processed_dir / "fires" / str(fire_id)

    def landfire_dir(self, year: int) -> Path:
        for rule in self.landfire_rules:
            if year <= rule.max_year:
                return self.path(self.inputs.landfire_root) / rule.version
        raise ValueError(f"No LANDFIRE rule covers fire year {year}")


def find_config(start: Path | None = None) -> Path:
    """$FIRESPREAD_CONFIG, else config/pipeline.yaml in the current folder or a parent."""
    env = os.environ.get(CONFIG_ENV)
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        if (d / DEFAULT_CONFIG).exists():
            return d / DEFAULT_CONFIG
    raise FileNotFoundError(f"{DEFAULT_CONFIG} not found from {here}; run from the repo or set {CONFIG_ENV}")


def load(path: str | Path | None = None) -> Config:
    p = Path(path).resolve() if path else find_config()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = Config.model_validate(data)
    cfg.root = p.parent.parent  # config/pipeline.yaml -> repo root
    return cfg

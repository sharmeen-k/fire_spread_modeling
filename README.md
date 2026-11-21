# Hybrid wildfire spread modeling (Southern California)

MS Computational Data Science capstone, University of California, Riverside.

Does a hybrid model (raster Rothermel core + ember spotting + ML residual correction) predict
12-hour fire perimeters better than physics-only or ML-only models on held-out Southern
California fires (Thomas 2017, Woolsey 2018)?

This repository is the data pipeline and models, packaged as `firespread`.

## Setup (Windows, PowerShell)

1. Install [uv](https://docs.astral.sh/uv/) once:
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Install **Build Tools for Visual Studio 2022** with *Desktop development with C++*
   (needed once to build the Fire Lab's `nfdrs4` package from source).
3. From the repo root:

```powershell
uv sync            # creates .venv from uv.lock (exact, reproducible versions)
uv run firespread --help
```

`uv run <command>` runs inside the project environment; no manual activation needed.

## Pipeline

All settings live in [`config/pipeline.yaml`](config/pipeline.yaml). Every step logs to the
console and to `outputs/logs/`, and records every download (URL, time, SHA-256) in
`data/manifest.jsonl`.

| Command | What it does | Main outputs |
|---|---|---|
| `firespread inspect` | Lists layers, columns and CRS of the source datasets | `outputs/*_schema.txt` |
| `firespread socal` | Fires whose first VIIRS centroid is in the 10 SoCal counties; real UTC observation times | `data/processed/socal_perimeters.gpkg`, `outputs/socal_fires.csv` |
| `firespread test-fires` | Matches FRAP Thomas/Woolsey to Largefire IDs (IoU) | `outputs/test_fire_candidates.csv` |
| `firespread split` | Train/val/test by fire with exclusion rules | `outputs/split.csv` |
| `firespread stack --fire-id N` | 30 m grid, 3DEP terrain, perimeter masks, LANDFIRE fuels (COG) | `fires/N/static/*.tif`, `fires/N/perimeters/` |
| `firespread weather --fire-id N` | Hourly HRRR (analyses + 1-h forecast solar/rain), earth-relative winds, QC flags | `fires/N/weather/hrrr_hourly.zarr` |
| `firespread stations --fire-id N` | HRRR vs ASOS/RAWS (bias, RMSE by wind regime, terrain position) | `outputs/fire_N_hrrr_station_*` |
| `firespread fuel-moisture --fire-id N` | Hourly 1/10/100/1000-h dead fuel moisture, Nelson model (NFDRS4) + gridMET | `fires/N/fuel_moisture/dead_fuel_moisture.zarr` |
| `firespread publish --fire-id N` | Time-enabled hosted feature layer in ArcGIS Online (run in ArcGIS Pro's Python) | AGOL item |

`firespread run-fire --fire-id N` runs stack → weather → stations → fuel-moisture.
`--all` instead of `--fire-id` runs every fire listed under `fires:` in the config.

### Reproducing everything with DVC

[`dvc.yaml`](dvc.yaml) declares every step with its inputs and outputs, so

```powershell
uv run dvc repro        # re-runs only what changed
uv run dvc dag          # shows the pipeline graph
```

Manually obtained inputs (Largefire archive, FRAP, county boundaries, LANDFIRE downloads) are
versioned with `dvc add` (see `data/raw/*.dvc`).

## Data conventions

Implemented once in [`src/firespread/io.py`](src/firespread/io.py):

* **Rasters:** Cloud-Optimized GeoTIFF with units, description and scale factors stored in the
  file (LANDFIRE CH/CBH are m × 10 and CBD kg/m³ × 100 on disk; readers get physical units).
* **Gridded time series:** Zarr cubes following **CF-1.10**: coordinates `time` (UTC),
  `latitude`, `longitude`, CF variable names/units/standard names from one registry, HRRR
  Lambert grid mapping.
* **Times:** UTC everywhere; ISO 8601 with `Z` in CSV files. Largefire `time` values are
  half-day *labels*, converted to real VIIRS overpass times (see `perimeters.py`).

## Development

```powershell
uv run pytest             # offline test suite (synthetic data, fake web services)
uv run ruff check .       # lint
uv run pre-commit install # lint automatically on every commit
```

GitHub Actions runs the tests on Windows and Linux for every push.

## Data sources

Chen et al. (2022) VIIRS Largefire / FEDS; CAL FIRE FRAP; USGS 3DEP; LANDFIRE LF2016 Remap and
LF2020; NOAA HRRR (AWS/Google archive via Herbie); Iowa Environmental Mesonet ASOS; Synoptic
RAWS; gridMET (Abatzoglou 2013); USDA Forest Service Missoula Fire Sciences Lab NFDRS4.

## Migrating from the old `scripts/` (one-time)

| Old script | New command |
|---|---|
| 01, 02 | `firespread inspect` |
| 03 | `firespread socal` |
| 04 | `firespread test-fires` |
| 05 | `firespread split` |
| 06 | `firespread publish --fire-id N` |
| 07 | `firespread stack --fire-id N` |
| 08 | `firespread weather --fire-id N` |
| 09 | `firespread stations --fire-id N` |
| 10 | `firespread fuel-moisture --fire-id N` |

Existing downloads are reused: the HRRR hourly cache (`data/raw/hrrr/<id>/`), station data
(`data/raw/stations/<id>/`) and the 3DEP DEM from the old layout. After the new outputs check out,
the old per-fire files directly under `data/processed/fires/<id>/` (`*.tif`, `hrrr_hourly.nc`,
`obs_times.csv`) can be deleted.

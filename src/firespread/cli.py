"""Command line: `firespread <step> [options]`. Run `firespread --help`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import config as config_mod


def _fire_steps(cfg, fire_ids, steps, args):
    from . import fuel_moisture, lfmc_check, live_fuel_moisture, stack, stations, surface_spread, weather

    for fid in fire_ids:
        for step in steps:
            if step == "stack":
                stack.run(cfg, fid, landfire_dir=args.landfire_dir)
            elif step == "weather":
                weather.run(cfg, fid, max_hours=args.max_hours, retry_missing=args.retry_missing)
            elif step == "stations":
                stations.run(cfg, fid, networks=args.networks, refresh=args.refresh)
            elif step == "fuel-moisture":
                fuel_moisture.run(cfg, fid, max_cells=args.max_cells, all_cells=args.all_cells)
            elif step == "live-fuel-moisture":
                live_fuel_moisture.run(cfg, fid)
            elif step == "lfmc-check":
                lfmc_check.run(cfg, fid)
            elif step == "surface-spread":
                surface_spread.run(cfg, fid)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="firespread", description="Hybrid wildfire spread modeling pipeline")
    ap.add_argument("--version", action="version", version=f"firespread {__version__}")
    ap.add_argument("--config", type=Path, help="pipeline YAML (default: config/pipeline.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="list layers/columns/CRS of the source datasets")
    sub.add_parser("socal", help="filter Largefire to the Southern California study area")
    sub.add_parser("test-fires", help="match FRAP named fires to Largefire fire IDs")
    sub.add_parser("split", help="train/val/test split by fire")

    def fire_parser(name, help_):
        p = sub.add_parser(name, help=help_)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--fire-id", type=int, nargs="+", help="one or more fire IDs")
        g.add_argument("--all", action="store_true", help="every fire listed in the config")
        return p

    p = fire_parser("stack", "30 m grid, terrain, perimeters, LANDFIRE fuels")
    p.add_argument("--landfire-dir", type=Path, help="override the LANDFIRE folder chosen by fire year")
    p = fire_parser("weather", "hourly HRRR weather cube")
    p.add_argument("--max-hours", type=int, help="only the first N hours (quick test)")
    p.add_argument("--retry-missing", action="store_true", help="re-check hours the archive lacked last time")
    p = fire_parser("stations", "check HRRR against ASOS/RAWS stations")
    p.add_argument("--networks", nargs="+", default=["asos", "raws"], choices=["asos", "raws"])
    p.add_argument("--refresh", action="store_true", help="ignore cached station downloads")
    p = fire_parser("fuel-moisture", "hourly Nelson dead fuel moisture (NFDRS4)")
    p.add_argument(
        "--max-cells", type=int, default=0, help="only N cells nearest ignition; writes *.test.zarr"
    )
    p.add_argument("--all-cells", action="store_true", help="include ocean cells")
    fire_parser("live-fuel-moisture", "daily live herbaceous/woody fuel moisture (NFDRS4 GSI, gridMET)")
    fire_parser("lfmc-check", "compare live fuel moisture with Globe-LFMC field samples (chamise)")
    fire_parser("surface-spread", "Rothermel spread per perimeter step + check against observed advance")
    p = fire_parser(
        "run-fire", "stack -> weather -> stations -> fuel-moisture -> live-fuel-moisture -> surface-spread"
    )
    p.add_argument("--landfire-dir", type=Path)
    p = fire_parser("publish", "publish observed perimeters to ArcGIS Online (needs arcgis)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="prepare the GeoPackage only")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.load(args.config)
    fire_ids = cfg.fires if getattr(args, "all", False) else getattr(args, "fire_id", None)
    try:
        if args.cmd == "inspect":
            from .catalog import inspect_sources

            inspect_sources(cfg)
        elif args.cmd == "socal":
            from .catalog import filter_socal

            filter_socal(cfg)
        elif args.cmd == "test-fires":
            from .catalog import find_test_fires

            find_test_fires(cfg)
        elif args.cmd == "split":
            from .split import make_split

            make_split(cfg)
        elif args.cmd == "publish":
            from . import publish

            for fid in fire_ids:
                publish.run(cfg, fid, overwrite=args.overwrite, dry_run=args.dry_run)
        elif args.cmd == "run-fire":
            defaults = argparse.Namespace(
                landfire_dir=args.landfire_dir,
                max_hours=None,
                retry_missing=False,
                networks=["asos", "raws"],
                refresh=False,
                max_cells=0,
                all_cells=False,
            )
            _fire_steps(
                cfg,
                fire_ids,
                ["stack", "weather", "stations", "fuel-moisture", "live-fuel-moisture", "surface-spread"],
                defaults,
            )
        else:
            _fire_steps(cfg, fire_ids, [args.cmd], args)
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as e:
        import logging

        logging.getLogger("firespread").error("FAILED: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

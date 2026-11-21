"""Publish a fire's observed perimeters to ArcGIS Online as a time-enabled hosted
feature layer (formerly script 06).

Fixes over the original script:
  * times are the real VIIRS observation times (UTC), not the Largefire labels
  * each perimeter is valid until the next observation (not a fixed +12 h)
  * sign-in uses ArcGIS Pro's session (works with UCR single sign-on), or OAuth

Run in ArcGIS Pro's Python (arcgispro-py3) with this package installed, or any
environment with the `publish` extra.
"""

from __future__ import annotations

import getpass

import pandas as pd

from . import logs
from .config import Config
from .perimeters import load_records


def prepare(cfg: Config, fire_id: int):
    g = load_records(cfg.socal_perimeters, fire_id)
    g["obs_time"] = g["obs_time_utc"]
    nxt = g["obs_time_utc"].shift(-1)
    g["end_time"] = nxt.fillna(g["obs_time_utc"].iloc[-1] + pd.Timedelta(hours=12))
    g["step"] = range(len(g))
    g["label"] = g["label_time"].dt.strftime("%Y-%m-%d %H:%M")
    bad = ~g.is_valid
    g.loc[bad, "geometry"] = g.loc[bad, "geometry"].make_valid()
    if not set(g.geom_type) <= {"Polygon", "MultiPolygon"}:
        raise ValueError(f"unexpected geometry types {set(g.geom_type)}")
    cols = [
        "fireID",
        "step",
        "label",
        "obs_time",
        "end_time",
        "farea",
        "fperim",
        "n_pixels",
        "n_newpixels",
        "meanFRP",
        "geometry",
    ]
    g = g[[c for c in cols if c in g.columns]].to_crs(4326)
    out = cfg.fire_dir(fire_id) / "publish" / f"fire_{fire_id}_observed.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out, layer="perimeter", driver="GPKG")
    return g, out


def connect(cfg: Config):
    from arcgis.gis import GIS

    p = cfg.publish
    if p.auth == "pro":
        return GIS("pro")
    if p.auth == "oauth":
        return GIS(p.portal_url, client_id=p.client_id)
    if p.auth == "builtin":
        return GIS(p.portal_url, input("ArcGIS username: "), getpass.getpass("ArcGIS password: "))
    raise ValueError(f"unknown publish.auth '{p.auth}'")


def run(cfg: Config, fire_id: int, overwrite: bool = False, dry_run: bool = False) -> str | None:
    log = logs.setup(cfg.outputs_dir / "logs", "publish", fire_id)
    g, path = prepare(cfg, fire_id)
    log.info("%d perimeters, %s to %s UTC -> %s", len(g), g["obs_time"].min(), g["obs_time"].max(), path)
    if dry_run:
        return None
    from arcgis.features import GeoAccessor

    gis = connect(cfg)
    me = gis.users.me.username
    title = cfg.publish.title_template.format(fire_id=fire_id)
    log.info("Signed in as %s", me)
    existing = gis.content.search(f'title:"{title}" AND owner:{me}')
    if existing:
        if not overwrite:
            raise RuntimeError(f"{len(existing)} item(s) titled {title} exist; re-run with --overwrite")
        for it in existing:
            it.delete()
    sdf = GeoAccessor.from_geodataframe(g)
    item = sdf.spatial.to_featurelayer(title, gis=gis)
    item.update(
        item_properties={
            "tags": "wildfire, VIIRS, FEDS, capstone",
            "snippet": f"Observed perimeters of fire {fire_id} at VIIRS overpass times (UTC); Chen et al. 2022.",
        }
    )
    item.layers[0].manager.update_definition(
        {
            "timeInfo": {
                "startTimeField": "obs_time",
                "endTimeField": "end_time",
                "timeReference": {"timeZone": "UTC"},
                "timeInterval": 12,
                "timeIntervalUnits": "esriTimeUnitsHours",
            }
        }
    )
    log.info("Published %s", item.homepage)
    return item.homepage

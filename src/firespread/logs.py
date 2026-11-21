"""Logging setup and plain-text run reports.

Every step logs to the console and appends to outputs/logs/<step>[_<fire>].log.
Human-readable summaries (outputs/*_summary.txt) are written with Report.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import __version__

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup(log_dir: Path, step: str, fire_id: int | None = None, level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{step}_{fire_id}" if fire_id is not None else step
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(level)
    fileh = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
    fmt = logging.Formatter(_FMT, "%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = time.gmtime
    fileh.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(fileh)
    for noisy in ("urllib3", "botocore", "fiona", "rasterio", "matplotlib", "herbie", "cfgrib", "numcodecs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger(f"firespread.{step}")
    log.info(
        "=== firespread %s | step %s%s | %s UTC ===",
        __version__,
        step,
        f" | fire {fire_id}" if fire_id is not None else "",
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )
    return log


class Report:
    """Collects summary lines, logging each one as it is added."""

    def __init__(self, log: logging.Logger):
        self.log = log
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        self.log.info(line)
        self.lines.append(line)

    def block(self, text: str) -> None:
        for line in text.splitlines():
            self(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.log.info("Wrote %s", path)

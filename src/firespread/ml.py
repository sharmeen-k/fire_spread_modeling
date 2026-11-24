"""ML-only and hybrid next-step perimeter models (PyTorch), trained on the sample shards.

Three configurations share one U-Net and one training loop, differing only in inputs/output:
- "ml": no physics inputs (the ML-only baseline);
- "physics_features": the Rothermel channels added as inputs;
- "hybrid": residual on the calibrated physics forecast: logit(p) = logit(p_phys) + U-Net(x), with the last
  layer zero-initialised so training starts from the physics forecast (requires the `phys_prob` channel
  written by the spread engine).

The model predicts the probability that each still-unburned block burns during the step; burned blocks
stay burned (perimeters are cumulative): p_t1 = b_t + (1 - b_t) * p_new.

Data: one Zarr shard per fire (`firespread build-samples`), plus outputs/split.csv. Training uses random
tiles centred on the fire front with dihedral augmentation that rotates/flips vector channels (wind, spread
direction, aspect) consistently. Evaluation runs on whole fires.

Metrics (blocks with >= 50 % burned counted as burned): IoU of the burned area at t+1, IoU of the newly
burned area (the hard part; persistence scores 0), and the predicted/observed new-area ratio.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is an optional dependency (extra "ml")
    torch = None

from .channels import FUEL_GROUPS, PHYSICS

PHYS_PROB = "phys_prob"
VECTORS = [("u10_mean", "v10_mean"), ("ros_x", "ros_y"), ("aspect_sin", "aspect_cos")]  # (x east, y north)
NO_NORM = {"burned_t", "nodata", PHYS_PROB, *[f"fuel_{g}" for g in FUEL_GROUPS]}
MODES = ("ml", "physics_features", "hybrid")


def mode_channels(mode: str, available: list[str]) -> list[str]:
    if mode == "ml":
        return [c for c in available if c not in PHYSICS and c != PHYS_PROB]
    if mode == "physics_features":
        return [c for c in available if c != PHYS_PROB]
    if mode == "hybrid":
        if PHYS_PROB not in available:
            raise ValueError("hybrid mode needs the phys_prob channel (run the spread engine first)")
        return list(available)
    raise ValueError(f"mode must be one of {MODES}")


# ------------------------------------------------------------------------- data
@dataclass
class Fire:
    fire_id: int
    split: str
    x: np.ndarray  # (S, C, H, W) float32, NaN outside the grid
    y: np.ndarray  # (S, H, W)
    valid: np.ndarray  # (H, W) bool
    channels: list[str]
    t_start: np.ndarray = field(default=None)


def load_shards(sample_dir: Path, split_csv: Path, splits=("train", "val", "test")) -> list[Fire]:
    """Load every <fireID>.zarr (or fires/<id>/samples/samples.zarr) under sample_dir."""
    import xarray as xr

    sp = pd.read_csv(split_csv).set_index("fireID")["split"]
    paths = sorted(Path(sample_dir).glob("*.zarr")) or sorted(Path(sample_dir).glob("*/samples/samples.zarr"))
    fires = []
    for p in paths:
        ds = xr.open_zarr(p, zarr_format=3, consolidated=False).load()
        fid = int(ds.attrs["fireID"])
        s = sp.get(fid, "excluded")
        if s not in splits:
            continue
        chans = list(ds["channel"].values)
        x = ds["inputs"].values.astype("float32")
        if PHYS_PROB in ds:
            x = np.concatenate([x, ds[PHYS_PROB].values[:, None].astype("float32")], 1)
            chans = [*chans, PHYS_PROB]
        fires.append(
            Fire(
                fid,
                s,
                x,
                ds["target"].values.astype("float32"),
                ds["valid"].values.astype(bool),
                chans,
                ds["t_start"].values,
            )
        )
    return fires


@dataclass
class Normalizer:
    channels: list[str]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, fires: list[Fire], channels: list[str]) -> Normalizer:
        idx = [fires[0].channels.index(c) for c in channels]
        vals = [
            f.x[:, idx][:, :, f.valid].transpose(1, 0, 2).reshape(len(idx), -1)
            for f in fires
            if f.split == "train"
        ]
        v = np.concatenate(vals, 1).astype("float64")
        mean, std = np.nanmean(v, 1), np.nanstd(v, 1)
        for i, c in enumerate(channels):
            if c in NO_NORM or std[i] < 1e-6:
                mean[i], std[i] = 0.0, 1.0
            if any(c in pair for pair in VECTORS):  # keep vectors centred at 0 so rotations stay valid
                mean[i] = 0.0
        # vector pairs share one scale
        for a, b in VECTORS:
            if a in channels and b in channels:
                ia, ib = channels.index(a), channels.index(b)
                s = math.sqrt((std[ia] ** 2 + std[ib] ** 2) / 2)
                std[ia] = std[ib] = s
        return cls(channels, mean.astype("float32"), std.astype("float32"))

    def apply(self, fire: Fire) -> np.ndarray:
        idx = [fire.channels.index(c) for c in self.channels]
        x = (fire.x[:, idx] - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return np.nan_to_num(x, nan=0.0).astype("float32")

    def to_json(self) -> dict:
        return {"channels": self.channels, "mean": self.mean.tolist(), "std": self.std.tolist()}


def augment(x: np.ndarray, y: np.ndarray, m: np.ndarray, k: int, flip: bool, channels: list[str]):
    """Dihedral transform of (C,H,W) inputs, (H,W) target and mask; vector channels rotated consistently.

    Arrays are north-up (row 0 = north). np.rot90 turns the picture counter-clockwise, which maps a vector
    (x, y) to (-y, x); a left-right flip maps (x, y) to (-x, y).
    """
    pairs = [(channels.index(a), channels.index(b)) for a, b in VECTORS if a in channels and b in channels]
    if flip:
        x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
        for ia, _ in pairs:
            x[ia] = -x[ia]
    for _ in range(k % 4):
        x, y, m = np.rot90(x, 1, (1, 2)), np.rot90(y), np.rot90(m)
        for ia, ib in pairs:
            xa, xb = x[ia].copy(), x[ib].copy()
            x[ia], x[ib] = -xb, xa
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(m)


class TileDataset:
    """Random training tiles, centred on the fire front with probability `p_front`."""

    def __init__(
        self, fires, xs, channels, tile=64, p_front=0.8, augment_=True, samples_per_epoch=2000, seed=0
    ):
        self.fires, self.xs, self.tile, self.p_front, self.aug = fires, xs, tile, p_front, augment_
        self.channels = list(channels)
        self.n = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.index = []  # (fire i, sample s, front cells)
        for i, f in enumerate(fires):
            bt = f.x[:, f.channels.index("burned_t")]
            for s in range(f.x.shape[0]):
                new = (f.y[s] - bt[s] > 0.05) & f.valid
                edge = (bt[s] > 0.02) & (bt[s] < 0.98) & f.valid
                front = np.argwhere(new | edge)
                self.index.append((i, s, front))

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        i, s, front = self.index[self.rng.integers(len(self.index))]
        f, x = self.fires[i], self.xs[i][s]
        h, w = f.valid.shape
        if len(front) and self.rng.random() < self.p_front:
            cy, cx = front[self.rng.integers(len(front))]
        else:
            vy, vx = np.nonzero(f.valid)
            j = self.rng.integers(len(vy))
            cy, cx = vy[j], vx[j]
        t = self.tile
        y0, x0 = (
            cy - t // 2 + self.rng.integers(-t // 4, t // 4 + 1),
            cx - t // 2 + self.rng.integers(-t // 4, t // 4 + 1),
        )
        xt = np.zeros((x.shape[0], t, t), "float32")
        yt = np.zeros((t, t), "float32")
        mt = np.zeros((t, t), "float32")
        sy0, sx0 = max(y0, 0), max(x0, 0)
        sy1, sx1 = min(y0 + t, h), min(x0 + t, w)
        if sy1 > sy0 and sx1 > sx0:
            dy, dx = slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0)
            xt[:, dy, dx] = x[:, sy0:sy1, sx0:sx1]
            yt[dy, dx] = np.nan_to_num(f.y[s, sy0:sy1, sx0:sx1])
            mt[dy, dx] = f.valid[sy0:sy1, sx0:sx1]
        if self.aug:
            xt, yt, mt = augment(
                xt, yt, mt, int(self.rng.integers(4)), bool(self.rng.integers(2)), self.channels
            )
        return xt, yt, mt


# ------------------------------------------------------------------------ model
if torch is not None:

    class Block(nn.Module):
        def __init__(self, cin, cout, dropout=0.0):
            super().__init__()
            g = 8 if cout % 8 == 0 else 1
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(g, cout),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(g, cout),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        """4-level U-Net (Ronneberger et al. 2015) with GroupNorm; output: logits (B,1,H,W)."""

        def __init__(self, cin, base=32, depth=4, dropout=0.1):
            super().__init__()
            ch = [base * 2**i for i in range(depth + 1)]
            self.enc = nn.ModuleList(
                [Block(cin, ch[0])] + [Block(ch[i], ch[i + 1], dropout) for i in range(depth)]
            )
            self.up = nn.ModuleList(
                [nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))]
            )
            self.dec = nn.ModuleList([Block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
            self.head = nn.Conv2d(ch[0], 1, 1)
            self.depth = depth

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x if i == 0 else F.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], 1))
            return self.head(x)

    class FireModel(nn.Module):
        """Wraps the U-Net with the cumulative-burn output and, in hybrid mode, the physics residual."""

        def __init__(self, channels, mode, **kw):
            super().__init__()
            self.channels, self.mode = list(channels), mode
            self.ib = self.channels.index("burned_t")
            self.ip = self.channels.index(PHYS_PROB) if mode == "hybrid" else None
            self.unet = UNet(len(channels), **kw)
            if mode == "hybrid":  # start exactly at the physics forecast
                nn.init.zeros_(self.unet.head.weight)
                nn.init.zeros_(self.unet.head.bias)

        def forward(self, x):
            bt = x[:, self.ib : self.ib + 1].clamp(0, 1)
            z = self.unet(x)
            if self.ip is not None:
                pp = x[:, self.ip : self.ip + 1].clamp(0, 1)
                p_new_phys = ((pp - bt) / (1 - bt).clamp_min(1e-3)).clamp(1e-4, 1 - 1e-4)
                z = z + torch.logit(p_new_phys)
            p_new = torch.sigmoid(z)
            return (bt + (1 - bt) * p_new).squeeze(1), bt.squeeze(1)

    def loss_fn(p, bt, y, m, dice_w=0.5):
        """BCE on not-yet-burned blocks + soft Dice on the newly burned area."""
        w = m * (bt < 0.99)
        bce = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y.clamp(0, 1), reduction="none")
        bce = (bce * w).sum() / w.sum().clamp_min(1)
        pn, yn = (p - bt).clamp_min(0) * m, (y - bt).clamp_min(0) * m
        dice = 1 - (2 * (pn * yn).sum() + 1) / (pn.sum() + yn.sum() + 1)
        return bce + dice_w * dice


# --------------------------------------------------------------------- metrics
def step_metrics(p: np.ndarray, bt: np.ndarray, y: np.ndarray, valid: np.ndarray, thr=0.5) -> dict:
    v = valid & np.isfinite(y)
    pb, yb, bb = (p >= thr) & v, (y >= thr) & v, (bt >= thr) & v
    pn, yn = pb & ~bb, yb & ~bb
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)  # noqa: E731
    return dict(
        iou=iou(pb, yb),
        iou_new=iou(pn, yn) if (pn | yn).any() else np.nan,
        new_obs=int(yn.sum()),
        new_pred=int(pn.sum()),
        tp_new=int((pn & yn).sum()),
        union_new=int((pn | yn).sum()),
        area_ratio=pn.sum() / max(yn.sum(), 1),
    )


def summarize(rows: pd.DataFrame) -> dict:
    """Mean per-step IoUs plus pooled (area-weighted) new-burn IoU over steps with observed growth."""
    if len(rows) == 0:
        return dict(
            steps=0,
            steps_growing=0,
            iou=np.nan,
            iou_new=np.nan,
            iou_new_pooled=np.nan,
            area_ratio_median=np.nan,
        )
    g = rows[rows["new_obs"] > 0]
    return dict(
        steps=len(rows),
        steps_growing=len(g),
        iou=rows["iou"].mean(),
        iou_new=g["iou_new"].mean(),
        iou_new_pooled=g["tp_new"].sum() / max(g["union_new"].sum(), 1),
        area_ratio_median=g["area_ratio"].median(),
    )


# -------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    mode: str = "ml"
    epochs: int = 40
    batch: int = 32
    tile: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    base: int = 32
    dropout: float = 0.1
    samples_per_epoch: int = 4000
    p_front: float = 0.8
    seed: int = 42
    workers: int = 2


def _pad16(a, mult=16):
    h, w = a.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)]), (h, w)


def predict_fire(model, fire: Fire, x: np.ndarray, device, batch=4) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, x.shape[0], batch):
            xb, (h, w) = _pad16(x[a : a + batch])
            p, _ = model(torch.from_numpy(xb).to(device))
            out.append(p.float().cpu().numpy()[:, :h, :w])
    return np.concatenate(out)


def evaluate(model, fires, xs, device, split) -> pd.DataFrame:
    rows = []
    for f, x in zip(fires, xs):
        if f.split != split:
            continue
        p = predict_fire(model, f, x, device)
        bt = np.nan_to_num(f.x[:, f.channels.index("burned_t")])
        for s in range(p.shape[0]):
            rows.append(dict(fire=f.fire_id, step=s, **step_metrics(p[s], bt[s], f.y[s], f.valid)))
    return pd.DataFrame(rows)


def persistence(fires, split) -> pd.DataFrame:
    rows = []
    for f in fires:
        if f.split == split:
            bt = np.nan_to_num(f.x[:, f.channels.index("burned_t")])
            for s in range(bt.shape[0]):
                rows.append(dict(fire=f.fire_id, step=s, **step_metrics(bt[s], bt[s], f.y[s], f.valid)))
    return pd.DataFrame(rows)


def train(fires: list[Fire], cfg: TrainConfig, out_dir: Path, device=None, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = mode_channels(cfg.mode, fires[0].channels)
    norm = Normalizer.fit(fires, channels)
    xs = [norm.apply(f) for f in fires]
    train_idx = [i for i, f in enumerate(fires) if f.split == "train"]
    ds = TileDataset(
        [fires[i] for i in train_idx],
        [xs[i] for i in train_idx],
        channels,
        cfg.tile,
        cfg.p_front,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=cfg.seed,
    )
    model = FireModel(channels, cfg.mode, base=cfg.base, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(ds) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    use_amp = device == "cuda"
    hist, best, best_state = [], -1.0, None
    rng = np.random.default_rng(cfg.seed)
    for ep in range(cfg.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        nb = math.ceil(len(ds) / cfg.batch)
        for _ in range(nb):
            items = [ds[int(rng.integers(len(ds)))] for _ in range(cfg.batch)]
            xb, yb, mb = (torch.from_numpy(np.stack(a)).to(device) for a in zip(*items))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                p, bt = model(xb)
            loss = loss_fn(p.float(), bt.float(), yb, mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        val = summarize(evaluate(model, fires, xs, device, "val"))
        hist.append(
            dict(
                epoch=ep + 1,
                loss=tot / nb,
                **{f"val_{k}": v for k, v in val.items()},
                seconds=time.time() - t0,
            )
        )
        score = np.nan_to_num(val["iou_new_pooled"]) if val["steps"] else ep  # no val fires: keep the last
        if score >= best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(
            f"epoch {ep + 1:3d}  loss {tot / nb:.4f}  val IoU {val['iou']:.3f}  new-burn IoU (pooled) "
            f"{val['iou_new_pooled']:.3f}  {time.time() - t0:.0f}s"
        )
    model.load_state_dict(best_state)
    res = {"config": asdict(cfg), "history": hist}
    for split in ("val", "test"):
        rows = evaluate(model, fires, xs, device, split)
        if len(rows):
            rows.to_csv(out_dir / f"{cfg.mode}_{split}_steps.csv", index=False)
            res[split] = summarize(rows)
            res[split + "_persistence"] = summarize(persistence(fires, split))
    torch.save(
        {"state_dict": model.state_dict(), "normalizer": norm.to_json(), "config": asdict(cfg)},
        out_dir / f"{cfg.mode}.pt",
    )
    (out_dir / f"{cfg.mode}_results.json").write_text(json.dumps(res, indent=2, default=float))
    return res

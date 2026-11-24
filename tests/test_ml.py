"""ML module: augmentation geometry, metrics, a tiny training run, and the hybrid's physics start."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from firespread import ml  # noqa: E402
from firespread.channels import CHANNELS  # noqa: E402


def synthetic_fires(n_fires=4, S=3, H=24, W=28, phys=False, seed=0):
    rng = np.random.default_rng(seed)
    chans = list(CHANNELS) + ([ml.PHYS_PROB] if phys else [])
    fires = []
    for i in range(n_fires):
        x = rng.normal(size=(S, len(chans), H, W)).astype("float32")
        yy, xx = np.mgrid[0:H, 0:W]
        y = np.zeros((S, H, W), "float32")
        for s in range(S):
            r0 = 3 + 2 * s
            bt = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2 <= r0**2).astype("float32")
            x[s, chans.index("burned_t")] = bt
            y[s] = ((yy - H / 2) ** 2 + (xx - W / 2 - 1) ** 2 <= (r0 + 2) ** 2) | (bt > 0)
            if phys:
                x[s, chans.index(ml.PHYS_PROB)] = y[s]
        split = ["train", "train", "val", "test"][i % 4]
        fires.append(ml.Fire(1000 + i, split, x, y, np.ones((H, W), bool), chans))
    return fires


def test_augment_rotates_vectors_with_the_picture():
    chans = list(CHANNELS)
    iu, iv = chans.index("u10_mean"), chans.index("v10_mean")
    x = np.zeros((len(chans), 4, 4), "float32")
    x[iu] = 1.0  # wind towards the east
    x[chans.index("burned_t"), 0, 3] = 1  # a burned cell in the north-east corner
    y, m = x[chans.index("burned_t")].copy(), np.ones((4, 4), "float32")
    xr_, yr_, _ = ml.augment(x.copy(), y, m, 1, False, chans)  # 90 deg counter-clockwise
    assert np.allclose(xr_[iu], 0) and np.allclose(xr_[iv], 1)  # east -> north
    assert yr_[0, 0] == 1  # north-east corner -> north-west corner
    xf, yf, _ = ml.augment(x.copy(), y, m, 0, True, chans)  # left-right flip
    assert np.allclose(xf[iu], -1) and yf[0, 0] == 1


def test_step_metrics():
    bt = np.zeros((4, 4))
    bt[0, 0] = 1
    y = bt.copy()
    y[0, 1] = y[1, 0] = 1
    p = bt.copy()
    p[0, 1] = 1
    m = ml.step_metrics(p, bt, y, np.ones((4, 4), bool))
    assert m["iou"] == pytest.approx(2 / 3) and m["iou_new"] == pytest.approx(1 / 2)
    assert ml.step_metrics(bt, bt, y, np.ones((4, 4), bool))["iou_new"] == 0  # persistence


def test_mode_channels():
    assert not set(ml.mode_channels("ml", CHANNELS)) & set(ml.PHYSICS)
    assert set(ml.PHYSICS) <= set(ml.mode_channels("physics_features", CHANNELS))
    with pytest.raises(ValueError):
        ml.mode_channels("hybrid", CHANNELS)


def test_tiny_training_run(tmp_path):
    fires = synthetic_fires()
    cfg = ml.TrainConfig(mode="physics_features", epochs=2, batch=4, tile=32, base=8, samples_per_epoch=8)
    res = ml.train(fires, cfg, tmp_path, device="cpu", log=lambda *_: None)
    assert {"val", "test", "val_persistence"} <= set(res) and len(res["history"]) == 2
    assert (tmp_path / "physics_features.pt").exists()
    assert res["val_persistence"]["iou_new_pooled"] == 0


def test_hybrid_starts_at_the_physics_forecast():
    fires = synthetic_fires(phys=True)
    chans = ml.mode_channels("hybrid", fires[0].channels)
    norm = ml.Normalizer.fit(fires, chans)
    model = ml.FireModel(chans, "hybrid", base=8)
    rows = ml.evaluate(model, fires, [norm.apply(f) for f in fires], "cpu", "val")
    assert ml.summarize(rows)["iou_new_pooled"] == pytest.approx(1.0)  # physics = truth here

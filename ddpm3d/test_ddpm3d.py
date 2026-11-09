"""Tests for ddpm3d. Run with:  python -m pytest -q ddpm3d"""
import json
import os

import numpy as np
import pytest
import torch

from ddpm3d import train as train_mod
from ddpm3d import select as select_mod
from ddpm3d import sample as sample_mod
from ddpm3d.data import PhaseCodec
from ddpm3d.diffusion import GaussianDiffusion, make_alphas_cumprod
from ddpm3d.unet import UNet3D

TINY = ["--base_channels", "8", "--channel_mult", "1,2", "--num_res_blocks", "1,2",
        "--attention_resolutions", "4", "--head_channels", "8", "--T", "20",
        "--batch_size", "2", "--bf16", "0", "--log_every", "1"]


def _toy_model(x, t):
    return 0.3 * x * torch.sin(t.float() + 1.0).view(-1, *([1] * (x.ndim - 1)))


@pytest.mark.parametrize("sc", [False, True])
def test_unet_shapes_and_zero_init(sc):
    for c in (1, 3):
        m = UNet3D(c, c, image_size=16, base_channels=8, channel_mult=(1, 2), num_res_blocks=(1, 2),
                   attention_resolutions=(8,), head_channels=8, self_condition=sc)
        x = torch.randn(2, c, 16, 16, 16)
        y = m(x, torch.tensor([0, 7]), torch.randn_like(x) if sc else None)
        assert y.shape == x.shape
        assert torch.count_nonzero(y) == 0  # zero-initialised output layer


@pytest.mark.parametrize("k,enc", [(2, "scalar"), (3, "scalar"), (3, "onehot")])
def test_codec_roundtrip(k, enc):
    codec = PhaseCodec(k, enc)
    lab = torch.randint(k, (2, 5, 5, 5), dtype=torch.uint8)
    x = codec.encode(lab)
    assert x.shape[1] == codec.channels and x.min() >= -1 and x.max() <= 1
    assert torch.equal(codec.decode(x + 0.2 * torch.randn_like(x).clamp(-0.4, 0.4) * 0), lab)


def test_schedule_properties():
    ab = make_alphas_cumprod("cosine", 1000, 0.125, zero_terminal_snr=True)
    assert ab[-1] == 0 and np.all(np.diff(ab) < 0) and ab[0] > 0.999
    a1 = make_alphas_cumprod("cosine", 1000, 1.0, zero_terminal_snr=False)
    a2 = make_alphas_cumprod("cosine", 1000, 0.125, zero_terminal_snr=False)
    logit = lambda a: np.log(a) - np.log1p(-a)
    sel = slice(0, 990)
    assert np.allclose(logit(a2[sel]) - logit(a1[sel]), np.log(0.125), atol=1e-6)
    with pytest.raises(ValueError):
        GaussianDiffusion(ab, prediction="eps")


def test_ddim_timesteps():
    d = GaussianDiffusion(make_alphas_cumprod("cosine", 1000))
    ts = d.ddim_timesteps(50)
    assert ts[0] == 0 and ts[-1] == 999 and len(ts) == 50 and np.all(np.diff(ts) > 0)


@pytest.mark.parametrize("pred,zts", [("v", True), ("v", False), ("eps", False)])
def test_ddim_eta1_full_equals_ddpm(pred, zts):
    d = GaussianDiffusion(make_alphas_cumprod("cosine", 20, 0.125, zts), prediction=pred)
    shape = (2, 1, 4, 4, 4)
    a = d.ddpm_sample(_toy_model, shape, "cpu", generator=torch.Generator().manual_seed(0))
    b = d.ddim_sample(_toy_model, shape, "cpu", steps=20, eta=1.0, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_samples_in_range():
    d = GaussianDiffusion(make_alphas_cumprod("cosine", 50, 0.125, True))
    for x in (d.ddim_sample(_toy_model, (2, 1, 4, 4, 4), "cpu", steps=10),
              d.dpmpp_sample(_toy_model, (2, 1, 4, 4, 4), "cpu", steps=10)):
        assert torch.isfinite(x).all() and x.abs().max() <= 1.0 + 1e-5


@pytest.mark.parametrize("pred,zts,sc", [("v", True, True), ("v", True, False), ("eps", False, False)])
def test_point_mass_is_recovered_by_every_sampler(pred, zts, sc):
    """With the exact denoiser for a single data point, every sampler must return it."""
    target = 0.3
    d = GaussianDiffusion(make_alphas_cumprod("cosine", 50, 0.125, zts), prediction=pred, self_condition=sc)

    def exact(x, t, x_sc=None):
        a = d.sqrt_ab[t].float().view(-1, 1, 1, 1, 1)
        s = d.sqrt_1m_ab[t].float().view(-1, 1, 1, 1, 1)
        x0 = torch.full_like(x, target)
        eps = (x - a * x0) / s
        return a * eps - s * x0 if pred == "v" else eps

    shape = (2, 1, 3, 3, 3)
    g = lambda: torch.Generator().manual_seed(1)
    outs = [d.ddpm_sample(exact, shape, "cpu", generator=g()),
            d.ddim_sample(exact, shape, "cpu", steps=10, generator=g()),
            d.ddim_sample(exact, shape, "cpu", steps=10, eta=1.0, generator=g()),
            d.dpmpp_sample(exact, shape, "cpu", steps=10, generator=g())]
    for x in outs:
        assert torch.allclose(x, torch.full_like(x, target), atol=1e-4), (x - target).abs().max()


def _make_data(path, n=12, size=16, k=2):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        g = gaussian_filter(rng.standard_normal((size,) * 3), 1.5)
        lab = np.digitize(g, np.quantile(g, np.linspace(0, 1, k + 1)[1:-1])).astype(np.uint8)
        np.save(os.path.join(path, f"v{i:03d}.npy"), lab)


def _params(run):
    ck = torch.load(os.path.join(run, "last.pt"), weights_only=False)
    return [ck["model"]] + [e["model"] for e in ck["emas"]]


def test_resume_is_exact(tmp_path):
    data = str(tmp_path / "data")
    _make_data(data)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    common = ["--data_dir", data, "--save_every", "3", "--snapshot_every", "100", "--preview", "0"] + TINY
    assert train_mod.main(["--out_dir", a, "--iters", "6"] + common) == 0
    assert train_mod.main(["--out_dir", b, "--iters", "3"] + common) == 0
    os.remove(os.path.join(b, "TRAINING_DONE"))
    assert train_mod.main(["--out_dir", b, "--iters", "6"] + common) == 0
    for sa, sb in zip(_params(a), _params(b)):
        for key in sa:
            assert torch.equal(sa[key], sb[key]), key


@pytest.mark.parametrize("k", [2, 3])
def test_end_to_end(tmp_path, k):
    data = str(tmp_path / "data")
    _make_data(data, k=k)
    run = str(tmp_path / "run")
    assert train_mod.main(["--data_dir", data, "--out_dir", run, "--iters", "4",
                           "--snapshot_every", "2", "--save_every", "2"] + TINY) == 0
    assert len(os.listdir(os.path.join(run, "snapshots"))) == 4  # 2 steps x 2 EMA decays
    assert select_mod.main(["--run_dir", run, "--n_ref", "8", "--n_eval", "2", "--steps", "5",
                            "--batch", "2", "--max_lag", "4", "--bf16", "0"]) == 0
    best = json.load(open(os.path.join(run, "selection", "best.json")))
    assert best["snapshot"].startswith("ema")
    for sampler in ("ddpm", "ddim", "dpmpp"):
        out = os.path.join(run, "samples", sampler)
        assert sample_mod.main(["--ckpt", os.path.join(run, "best.pt"), "--out_dir", out, "--n", "3",
                                "--sampler", sampler, "--steps", "5", "--batch", "2", "--bf16", "0"]) == 0
        files = sorted(os.listdir(os.path.join(out, "labels")))
        assert len(files) == 3
        v = np.load(os.path.join(out, "labels", files[0]))
        assert v.shape == (16, 16, 16) and v.dtype == np.uint8 and set(np.unique(v)) <= set(range(k))
        timing = json.load(open(os.path.join(out, "timing.json")))
        assert timing["network_evaluations_per_sample"] == (20 if sampler == "ddpm" else 5)

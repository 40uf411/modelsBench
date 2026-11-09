# Audit of the previous 3D DDPM code (`3DMPDDPM/`)

Scope: the multiphase pipeline that `3DMPDDPM/MainMP.py` runs (`DiffusionMP/`),
plus the DDIM variants (`MainMP_DDIM.py`, `DiffusionMP_DDIM/`, `DDIM/`).
Line numbers refer to commit `4dbdd03`.

**Verdict:** the code on `main` cannot have produced the paper's DDPM samples.
The U-Net raises a shape error on its first forward pass (checked:
`RuntimeError: The size of tensor a (5) must match the size of tensor b (17)`).
The ancestral sampler stops after one step, and the data path feeds 1-channel
{0,1} volumes to a 2-channel model expecting [-1,1]. `ddpm3d/` replaces it.

## Breaks the run

| # | Where | Problem |
| --- | --- | --- |
| 1 | `DiffusionMP/Model.py:113, 122` | Both convolutions in every `ResBlock` use `stride=2`. Each block shrinks the volume 4×, while the shortcut keeps full size, so `h + shortcut(x)` fails. |
| 2 | `DiffusionMP/Model.py:114` | `BatchNorm3d(in_ch)` follows a convolution with `out_ch` outputs, so any block with `in_ch != out_ch` fails. |
| 3 | `DiffusionMP/Model.py:59` | `UpSample` interpolates ×2, then applies a stride-2 convolution, so the net resolution is unchanged and skip connections cannot match. |
| 4 | `DiffusionMP/Model.py:154` | The head convolution has stride 2, so the output is smaller than the input and the noise target. |
| 5 | `DiffusionMP/Diffusion.py:78, 85` | `GaussianDiffusionSampler.forward` prints every step and `break`s after the first, so DDPM sampling is one denoising step from pure noise. |
| 6 | `DiffusionMP/Train.py:17` + `image_datasets.py:31` | Data are read raw (values {0,1}, one channel). The model has `number_of_phases = 2` input channels, and the sampler assumes [-1,1] (`clip(-1,1)`, then `*0.5 + 0.5`, line 78). |
| 7 | `DiffusionMP/Train.py:17` | The dataset path is hard-coded (`./datasets/illite_2p_64`), whatever the config says. |

## Wrong results even once it runs

| # | Where | Problem |
| --- | --- | --- |
| 8 | `DiffusionMP/Diffusion.py:98` | `ImprovedDDIMSampler` hard-codes a cosine β schedule; the model is trained with a linear schedule (line 24). Every DDIM step uses the wrong noise level. |
| 9 | `DiffusionMP/Diffusion.py:118` + `forward` | DDIM timesteps are `linspace(0, T-1, S)` iterated from index 0, so sampling runs from t = 0 towards t = 999 and then jumps 999 → 0 in one step. |
| 10 | `DiffusionMP/Train.py:76-78` (`eval`) | `eval()` calls the DDIM sampler, not DDPM. The noise shape is hard-coded to `[B, 2, 64, 64, 64]`, and samples are saved as continuous values in [0,1] with no discretisation. |
| 11 | `DiffusionMP/Diffusion.py` | No x0 clipping during sampling (only the final output is clipped). Standard practice for data in [-1,1] is to clip the predicted x0 at every step. |
| 12 | `DiffusionMP/Model.py:202` | `forward(self, x, t=th.randint(1000, (1,)))`: the default timestep is drawn once, at import. |

## Hurts training

| # | Where | Problem |
| --- | --- | --- |
| 13 | `DiffusionMP/Train.py:39` | `loss = mse(...).sum() / 1000`: the loss scales with the number of voxels (2 × 64³ / 1000 ≈ 524 × the mean). With `grad_clip = 1`, almost every step is clipped, so training becomes normalised-gradient descent at an unintended rate. |
| 14 | `DiffusionMP/Model.py` | BatchNorm inside a diffusion U-Net. With batch 2 and a different noise level per sample, batch statistics are noisy and differ between training and sampling. GroupNorm is standard. |
| 15 | `DiffusionMP/Model.py` | Initialisation code is commented out. No zero-initialised residual or output layers; no pre-activation norm in `ResBlock.block1`. |
| 16 | `DiffusionMP/Train.py` | No EMA of the weights. EMA is part of every standard DDPM recipe and matters for sample quality. |
| 17 | `DiffusionMP/Train.py:50, 52` + `MainMP.py:32` | `break` after one batch makes each "epoch" a single iteration, and `save_interval = 1` writes a checkpoint every iteration: 100,001 files of the full model. |
| 18 | `MainMP.py:8, 17` | Batch 2 and lr 1e-5 (warm-up to 2e-5, then cosine decay to 0). Both are low for a 64³ DDPM on a modern GPU. |
| 19 | `DiffusionMP/Train.py` | No optimiser state or RNG in the checkpoints, so training cannot resume after a time limit. |
| 20 | `image_datasets.py:61, 65` | One loader worker, every volume read from disk each time, no augmentation. |
| 21 | `MainMP.py:21, 24` | `img_size: 32` is never used (sampling assumes 64³), and `device: "cpu"`. |
| 22 | `3DMPDDPM/` | Eight diverging copies of the same model (`Diffusion`, `DiffusionMP`, `DiffusionMP_DDIM`, `DDIM`, `DiffusionST_*`, `DiffusionConditioned`, `DiffusionFreeGuidence`). It is unclear which produced any result. |

## What `ddpm3d/` does instead

| Item | Choice |
| --- | --- |
| Network | 3D U-Net, base 64 channels, multipliers (1, 2, 2, 4), 2 residual blocks per level, GroupNorm + SiLU pre-activation, time injected by adaptive group-norm scale/shift, zero-initialised residual and output layers, multi-head attention at 16³ and 8³ (`scaled_dot_product_attention`), dropout 0.1. 49.4 M parameters. |
| Diffusion | T = 1,000, linear β 1e-4 → 0.02 (cosine available), ε-prediction, L_simple loss (mean over all elements). |
| Sampling | DDPM: all 1,000 steps, posterior variance, x0 clipped each step. DDIM: η = 0, evenly spaced training timesteps from noisy to clean, same schedule as training, x0 clipped. A test checks that DDIM with all steps and η = 1 equals DDPM. |
| Data | All volumes loaded once onto the GPU as class indices. Values detected from the data (≤ 16 labels). Binary → one channel at ±1; more phases → one-hot. Random flips along x, y and z; axis permutations optional (off: some materials are anisotropic). |
| Optimisation | AdamW, lr 1e-4, 1,000-step linear warm-up then constant, batch 8, gradient clip 1, EMA 0.9999 with warm-up, bf16 autocast, TF32. 150,000 iterations by default. |
| Checkpoints | `last.pt` every 2,000 steps (model, EMA, optimiser, RNG). EMA snapshot every 10,000 steps, plus a DDIM-50 preview PNG. Resume is bit-exact (tested). SIGUSR1/SIGTERM → save and exit 99 for requeue. |
| Selection | Every EMA snapshot scored on 32 DDIM-50 samples against 256 training volumes: phase-fraction W1 + error of the mean two-point function along x, y and z. The same criterion is used for WGAN-GP. |
| Outputs | Per sampler: 150 volumes in the original label values, raw float16 output, and `timing.json` with network evaluations and generation-only wall time. |

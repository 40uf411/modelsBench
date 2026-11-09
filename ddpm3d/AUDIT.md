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

The defaults follow current diffusion practice, not the original paper's configuration.

| Item | Choice | Why |
| --- | --- | --- |
| Noise schedule | Cosine in logSNR, shifted by log(1/8) | A 64³ volume has as many elements as a 512² image, so averaging over voxels strips noise from coarse structure and the standard schedule barely trains the high-noise regime. The shift (Hoogeboom et al. 2023, "simple diffusion") gives the 8³ bottleneck the SNR a 64² image's 8² bottleneck would see. |
| Terminal SNR | Zero (alpha_bar_T = 0) | Training then sees the pure noise that sampling starts from (Lin et al. 2024). |
| Prediction target | v = sqrt(ab)·eps − sqrt(1−ab)·x0 | Defined at zero SNR (eps-prediction is not) and stable at high noise (Salimans & Ho 2022). Loss: plain MSE on v. |
| Self-conditioning | On (p = 0.5 during training; previous x0 estimate at sampling) | The Analog Bits recipe for discrete data diffused as ±1 values (Chen, Zhang & Hinton 2023), which is what segmented volumes are. Costs about 25% more training compute. |
| Network | 3D U-Net, base 64, multipliers (1, 2, 4, 4), 2 residual blocks per level (per-level counts allowed), GroupNorm + SiLU, adaptive scale/shift time conditioning, zero-initialised residual and output layers, dropout 0.1, attention at 16³ and 8³ with QK RMS-norm, `scaled_dot_product_attention`. 70.9 M parameters. | Extra width goes to 16³/8³, where coarse 3D structure forms and convolutions are cheap. The 64³ level, which dominates compute, keeps 64 channels. QK-norm prevents attention-logit blow-up in bf16. |
| Optimiser | AdamW, lr 1e-4, betas (0.9, 0.99), no weight decay, 1k-step warm-up then constant, gradient clip 1 | Standard for diffusion U-Nets; beta2 = 0.99 adapts faster to the shifting loss scale across noise levels. |
| Batch and length | 16 volumes × 100k iterations (1.6 M samples seen, with random flips) | Check `gpu_mem_gb` and `it_per_s` in the first minutes; adjust via `TRAIN_ARGS`. |
| Weight averaging | Two EMAs (0.999 and 0.9999), both snapshotted every 10k steps | Which decay works best depends on run length; the selection step picks it instead of guessing. |
| Speed | bf16 autocast, TF32, channels-last-3d, `torch.compile` (falls back to eager on error), whole dataset on the GPU | |
| Samplers | DDPM (1,000 steps, posterior variance), DDIM (η = 0, 50 and 100 steps), DPM-Solver++(2M) (20 steps; Lu et al. 2022). All use the training schedule, run noisy → clean, clip x0 each step, and pass the self-conditioning estimate. | DPM-Solver++ is the standard accelerated sampler for this model class; it answers the reviewer's request for accelerated DDPM sampling. |
| Checks | 17 tests, including: every sampler recovers a point mass exactly given the exact denoiser; DDIM with all steps and η = 1 equals DDPM; the schedule shift is exactly log(1/8); resume is bit-exact. | |
| Selection | Every EMA snapshot from step 30k on: 32 DDIM-50 samples vs 256 training volumes, phase-fraction W1 + mean two-point-function error along x, y, z. Same criterion as WGAN-GP. | |

Not adopted, deliberately: EDM-style continuous-time preconditioning and transformer backbones (DiT/U-ViT). The first would no longer be a DDPM in the benchmark's sense; the second is data-hungry and expensive at 64³ in pixel space with ~2,500 training volumes.

# 3D WGAN-GP: training on the cluster

The model, loss and optimiser are unchanged (`models_3d.py`, `training_3d.py`,
`dataloaders_3d.py`). Three files were added around them:

| File | Role |
| --- | --- |
| `train_3d_cli.py` | `main_3d.py` with arguments instead of hard-coded paths; same defaults (batch 6, Adam lr 1e-4, betas (0.5, 0.99), lambda 10, 5 critic steps, 1,500 epochs, checkpoint every 10 epochs). Resumes from the latest checkpoint; on SIGUSR1/SIGTERM it finishes the epoch, checkpoints and exits with code 99. |
| `select_and_sample_3d.py` | Scores every checkpoint against the training data and keeps the lowest score, then generates the final samples from it. |
| `microstats.py` | Phase fractions and two-point probability S2 used by the score. |

## Checkpoint selection

Replaces selection by visual inspection. For each checkpoint, 64 volumes are
generated (fixed seed, generator in eval mode), thresholded at 0 and compared
with 256 random training volumes:

    score = mean_k W1(phase fraction k) / std_ref(k)  +  mean_k,axis ||S2_gen - S2_ref|| / ||S2_ref||

with S2 the mean two-point probability along x, y and z for lags 1-32.
All scores are written to `selection/scores.csv`; the winner to `selection/best.json`.

## Outputs per dataset (`$OUT_ROOT/<dataset>/`)

    checkpoints/ckpt_epoch_XXXX.pt     generator + critic + optimisers
    train_log.jsonl                    one line per epoch
    selection/scores.csv, best.json
    samples/raw/samples_raw.npy        150 x 64^3, float16 tanh output
    samples/bin/sample_XXX.npy         uint8 {0,1}, same convention as the data
    samples/bin_median3/...            the same samples after a 3^3 median filter
    samples/timing.json                generation time only, GPU name

`bin/` and `bin_median3/` hold the same samples, so the effect of the filter
can be reported directly.

## Running

    bash slurm/setup_env.sh                          # once, on a login node
    # edit DATA_ROOT in slurm/train_wgan_gp_3d.slurm
    sbatch slurm/train_wgan_gp_3d.slurm              # all datasets in sequence
    sbatch slurm/train_wgan_gp_3d.slurm illite_bin   # selected datasets only

Checkpoints take roughly 0.3 GB each (generator, critic and both Adam states),
so 150 per dataset need about 50 GB: keep `OUT_ROOT` on `$GLOBALSCRATCH` (the default).
The data loader expects binary volumes stored as {0,1}; multiphase data are not supported
by this generator (one tanh channel).

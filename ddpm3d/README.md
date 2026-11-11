# ddpm3d: 3D DDPM with DDIM sampling

Replaces `3DMPDDPM/` for the 3D benchmark. `AUDIT.md` lists what was wrong
with the previous code and what changed.

## Run on the cluster

    bash slurm/setup_env.sh                       # once, on a login node
    # edit DATA_ROOT in slurm/train_ddpm_3d.slurm
    sbatch slurm/train_ddpm_3d.slurm              # all datasets in sequence
    sbatch slurm/train_ddpm_3d.slurm illite_bin   # selected datasets only

For each dataset the job trains, selects the best EMA snapshot, and then
generates 150 samples each with DDPM (1,000 steps), DDIM (50 and 100 steps,
`DDIM_STEPS`) and DPM-Solver++(2M) (20 steps, `DPMPP_STEPS`). Near the time limit, training saves its state and the job
requeues itself; finished stages are skipped on restart.

Per-dataset outputs (`$GLOBALSCRATCH/modelsbench/ddpm_3d/<dataset>/`):

    config.json                    all settings, data values, parameter count
    train_log.csv                  loss, lr, gradient norm, it/s, GPU memory
    snapshots/ema<decay>_<step>.pt EMA weights (0.999 and 0.9999) every 10,000 steps
    previews/step_XXXXXXX.png      central slices of 4 DDIM-50 samples
    selection/scores.csv, best.json
    best.pt                        selected snapshot
    samples/ddpm/labels/sample_XXX.npy
    samples/ddim50/..., samples/ddim100/..., samples/dpmpp20/...
    samples/*/raw/samples_raw.npy  output before discretisation (float16)
    samples/*/timing.json          network evaluations, generation time, GPU

Samples use the data's own label values (e.g. {0,1}), so the descriptor and
Yantra scripts can read them like the training volumes.

## Run by hand

    python -m ddpm3d.train  --data_dir DATA --out_dir RUN [--iters 150000 --batch_size 8 ...]
    python -m ddpm3d.select --run_dir RUN
    python -m ddpm3d.sample --ckpt RUN/best.pt --out_dir RUN/samples/ddpm --sampler ddpm
    python -m ddpm3d.sample --ckpt RUN/best.pt --out_dir RUN/samples/ddim50 --sampler ddim --steps 50

`python -m ddpm3d.train -h` lists every option. Tests: `python -m pytest -q ddpm3d`.

## Data

One segmented volume per `.npy` file, all the same shape (the default model
expects 64³). The label values are detected from the data (at most 16
distinct values). Two labels are encoded as one channel at ±1; more labels are
one-hot encoded (`--encoding` overrides this). Random flips along x, y and z
are on by default. Axis permutations (`--permute_axes 1`) are off, because some
of the materials are anisotropic.

## Defaults

v-prediction, shifted-cosine schedule (SNR × 1/8) with zero terminal SNR,
self-conditioning, U-Net 64 × (1, 2, 4, 4) with QK-normalised attention at
16³ and 8³ (70.9 M parameters), AdamW lr 1e-4, batch 16, 100k iterations,
EMAs 0.999 and 0.9999, bf16, `torch.compile`. `AUDIT.md` gives the reasoning
and references.

GPU time per dataset is not measured yet. Check `it_per_s` and `gpu_mem_gb`
in `train_log.csv` after the first minutes; change `--batch_size` or `--iters`
through `TRAIN_ARGS` if needed.

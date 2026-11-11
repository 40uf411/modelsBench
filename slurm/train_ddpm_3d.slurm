#!/bin/bash
#SBATCH --job-name=ddpm3d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ali.aouf@uclouvain.be
#
# For each dataset in turn:
#   1. train the 3D DDPM (ddpm3d.train)
#   2. select the best EMA snapshot (ddpm3d.select)
#   3. generate 150 samples with the full 1,000-step DDPM chain
#   4. generate 150 samples with DDIM for every value in DDIM_STEPS
#   5. generate 150 samples with DPM-Solver++(2M) for every value in DPMPP_STEPS
#      (set DPMPP_STEPS="" to skip)
#
# Submit from the repository root:
#   sbatch slurm/train_ddpm_3d.slurm                     # all datasets, one after the other
#   sbatch slurm/train_ddpm_3d.slurm illite_bin          # only the listed datasets
#
# To finish sooner, submit one job per dataset (14 H200 GPUs on the gpu partition):
#   for d in boom_clay_bin illite_bin bentheimer_bin banderabrown_bin savonnieres_bin; do
#       sbatch slurm/train_ddpm_3d.slurm "$d"; done
#
# 15 min before the time limit, training saves last.pt and the job requeues
# itself; the next run resumes. Each finished stage leaves a marker file, so
# requeued jobs skip what is done.
#
# One-time setup: bash slurm/setup_env.sh

set -uo pipefail

# ---- edit on the server ------------------------------------------------------
DATA_ROOT="/CHANGE/ME/datasets_3d"   # parent folder; one sub-folder of .npy volumes per dataset
# -------------------------------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-$GLOBALSCRATCH/modelsbench/ddpm_3d}"
VENV="${VENV:-$HOME/venvs/modelsbench}"
N_SAMPLES="${N_SAMPLES:-150}"
DDIM_STEPS="${DDIM_STEPS:-50 100}"
DPMPP_STEPS="${DPMPP_STEPS-20}"
TRAIN_ARGS="${TRAIN_ARGS:---iters 100000}"          # all other defaults: python -m ddpm3d.train -h
SELECT_MIN_STEP="${SELECT_MIN_STEP:-30000}"        # ignore snapshots before this step

DATASETS=(
  boom_clay_bin
  illite_bin
  bentheimer_bin
  banderabrown_bin
  savonnieres_bin
  # illite_3phases   # supported: three labels are detected and one-hot encoded
)
if [ "$#" -gt 0 ]; then DATASETS=("$@"); fi

module --force purge 2>/dev/null || module purge
module load releases/2024a
module load Python/3.12.3-GCCcore-13.3.0
# shellcheck disable=SC1091
source "$VENV/bin/activate"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# accept submission from the repo root or from slurm/
if [ ! -d "$REPO_DIR/slurm" ] && [ -d "$REPO_DIR/../slurm" ]; then REPO_DIR="$(cd "$REPO_DIR/.." && pwd)"; fi
cd "$REPO_DIR" || exit 1
mkdir -p "$OUT_ROOT"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- requeue handling ----------------------------------------------------------
REQUEUE=0
CHILD=0
CHILD_IS_TRAINING=0
on_usr1() {
  echo "$(date) USR1 received: time limit is close."
  REQUEUE=1
  if [ "$CHILD" -ne 0 ] && [ "$CHILD_IS_TRAINING" -eq 1 ]; then
    kill -USR1 "$CHILD" 2>/dev/null
  fi
}
trap on_usr1 USR1

run_child() {  # background + wait so the trap can fire; returns the command's status
  "$@" &
  CHILD=$!
  local status
  while true; do
    wait "$CHILD"; status=$?
    kill -0 "$CHILD" 2>/dev/null || break
  done
  CHILD=0
  return "$status"
}

requeue_and_exit() {
  echo "$(date) Requeueing job $SLURM_JOB_ID."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

stage() {  # stage <marker> <command...>: run once, mark on success
  local marker="$1"; shift
  if [ -f "$marker" ]; then echo "   (done: $(basename "$marker"))"; return 0; fi
  run_child "$@"
  local st=$?
  [ "$st" -eq 0 ] && touch "$marker"
  return "$st"
}

# ---- main loop -------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
  run_dir="$OUT_ROOT/$ds"
  data_dir="$DATA_ROOT/$ds"
  if [ ! -d "$data_dir" ]; then echo "== $ds: data folder $data_dir not found, skipping"; continue; fi
  mkdir -p "$run_dir"

  echo "== $ds: training ($(date))"
  CHILD_IS_TRAINING=1
  # shellcheck disable=SC2086
  run_child python -u -m ddpm3d.train --data_dir "$data_dir" --out_dir "$run_dir" $TRAIN_ARGS
  st=$?
  CHILD_IS_TRAINING=0
  if [ "$st" -eq 99 ] || [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi
  if [ "$st" -ne 0 ]; then echo "== $ds: training failed (exit $st)"; continue; fi

  echo "== $ds: snapshot selection ($(date))"
  stage "$run_dir/SELECT_DONE" python -u -m ddpm3d.select --run_dir "$run_dir" --min_step "$SELECT_MIN_STEP" \
    || { echo "== $ds: selection failed"; continue; }
  if [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi

  echo "== $ds: DDPM sampling, $N_SAMPLES samples ($(date))"
  stage "$run_dir/SAMPLES_DDPM_DONE" python -u -m ddpm3d.sample --ckpt "$run_dir/best.pt" \
      --out_dir "$run_dir/samples/ddpm" --sampler ddpm --n "$N_SAMPLES" \
    || echo "== $ds: DDPM sampling failed"
  if [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi

  for k in $DDIM_STEPS; do
    echo "== $ds: DDIM-$k sampling, $N_SAMPLES samples ($(date))"
    stage "$run_dir/SAMPLES_DDIM${k}_DONE" python -u -m ddpm3d.sample --ckpt "$run_dir/best.pt" \
        --out_dir "$run_dir/samples/ddim$k" --sampler ddim --steps "$k" --n "$N_SAMPLES" \
      || echo "== $ds: DDIM-$k sampling failed"
    if [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi
  done

  for k in $DPMPP_STEPS; do
    echo "== $ds: DPM-Solver++-$k sampling, $N_SAMPLES samples ($(date))"
    stage "$run_dir/SAMPLES_DPMPP${k}_DONE" python -u -m ddpm3d.sample --ckpt "$run_dir/best.pt" \
        --out_dir "$run_dir/samples/dpmpp$k" --sampler dpmpp --steps "$k" --n "$N_SAMPLES" \
      || echo "== $ds: DPM-Solver++-$k sampling failed"
    if [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi
  done
done

echo "All done ($(date))."

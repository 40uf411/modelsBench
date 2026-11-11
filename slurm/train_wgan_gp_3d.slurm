#!/bin/bash
#SBATCH --job-name=wgangp3d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaH200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ali.aouf@uclouvain.be
#
# Train 3D WGAN-GP on each dataset in turn, then pick the best checkpoint
# and generate 150 samples from it.
#
# Submit from the repository root:
#   sbatch slurm/train_wgan_gp_3d.slurm                    # all datasets, one after the other
#   sbatch slurm/train_wgan_gp_3d.slurm bentheimer_bin     # only the listed datasets
#
# To finish sooner, submit one job per dataset (there are 14 H200 GPUs on
# the gpu partition):
#   for d in boom_clay_bin illite_bin bentheimer_bin banderabrown_bin savonnieres_bin; do
#       sbatch slurm/train_wgan_gp_3d.slurm "$d"; done
#
# Time limit: 15 min before the 2-day limit, training finishes its epoch,
# writes a checkpoint and the job requeues itself; the next run resumes.
# Finished datasets are skipped (SAMPLES_DONE marker).
#
# One-time setup: bash slurm/setup_env.sh

set -uo pipefail

# ---- edit on the server ------------------------------------------------------
DATA_ROOT="/CHANGE/ME/datasets_3d"   # parent folder; one sub-folder of .npy volumes per dataset
# -------------------------------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-$GLOBALSCRATCH/modelsbench/wgan_gp_3d}"
VENV="${VENV:-$HOME/venvs/modelsbench}"
N_SAMPLES="${N_SAMPLES:-150}"

DATASETS=(
  boom_clay_bin
  illite_bin
  bentheimer_bin
  banderabrown_bin
  savonnieres_bin
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
cd "$REPO_DIR/wgan-gp" || exit 1
mkdir -p "$OUT_ROOT"

echo "Job $SLURM_JOB_ID on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
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

run_child() {  # run a command in the background so the trap can fire; return its status
  "$@" &
  CHILD=$!
  local status
  while true; do
    wait "$CHILD"; status=$?
    kill -0 "$CHILD" 2>/dev/null || break   # still running: wait was interrupted by the trap
  done
  CHILD=0
  return "$status"
}

requeue_and_exit() {
  echo "$(date) Requeueing job $SLURM_JOB_ID."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

# ---- main loop -------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
  run_dir="$OUT_ROOT/$ds"
  data_dir="$DATA_ROOT/$ds"
  if [ -f "$run_dir/SAMPLES_DONE" ]; then
    echo "== $ds: already done, skipping"; continue
  fi
  if [ ! -d "$data_dir" ]; then
    echo "== $ds: data folder $data_dir not found, skipping"; continue
  fi
  mkdir -p "$run_dir"
  echo "== $ds: training ($(date))"

  CHILD_IS_TRAINING=1
  run_child python -u train_3d_cli.py --data_root "$data_dir" --out_dir "$run_dir" \
      --num_workers "${SLURM_CPUS_PER_TASK:-8}"
  st=$?
  CHILD_IS_TRAINING=0
  if [ "$st" -eq 99 ] || [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi
  if [ "$st" -ne 0 ]; then echo "== $ds: training failed (exit $st)"; continue; fi

  echo "== $ds: checkpoint selection and sampling ($(date))"
  run_child python -u select_and_sample_3d.py --run_dir "$run_dir" --data_root "$data_dir" \
      --n_samples "$N_SAMPLES" --median 3
  st=$?
  if [ "$st" -eq 0 ]; then
    touch "$run_dir/SAMPLES_DONE"
  else
    echo "== $ds: selection/sampling failed (exit $st)"
  fi
  if [ "$REQUEUE" -eq 1 ]; then requeue_and_exit; fi
done

echo "All done ($(date))."

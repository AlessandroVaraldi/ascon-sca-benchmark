#!/bin/bash
#SBATCH --job-name=ascon_sca
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=ascon_subkeys_%j.out

# Load environment
source ~/.bashrc
conda activate torch-gpu

# Directory where you launched sbatch (must contain run.py, models/, loss_functions/, etc.)
PROJECT_DIR="$SLURM_SUBMIT_DIR"

# Work directory on SCRATCH (fast local storage on the node)
WORKDIR="$SCRATCH/run_$SLURM_JOB_ID"
mkdir -p "$WORKDIR"

echo "SLURM_JOB_ID      = $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR  = $SLURM_SUBMIT_DIR"
echo "SCRATCH           = $SCRATCH"
echo "WORKDIR           = $WORKDIR"
echo "HOSTNAME          = $(hostname)"

# Copy the whole project (scripts, models, loss_functions, etc.) to SCRATCH
rsync -av "$PROJECT_DIR"/ "$WORKDIR"/

cd "$WORKDIR"

# Where to store results inside WORKDIR
OUTPUT_DIR="$WORKDIR/run_outputs"
mkdir -p "$OUTPUT_DIR"

# Run the full 3-bit-subkey recovery pipeline
# >>> Adjust --h5-path, --epochs, --window-size, etc. as needed <<<
python3 run.py \
  --h5-path "$HOME/data/ascon_hw_unprotected.h5" \
  --output-dir "$OUTPUT_DIR" \
  --loss-type ranking \
  --models mlp cnn tcn transformer \
  --epochs 50 \
  --batch-size 256 \
  --window-size 500

# Results directory in the original project folder
RESULTS_DIR="$PROJECT_DIR/run_results/job_$SLURM_JOB_ID"
mkdir -p "$RESULTS_DIR"

# Sync all logs, plots, summaries, etc. back to the project directory
rsync -av "$WORKDIR"/ "$RESULTS_DIR"/

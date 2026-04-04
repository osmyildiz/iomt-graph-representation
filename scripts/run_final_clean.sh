#!/bin/bash
#SBATCH --job-name=iomt_final
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

echo "=== Final Clean Experiment ==="
echo "Job ID: $SLURM_JOB_ID | Start: $(date)"

source /network/rit/dgx/dgx_subasi_lab/osman/ADNI_CausalGNN/venv/bin/activate

python -m src.evaluation \
  --experiment final_clean \
  --graph_dir /network/rit/dgx/dgx_subasi_lab/osman/osman-net/data/natural_graphs \
  --results_dir results/

echo "=== DONE: $(date) ==="

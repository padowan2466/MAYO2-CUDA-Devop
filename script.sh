#!/bin/bash
#SBATCH --job-name=mayo_h100
#SBATCH --time=00:05:00
#SBATCH --account=def-achar
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h100:1
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge
module load StdEnv/2023
module load cuda

cd "$SLURM_SUBMIT_DIR"

echo "Node:"
hostname

echo "GPU:"
nvidia-smi

echo "NVCC:"
which nvcc
nvcc --version

echo "Compiling:"
make clean
make

echo "Running:"
srun ./mayo-2-gpu-fine

#!/bin/bash
# The interpreter used to execute the script
#\#SBATCH" directives that convey submission options:
#SBATCH --job-name=eecs442
#SBATCH --mail-user=pohsun@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=01:00:00
#SBATCH --account=engin1
#SBATCH --partition=gpu,gpu_mig40,spgpu
#SBATCH --gpus=1
#SBATCH --output=./tmp.log

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time is: $(date)"
echo "Directory is: $(pwd)"


# Activate your environment
source ~/.bashrc
conda activate EECS442

python src/main.py > tmp.txt

echo "Job finished with exit code $? at: $(date)"
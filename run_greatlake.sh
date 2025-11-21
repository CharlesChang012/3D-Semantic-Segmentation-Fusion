#!/bin/bash
# The interpreter used to execute the script
#\#SBATCH" directives that convey submission options:
#SBATCH --job-name=eecs442
#SBATCH --mail-user=pohsun@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=./log/train.log

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time is: $(date)"
echo "Directory is: $(pwd)"


# Activate your environment
source ~/.bashrc
conda activate 3DSSF

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scheduleParams.txt)

echo "Running params: $PARAMS"

python main_train.py $PARAMS > ./log/train.txt

echo "Job finished with exit code $? at: $(date)"
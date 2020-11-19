#!/bin/bash -l

#SBATCH -J sac-her-0
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH -A cembra1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu
#SBATCH --output="data/sac_her_0.out"
#SBATCH --error="data/sac_her_0.err"

cd $SLURM_SUBMIT_DIR

module load plgrid/tools/python/3.7

cd ..
source .venv/bin/activate
python train_sac_her_0.py

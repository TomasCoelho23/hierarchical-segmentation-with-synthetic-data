#!/bin/bash


#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


module load cuda/12.4
conda activate #insert env
python ../ThirdPlace/phenobench_challenge-main/mask2former_at_phenobench/inference_all.py
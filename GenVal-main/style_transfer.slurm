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
source activate #insert env
python ../GenVal-main/ControlNet/style_transfer.py --num_samples=500 --domain=fog
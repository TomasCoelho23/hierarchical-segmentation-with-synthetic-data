#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2



module load cuda/12.4
source activate #insert env
python ../GenVal-main/ControlNet/train.py 
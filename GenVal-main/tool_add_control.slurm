#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=24
#SBATCH --partition=short
#SBATCH --begin=now
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR




module load cuda/12.4
conda activate #insert env
python ../GenVal-main/ControlNet/tool_add_control.py ../ControlNet/models/control_seg.ckpt
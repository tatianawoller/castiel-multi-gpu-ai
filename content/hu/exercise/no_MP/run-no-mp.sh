#!/bin/bash
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=output_no_LWMP.out

module load profile/deeplrn
module load cineca-ai/4.1.1

torchrun no_LWMP.py --epochs=25


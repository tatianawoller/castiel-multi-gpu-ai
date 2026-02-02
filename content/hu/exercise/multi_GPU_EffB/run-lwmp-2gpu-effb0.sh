#!/bin/bash
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:2
#SBATCH --output=output_LWMP_2_GPU_EffB0.out

module load profile/deeplrn
module load cineca-ai/4.1.1

torchrun LWMP_2GPU_EffB0.py --epochs=25


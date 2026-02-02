#!/bin/bash
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:4
#SBATCH --output=output_lwmp_4_gpu.out
#SBATCH --exclusive

module load profile/deeplrn
module load cineca-ai/4.1.1

torchrun LWMP_4_GPU.py --epochs=25

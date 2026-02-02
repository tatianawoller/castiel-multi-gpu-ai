#!/bin/bash
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:4
#SBATCH --output=output_large_model.out
#SBATCH --exclusive

module load profile/deeplrn
module load cineca-ai/4.1.1

nvidia-smi

torchrun roberta_split.py

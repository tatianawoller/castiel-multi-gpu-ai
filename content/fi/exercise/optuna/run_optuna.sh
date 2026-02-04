#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=tra26_castiel2

module purge
module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source $WORK/otaubert/castielvenv/bin/activate

srun python -u optuna_mnist.py

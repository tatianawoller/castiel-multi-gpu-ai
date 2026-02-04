#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=tra26_castiel2
#SBATCH --array=0-7

module purge
module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

Bs=(16 32)
Ds=(64 128)
Ls=(0.001 0.0001)

B=${Bs[${SLURM_ARRAY_TASK_ID}%2]}
D=${Ds[${SLURM_ARRAY_TASK_ID}/2%2]}
L=${Ls[${SLURM_ARRAY_TASK_ID}/4%2]}

set -exuo

srun python train_mnist.py -b $B -d $D -l $L

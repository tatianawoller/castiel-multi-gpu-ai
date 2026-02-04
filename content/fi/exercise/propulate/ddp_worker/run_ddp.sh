#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:20:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=tra26_castiel2

module purge
module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source $WORK/otaubert/castielvenv/bin/activate
export MLFLOW_TRACKING_URI=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/PDLd3/mlruns

set -xv

srun --label bash -c " \
    RANK=\$SLURM_PROCID \
    LOCAL_RANK=\$SLURM_LOCALID \
    python ddp.py"

#!/bin/bash
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:4
#SBATCH --output=output_DDP_LWMP_EffB0.out
#SBATCH --exclusive

export RDZV_HOST=$(hostname)
export RDZV_PORT=30000          

module load profile/deeplrn
module load cineca-ai/4.1.1

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    DDP_LWMP_2x4GPU_EffB0.py --epochs=25

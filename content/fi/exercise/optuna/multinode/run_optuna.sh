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


mkdir -p logs

echo "starting sql server"
# Determine hostname
LIBSQL_HOST=$(hostname -I | cut -f 1 -d ' ')

# Generate port, user and password
LIBSQL_PORT=$(( $SLURM_JOB_ID + 10000))
LIBSQL_USER=$( openssl rand -hex 10 )
LIBSQL_PASSWORD=$( openssl rand -hex 10 )

# Create hashed token that will be used for authentication
export LIBSQL_TOKEN=$(echo ${LIBSQL_USER}:${LIBSQL_PASSWORD} | base64)

# Set up libSQL-server's environment
export SQLD_HTTP_LISTEN_ADDR=${LIBSQL_HOST}:${LIBSQL_PORT}
export SQLD_HTTP_AUTH=basic:$LIBSQL_TOKEN

# Set Optuna connection string
export OPTUNA_STORAGE="sqlite+libsql://${SQLD_HTTP_LISTEN_ADDR}"

# Start up the server
srun -n 1 $WORK/otaubert/libsql-server-x86_64-unknown-linux-gnu/sqld > logs/libsql_${SLURM_JOB_ID}.out 2>&1 &
export LIBSQL_PID=$?

Wait for database to fully initialize
sleep 5

srun python -u optuna_mnist.py

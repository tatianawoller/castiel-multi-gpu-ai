#!/bin/bash
        
#SBATCH --job-name=jupyter_environment
#SBATCH --time=01:00:00 ### Change compute time based on your needs ###
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation= ### If you have a reservation, write it here ###
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error jupyter-%j.err
#SBATCH --output jupyter-%j.out

# Load the cineca-ai module (module load cineca-ai) 
# Alternatively, load the python module you used to create your venv 
module load python/3.11.6--gcc--8.5.0

# Activate your venv
source .../bin/activate ### Write here the path to your venv ###


# Get the worker list associated to this slurm job
worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Set the first worker as the head node and get his ip
head_node=${worker_list[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Open a SSH tunnel to the login node and then to the compute node from another shell on your local machine, copying and pasting the following instructions printed in the .out file.
# Print ssh tunnel instruction
jupyter_port=$(($RANDOM%(64511-50000+1)+50000))
jupyter_token=${USER}_${jupyter_port}
echo ===================================================
echo [INFO]: To access the Jupyter server, remember to open a ssh tunnel from your local machine with: 
echo ssh -L $jupyter_port:$head_node_ip:$jupyter_port ${USER}@login02-ext.leonardo.cineca.it -N
echo then you can connect to the jupyter server at http://127.0.0.1:$jupyter_port/lab?token=$jupyter_token
echo ===================================================

# Start the head node
echo [INFO]: Starting jupyter notebook server on $head_node 

# Note that the jupyter notebook command is available only because we have enabled the venv
command="jupyter lab --ip=0.0.0.0 --port=${jupyter_port} --NotebookApp.token=${jupyter_token}"
echo [INFO]: $command
$command &

echo [INFO]: Your env is up and running.

sleep infinity



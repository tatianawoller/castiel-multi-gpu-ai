#!/bin/bash
        
#SBATCH --job-name=train_ray_jupyter_environ
#SBATCH --time=04:00:00
#SBATCH --account=tra26_castiel2
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_cast7
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16 # Use a number of cpus appropriate to the partition you are in
#SBATCH --gres=gpu:2
#SBATCH --mem=100000MB
#SBATCH --error ray_jupyter-%j.err
#SBATCH --output ray_jupyter-%j.out

# Enable cuda and load the python module and enable the venv.
module load python/3.11.6--gcc--8.5.0
module load cuda/12.1

source /leonardo_scratch/fast/tra26_castiel2/environments/ray_venv/bin/activate

# Get the worker list associated to this slurm job
worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Set the first worker as the head node and get his ip
head_node=${worker_list[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

ray_gcs_port=$(($RANDOM%(35000-20000+1)+20000))

# Define the ray address variable for all the child processes
export RAY_ADDRESS=$head_node_ip:$ray_gcs_port

# Start the head node
echo [INFO]: Starting head node $head_node at $head_node_ip

# Note that ntasks in srun is set to 1 because otherwise the ray start command would be executed ntasks times, spawning ntasks*--num-cpus workers.
srun --nodes=1 --ntasks=1 -w "$head_node" ray start --head --dashboard-host="0.0.0.0" --node-ip-address="$head_node_ip" --port="$ray_gcs_port" --num-cpus ${SLURM_CPUS_PER_TASK} --block &

# Print ssh tunnel instruction
ray_dashboard_port=$(($RANDOM%(50000-35000+1)+35000))
echo ===================================================
echo [INFO]: To access the RAY web ui, remember to open a ssh tunnel with: 
echo ssh -L $ray_dashboard_port:$head_node_ip:8265 ${USER}@login02-ext.leonardo.cineca.it -N
echo then you can connect to the dashboard at http://127.0.0.1:$ray_dashboard_port
echo ===================================================
echo [INFO]: everything is running.

# Print ssh tunnel instruction
jupyter_port=$(($RANDOM%(64511-50000+1)+50000))
jupyter_token=${USER}_${jupyter_port}
echo ===================================================
echo [INFO]: To access the Jupyter server, remember to open a ssh tunnel with: 
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

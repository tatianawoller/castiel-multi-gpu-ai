#!/bin/bash
        
#SBATCH --job-name=jupyter_environ
#SBATCH --time=01:00:00
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
##SBATCH --reservation=s_tra_bbs5
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8 # Use a number of cpus appropriate to the partition you are in
#SBATCH --gres=gpu:4 # Use a number of gpus appropriate to the partition you are in
#SBATCH --error jupyter-%j.err
#SBATCH --output jupyter-%j.out

# Load the python module and enable the venv
source /leonardo_scratch/fast/tra26_castiel2/environments/rag_venv/bin/activate 

# Get the worker list associated to this slurm job
worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Set the first worker as the head node and get his ip
worker_node=${worker_list[0]}
worker_node_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address)

# Export environment variables
# Models were downloaded to /leonardo_scratch/fast/tra26_castiel2/models/hub
# by the 0_configure_env.sh script. If you changed the location, update the following two variables.
export HF_HOME="/leonardo_scratch/fast/tra26_castiel2/models/hub"
export HF_HUB_CACHE="/leonardo_scratch/fast/tra26_castiel2/models/hub"
export HF_HUB_OFFLINE="1"
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING="1"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"

# Print ssh tunnel instruction
jupyter_port=$((1025 + $RANDOM % 64511))
jupyter_token=${USER}_${jupyter_port}
echo ===================================================
echo [INFO]: To access the Jupyter server, remember to open a ssh tunnel with: 
echo ssh -L $jupyter_port:$worker_node_ip:$jupyter_port ${USER}@login02-ext.leonardo.cineca.it -N
echo then you can connect to the jupyter server at http://127.0.0.1:$jupyter_port/lab?token=$jupyter_token
echo ===================================================

# Start the head node
echo [INFO]: Starting jupyter notebook server on $worker_node 

# Note that the jupyter notebook command is available only because we have enabled the venv
command="jupyter lab --ip=0.0.0.0 --port=${jupyter_port} --NotebookApp.token=${jupyter_token}"
echo [INFO]: $command
$command &

# Start vllm endpoint
vllm_command="vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 --pipeline-parallel-size $SLURM_JOB_NUM_NODES --api-key password --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --guided_decoding_backend auto --enable-auto-tool-choice --disable-log-requests"
echo "[INFO]: $vllm_command"
$vllm_command &

echo [INFO]: Your env is up and running.

sleep infinity

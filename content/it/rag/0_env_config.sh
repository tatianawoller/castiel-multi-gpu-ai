# Config paths
HF_MODEL_DIR="/leonardo_scratch/fast/tra26_castiel2/models"
HF_MODEL_CACHE="/leonardo_scratch/fast/tra26_castiel2/models/hub"

# Create python3 env
module load python/3.11.6--gcc--8.5.0
python3 -m venv venv
module purge

# Install the required libraries
source venv/bin/activate
pip3 install -r requirements.txt

# Download HF models
#mkdir -p $HF_MODEL_DIR
export HF_HOME=$HF_MODEL_DIR
export HF_HUB_CACHE=$HF_MODEL_CACHE

# Mistral3.1
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503
huggingface-cli download BAAI/bge-m3

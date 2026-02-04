# Hyperparameter Seach Exercise

## Random Search with Array Jobs

There is a script training a simple neural network on MNIST (`train_mnist.py`) and two slurm job scripts given (`run_mnist.sh` and `run_mnist_grid.sh`).
Some hyperparameters can be passed to the script.
If you pass a `random_seed` it will generate those hyperparameters randomly from that seed, ignoring the given parameter values.

Your task is to be inspired by the `run_mnist_grid.sh` script and write a `run_mnist_random.sh` job script, that performs a random search of the parameter space.
Your job script should run 10 trainings.
The search space is already defined as the `limits` variable in the training script.

## Bayesian Search with Optuna

There is a python script using Optuna for a more sophisticated hyperparameter search.
It is using two GPUs, but since it communicates via a journal storage, it can not run on more than a single node.
Some lines are missing.

You task is to use the trial to generate new candidate parameters.

There is a multi node [example](https://github.com/AaltoRSE/hpo-on-hpc/tree/main/on-triton) from Aalto University.
The official documentation is less HPC focused.

## Parallel Evolutionary Search with Propulate
There is a python script using Propulate using multiple GPUs.
The code for setting the devices for each worker correctly is missing.

There are more examples at the [project github](https://github.com/Helmholtz-AI-Energy/propulate/tree/main/tutorials)


## Notes on Environment Setup

The environment on Leonardo was created this way:
```bash
module load profile/deeplrn
module load cineca-ai/4.3.0

python -m venv castielvenv
source castielvenv/bin/activate
pip install optuna
git clone https://github.com/Helmholtz-AI-Energy/propulate.git
cd propulate
pip install -e .
```
propulate is also pip installable.

To use multu node with optuna, it would also need 
``` bash
pip install sqlalchemy-libsql
```

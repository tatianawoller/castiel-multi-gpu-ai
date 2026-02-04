import datetime as dt
import os
import random
import socket
import time

import torch
import torch.distributed as dist
from mpi4py import MPI
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import MNIST
import mlflow

from propulate import Islands
from propulate.utils import get_default_propagator

from ddp_utils import get_dataloaders

GPUS_PER_NODE = int(os.environ["SLURM_GPUS_PER_NODE"])
NUM_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
SUBGROUP_COMM_METHOD = "nccl-slurm"
checkpoint_path = "./ddp_pcheckpoints"
dataset_path = f"{os.environ['FAST']}/data/"
num_generations = 100
num_islands = 2
migration_prob = 0.1

seed = 42
pollination = True
limits = {
    "num_layers": (2, 10),
    "activation": ("relu", "sigmoid", "tanh"),
    "lr": (0.01, 0.0001),
    "d_hidden": (2, 128),
    "gamma": (0.5, 0.999),
    "batch_size": ("1", "2", "4", "8", "16", "32", "64", "128"),
}

num_epochs = 20
# NOTE map categorical variable to python objects
activations = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

class CNN(nn.Module):
    def __init__(self, num_layers, activation, d_hidden):
        super().__init__()

        layers = []  # Set up the model architecture (depending on number of convolutional layers specified).
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=d_hidden, kernel_size=3, padding=1),
                activation(),
            ),
        ]
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=3, padding=1),
                activation(),
            )
            for _ in range(num_layers - 1)
        ]

        # NOTE due to padding output of final conv layer is 28*28*d_hidden
        self.fc = nn.Linear(in_features=28*28*d_hidden, out_features=10)
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def torch_process_group_init(subgroup_comm, method) -> None:
    """
    Create the torch process group of each multi-rank worker from a subgroup of the MPI world.

    Parameters
    ----------
    subgroup_comm : MPI.Comm
        The split communicator for the multi-rank worker's subgroup. This is provided to the individual's loss function
        by the ``Islands`` class if there are multiple ranks per worker.
    method : str
        The method to use to initialize the process group.
        Options: [``nccl-slurm``, ``nccl-openmpi``, ``gloo``]
        If CUDA is not available, ``gloo`` is automatically chosen for the method.
    """
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    comm_rank, comm_size = subgroup_comm.rank, subgroup_comm.size

    # Get master address and port
    # Don't want different groups to use the same port.
    subgroup_id = MPI.COMM_WORLD.rank // comm_size
    port = 29500 + subgroup_id

    if comm_size == 1:
        raise RuntimeError("Worker comm not set!")

    master_address = socket.gethostname()
    # Each multi-rank worker rank needs to get the hostname of rank 0 of its subgroup.
    master_address = subgroup_comm.bcast(str(master_address), root=0)

    # Save environment variables.
    os.environ["MASTER_ADDR"] = master_address
    # Use the default PyTorch port.
    os.environ["MASTER_PORT"] = str(port)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available!")
    else:
        num_cuda_devices = torch.cuda.device_count()
        device_number = MPI.COMM_WORLD.rank % num_cuda_devices
        torch.cuda.set_device(device_number)

    time.sleep(0.001 * comm_rank)  # Avoid DDOS'ing rank 0.
    if method == "nccl-openmpi":  # Use NCCL with OpenMPI.
        dist.init_process_group(
            backend="nccl",
            rank=comm_rank,
            world_size=comm_size,
        )

    elif method == "nccl-slurm":  # Use NCCL with a TCP store.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    else:
        raise NotImplementedError(f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm]!")

    # Call a barrier here in order for sharp to use the default comm.
    if dist.is_initialized():
        print("testing dist worker setup")
        dist.barrier()
        device = MPI.COMM_WORLD.rank % GPUS_PER_NODE

        disttest = torch.ones(1, device=device)

        dist.all_reduce(disttest)
        assert disttest[0] == comm_size, "Failed test of dist!"
        print("test complete")
    else:
        disttest = None
        raise RuntimeError("Dist not initialized!")


def loss_fn(params, subgroup_comm):
    torch_process_group_init(subgroup_comm, method=SUBGROUP_COMM_METHOD)
    # Extract hyperparameter combination to test from input dictionary.
    num_layers = int(params["num_layers"])  # Number of convolutional layers
    activation = str(params["activation"])  # Activation function
    lr = float(params["lr"])  # Learning rate
    gamma = float(params["gamma"])  # Learning rate reduction factor
    batch_size = int(params["batch_size"])
    d_hidden = int(params["d_hidden"])

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }  # Define activation function mapping.
    activation = activations[activation]  # Get activation function.
    loss_fn = torch.nn.NLLLoss()

    # Set up neural network with specified hyperparameters.
    model = CNN(num_layers, activation, d_hidden)

    train_dl, val_dl = get_dataloaders(
        batch_size=batch_size, subgroup_comm=subgroup_comm, num_workers=NUM_WORKERS, root=dataset_path
    )  # Get training and validation data loaders.

    device = MPI.COMM_WORLD.rank % GPUS_PER_NODE
    model = model.to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model)  # Wrap model with DDP.

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    log_interval = 10000
    best_val_loss: float = 1000000.0
    early_stopping_count, early_stopping_limit = 0, 5
    set_new_best = False
    model.train()

    # Initialize history lists.
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []

    for epoch in range(num_epochs):  # Loop over epochs.
        train_dl.sampler.set_epoch(epoch)  # Set current epoch in samplers.
        val_dl.sampler.set_epoch(epoch)
        # ------------ Train loop ------------
        for batch_idx, (data, target) in enumerate(train_dl):  # Loop over training batches.
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            torch.distributed.all_reduce(loss)  # Allreduce rank-local mini-batch train losses.
            loss /= dist.get_world_size()  # Average all-reduced rank-local mini-batch train losses over all ranks.
            train_loss_history.append(loss.item())  # Append globally averaged train loss of this epoch to history list.

            if batch_idx % log_interval == 0 or batch_idx == len(train_dl) - 1:
                print(
                    f"Train Epoch: {epoch} [{batch_idx}/{len(train_dl)} "
                    f"({100.0 * batch_idx / len(train_dl):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        # ------------ Validation loop ------------
        model.eval()
        val_loss: float = 0.0  # Initialize rank-local validation loss.
        correct: int = 0  # Initialize number of correctly predicted samples in rank-local validation set.
        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target).item()  # Sum up batch loss.
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= float(len(val_dl.dataset))  # Average rank-local validation loss over number of samples in validation set.
        num_val_samples = len(val_dl.dataset)  # Get overall number of samples in local validation set.
        # Convert to tensors for all-reduce communication.
        val_loss_tensor = torch.tensor([val_loss], device=device)
        correct_tensor = torch.tensor([correct], device=device)
        num_val_samples_tensor = torch.tensor([num_val_samples], device=device)
        # Allreduce rank-local validation losses, numbers of correctly predicted samples, and numbers of overall samples
        # in validation dataset over all ranks.
        torch.distributed.all_reduce(val_loss_tensor)
        torch.distributed.all_reduce(correct_tensor)
        torch.distributed.all_reduce(num_val_samples_tensor)
        val_loss_tensor /= dist.get_world_size()  # Average all-reduced rank-local validation losses over all ranks.
        val_loss_history.append(val_loss_tensor.item())  # Save globally averaged validation loss of this epoch.
        if val_loss_tensor.item() < best_val_loss:
            best_val_loss = val_loss_tensor.item()
            set_new_best = True
        # Calculate global validation accuracy and save in history list.
        val_acc = correct_tensor.item() / num_val_samples_tensor.item()
        val_acc_history.append(val_acc)

        print(f"\nValidation set: Average loss: {val_loss_tensor.item():.4f}, Accuracy: {100.0 * val_acc:.0f} %)\n")

        if not set_new_best:
            early_stopping_count += 1
        if early_stopping_count >= early_stopping_limit:
            print("Hit early stopping count, breaking.")
            break

        scheduler.step()
        set_new_best = False

    # Return best validation loss as an individual's loss (trained so lower is better).
    dist.destroy_process_group()

    if subgroup_comm.rank == 0:
        mlflow.set_experiment("ddp_experiment")
        with mlflow.start_run(run_name=f"run_{params.island}_{params.rank}_{params.generation}"):
            mlflow.log_params(params)
            mlflow.log_param("generation", params.generation)
            mlflow.log_metric("accuracy", best_val_loss)

    return best_val_loss


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        MNIST(download=True, root=".", transform=None, train=True)
        MNIST(download=True, root=".", transform=None, train=False)
    comm.Barrier()
    pop_size = 2 * comm.size
    rng = random.Random(seed + comm.rank)

    propagator = get_default_propagator(
        pop_size=pop_size,
        limits=limits,
        rng=rng,
    )

    # Set up island model.
    islands = Islands(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        generations=num_generations,
        num_islands=num_islands,
        migration_probability=migration_prob,
        pollination=pollination,
        checkpoint_path=checkpoint_path,
        # ----- SPECIFIC FOR MULTI-RANK UCS -----
        ranks_per_worker=2,  # Number of ranks per (multi rank) worker
    )

    islands.propulate()

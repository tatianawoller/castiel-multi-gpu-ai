from typing import Tuple

from mpi4py import MPI
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.utils.data.distributed as datadist

def get_dataloaders(batch_size: int, subgroup_comm: MPI.Comm, num_workers: int, root) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST train and validation dataloaders.

    Parameters
    ----------
    batch_size : int
        The batch size.
    subgroup_comm: MPI.Comm
        The MPI communicator object for the local class

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    torch.utils.data.DataLoader
        The validation dataloader.
    """
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(download=False, root=root, transform=data_transform, train=True)
    val_dataset = MNIST(download=False, root=root, transform=data_transform, train=False)
    if subgroup_comm.size > 1:  # Make the samplers use the torch world to distribute data
        train_sampler = datadist.DistributedSampler(train_dataset)
        val_sampler = datadist.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        dataset=train_dataset,  # Use MNIST training dataset.
        batch_size=batch_size,  # Batch size
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=(train_sampler is None),  # Shuffle data only if no sampler is provided.
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        batch_size=1,  # Batch size
        shuffle=False,  # Do not shuffle data.
        sampler=val_sampler,
    )
    return train_loader, val_loader

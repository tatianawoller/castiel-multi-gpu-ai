import os
import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config


data_root = f"{os.environ['FAST']}/data/"


class Model(nn.Module):
    def __init__(self, hidden_dim, activation):
        super().__init__()
        if activation == "relu":
            af = nn.ReLU
        elif activation == "sigmoid":
            af = nn.Sigmoid
        else:
            raise ValueError()

        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim),
            af(),
            nn.Linear(hidden_dim, hidden_dim),
            af(),
            nn.Linear(hidden_dim, 10),
        )
    def forward(self, x):
        return self.f(x)


def train(dl, model, loss_fn, optimizer, device=torch.device("cuda:0")):
    model = model.to(device)
    model.train()
    for step, batch in (enumerate(dl)):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()


def test(dl, model, loss_fn, device=torch.device("cuda:0")):
    model = model.to(device)
    model.eval()
    loss, correct = 0.0, 0.0
    with torch.no_grad():
        for step, batch in enumerate(dl):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()
            pred = model(x)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= len(dl)
    correct /= (len(dl.dataset))
    
    return correct


def ind_loss(params):

    learning_rate = params["learning_rate"]
    activation = params["activation_function"]
    hidden_dim = params["hidden_dim"]
    batch_size = params["batch_size"]

    epochs = 5
    best_acc = 0.

    train_ds = datasets.MNIST(
        root=data_root,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_ds = datasets.MNIST(
        root=data_root,
        train=False,
        download=False,
        transform=ToTensor(),
    )


    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = Model(hidden_dim, activation)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train(train_dl, model, loss_fn, optimizer)
        acc = test(train_dl, model, loss_fn)
        if acc > best_acc:
            best_acc = acc
    # NOTE minimization
    return -acc


if __name__ == "__main__":
    num_generations = 10
    pop_size = 4
    comm = MPI.COMM_WORLD
    rng = random.Random(comm.rank)
    limits = {"learning_rate": (1e-6, 1e-2),
              "activation_function": ("relu", "sigmoid", "tanh"),
              "hidden_dim": (8, 512),
              "batch_size": [16, 32, 64, 128, 256]}

    set_logger_config()
    # TODO set up a propagator, a Propulator, and then let it propulate the ind_loss

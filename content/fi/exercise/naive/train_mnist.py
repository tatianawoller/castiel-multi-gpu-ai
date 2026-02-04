import os
import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm


data_root = f"{os.environ['FAST']}/data/"

limits = {"learning_rate": (1e-6, 1e-2),
          "activation_function": ["relu", "sigmoid"],
          "hidden_dim": (8, 512),
          "batch_size": [16, 32, 64, 128, 256]}


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
    for step, batch in (enumerate(pbar:=tqdm(dl))):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        pbar.set_description(f"Loss: {(loss.item()):>7f}")
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
    print(f"Test accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")

def main(learning_rate, batch_size, hidden_dim, activation, random_seed):
    epochs = 5
    if random_seed is not None:
        random.seed(random_seed)
        learning_rate = random.uniform(*limits["learning_rate"])
        activation_function = random.choice(limits["activation_function"])
        hidden_dimm = random.randint(*limits["hidden_dim"])
        batch_size = random.choice(limits["batch_size"])

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
        print(f"Epoch {epoch}:")
        train(train_dl, model, loss_fn, optimizer)
        test(train_dl, model, loss_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d', '--hidden_dim', type=int, default=50)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-a', '--activation', type=str, default='relu',  choices=['relu', 'sigmoid'])
    parser.add_argument('-r', '--random_seed', type=int, default=None)

    args = parser.parse_args()

    main(args.learning_rate, args.batch_size, args.hidden_dim, args.activation, args.random_seed)

import os
import random
from multiprocessing import Pool

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import optuna

from tqdm import tqdm


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

def train(dl, model, loss_fn, optimizer, device):
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


def test(dl, model, loss_fn, device):
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


def objective(trial):
    epochs = 5
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    hidden_dim = trial.suggest_int("hidden_dim", 8, 512)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    activation = trial.suggest_categorical("activation_function", ["relu", "sigmoid"])

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

    best_acc = 0.
    for epoch in range(epochs):
        device = torch.device(f"cuda:{os.environ['SLURM_LOCALID']}")
        train(train_dl, model, loss_fn, optimizer, device)
        acc = test(train_dl, model, loss_fn, device)
        if acc > best_acc:
            best_acc = acc

    return best_acc


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url=os.environ["OPTUNA_STORAGE"], engine_kwargs={"connect_args": {"auth_token": os.environ["LIBSQL_TOKEN"],}})

    study = optuna.create_study(study_name="MNIST_study", storage=storage, load_if_exists=True)

    study.optimize(objective, n_trials=10)

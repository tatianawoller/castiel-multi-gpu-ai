from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.flatten = nn.Flatten().cpu()
        self.fc1 = nn.Linear(28*28, 512).cpu()
        self.fc2 = nn.Linear(512, 512).to('cuda:0')
        self.fc3 = nn.Linear(512, 512).to('cuda:1')
        self.fc4 = nn.Linear(512, 512).to('cuda:1')
        self.fc5 = nn.Linear(512, 512).to('cuda:2')
        self.fc6 = nn.Linear(512, 512).to('cuda:2')
        self.fc7 = nn.Linear(512, 512).to('cuda:3')
        self.fc = nn.Linear(512, num_classes).to('cuda:3')
        
    def forward(self, x):
        out = self.flatten(x).cpu()
        out = torch.relu(self.fc1(out)).to('cuda:0')
        out = torch.relu(self.fc2(out)).to('cuda:1')
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out)).to('cuda:2')
        out = torch.relu(self.fc5(out))
        out = torch.relu(self.fc6(out)).to('cuda:3')
        out = torch.relu(self.fc7(out))
        out = out.reshape(out.size(0), -1)
        return self.fc(out)
        


def train(num_epochs):
    torch.manual_seed(0)
    model = DenseNet()
    optimizer = optim.SGD(model.parameters(), lr=3e-2)
    #model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    batch_size = 100 
    train_dataset = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=True, with_stack=True, profile_memory=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('log')) as prof:

        start = datetime.now()
        for epoch in range(num_epochs):
            tot_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                images = images.cpu()
                labels = labels.to('cuda:3', non_blocking=True) 
            
                outputs = model(images)
                loss = criterion(outputs, labels)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                prof.step()
            print('Epoch [{}/{}], average loss: {:.4f}'.format(epoch + 1, num_epochs, tot_loss / (i+1)))
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(args.epochs)


if __name__ == '__main__':
    main()

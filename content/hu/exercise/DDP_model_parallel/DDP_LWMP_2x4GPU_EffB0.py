from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


class EfficientNetParallel(nn.Module):
    def __init__(self, dev, num_classes=10):
        super(EfficientNetParallel, self).__init__()
        self.dev = dev

        # load the model from file
        model = torch.load('EfficientNetB0/efficientnet_b0.pth', weights_only=False)
        # change the first layer to make it compatible with MNIST dataset
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.layer1 = model.features[:5].to(dev[0])
        self.layer2 = model.features[5:].to(dev[1])
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Dropout(p=0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(in_features=1280, out_features=10, bias=True)
                ).to(dev[1])

    def forward(self, x):
        out = self.layer1(x).to(self.dev[1])
        out = self.layer2(out)
        return self.classifier(out)




def train(num_epochs):
    dist.init_process_group(backend='nccl')
    #rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    dev0 = local_rank * 2
    dev1 = local_rank * 2 + 1
    dev = [dev0, dev1]
    verbose = dist.get_rank() == 0 # print only on global_rank==0
    
    torch.manual_seed(0)
    model = EfficientNetParallel(dev=[dev0, dev1])
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    #model = torch.compile(model)
    model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    batch_size = 100

    train_dataset = MNIST(root='data', train=True,
                          transform=transforms.ToTensor(), download=True)
    train_sampler = DistributedSampler(train_dataset) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True,
                              sampler=train_sampler)


    start = datetime.now()
    for epoch in range(num_epochs):
        tot_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(dev[0], non_blocking=True) 
            labels = labels.to(dev[1], non_blocking=True) 

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

        if verbose:
            print('Epoch [{}/{}], average loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                tot_loss / (i+1)))
    if verbose:
        print("Training completed in: " + str(datetime.now() - start))

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    train(args.epochs)


if __name__ == '__main__':
    main()

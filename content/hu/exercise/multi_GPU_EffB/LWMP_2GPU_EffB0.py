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


class EfficientNetParallel(nn.Module):
    def __init__(self, dev, num_classes=10):
        super(EfficientNetParallel, self).__init__()
        self.dev = dev

        # load the model from file
        model = torch.load('EfficientNetB0/efficientnet_b0.pth', weights_only=False)
        # change the first layer to make it compatible with MNIST dataset
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.layer1 = model.features[:5].to(f'cuda:{dev[0]}')
        self.layer2 = model.features[5:].to(f'cuda:{dev[1]}')
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Dropout(p=0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(in_features=1280, out_features=10, bias=True)
                ).to(f'cuda:{dev[1]}')
    
    def forward(self, x):
        out = self.layer1(x).to(f'cuda:{self.dev[1]}')
        out = self.layer2(out)
        return self.classifier(out)




def train(num_epochs):
    # the job can receive other device ordinals than 0 and 1.
    cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    dev = [int(i) for i in cuda_devices.split(',')]
    
    torch.manual_seed(0)
    model = EfficientNetParallel(dev=dev)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
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

                images = images.to(f'cuda:{dev[0]}', non_blocking=True)
                labels = labels.cuda(f'cuda:{dev[1]}', non_blocking=True)
            
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

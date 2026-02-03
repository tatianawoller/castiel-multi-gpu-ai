import torch
import os
import torchvision.models as models
from torchinfo import summary

# next two lines doesn't cause any hiccup, if the model is already downloaded
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
torch.save(model, 'EfficientNetB0/efficientnet_b0.pth')

model = torch.load('EfficientNetB0/efficientnet_b0.pth', weights_only=False)
summary(model, input_size=(1, 3, 224, 224))


print('\n----------- MODEL FEATURES -------------')
summary(model.features[0][0])
summary(model.features[:5])
summary(model.features[5:])


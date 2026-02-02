from transformers import AutoModel
import torch
from torchsummary import summary

# get a large model from here: https://huggingface.co/models?sort=downloads
model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-large')

# convert the model to torch format
torch.save(model, 'xlm-roberta-large.pth')
summary(model)

print('\n-------------------------------------')
summary(model.embeddings)
summary(model.encoder.layer[:10])
summary(model.encoder.layer[10:])
summary(model.pooler)

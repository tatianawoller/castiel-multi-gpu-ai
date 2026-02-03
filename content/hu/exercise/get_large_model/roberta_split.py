import os
import torch
import torch.nn as nn

class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()

        model = torch.load('xlm-roberta-large.pth', weights_only=False)

        self.layer1 = model.embeddings.to('cuda:0')
        self.layer2 = model.encoder.layer[:10].to('cuda:1')
        self.layer3 = model.encoder.layer[10:].to('cuda:2')
        self.layer4 = model.pooler.to('cuda:3')

        # Verify device placement
        print("Layer1 (embeddings) device:", next(self.layer1.parameters()).device)
        print("Layer2 (first 10 layers) device:", next(self.layer2[0].parameters()).device)
        print("Layer3 (remaining layers) device:", next(self.layer3[0].parameters()).device)
        print("Layer4 (pooler) device:", next(self.layer4.parameters()).device)

    def forward(self, input_ids, attention_mask=None):
        # Ensure input_ids and attention_mask are on the correct device
        input_ids = input_ids.to('cuda:0')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda:0')

        # Layer 1: Embeddings
        hidden_states = self.layer1(input_ids)
        hidden_states = hidden_states.to('cuda:1')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda:1')

        # Layer 2: First 10 encoder layers
        for layer in self.layer2:
            layer_output = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_output[0]
        hidden_states = hidden_states.to('cuda:2')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda:2')

        # Layer 3: Remaining encoder layers
        for layer in self.layer3:
            layer_output = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_output[0]
        hidden_states = hidden_states.to('cuda:3')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda:3')

        # Layer 4: Pooler
        pooled_output = self.layer4(hidden_states)

        return pooled_output

model = Roberta()

os.system('nvidia-smi')

batch_size, seq_len = 1, 512
input_ids = torch.randint(0, 250002, (batch_size, seq_len))  # Random token IDs
attention_mask = torch.ones(batch_size, seq_len)  # Dummy attention mask
with torch.no_grad():
    output = model(input_ids, attention_mask)
    print("Forward pass output shape:", output.shape)

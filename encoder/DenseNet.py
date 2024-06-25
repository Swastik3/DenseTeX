import torch
import torch.nn as nn
from torchvision import models
from DataLoader import get_dataloader

# Define the DenseNet169 model
model = models.densenet169(pretrained=True)

#print(*list(model.children()))

# Remove the final fully connected layer to get the final feature maps
model = nn.Sequential(*list(model.children())[:-1])

train_loader = get_dataloader(batch_size=10)

# Assuming the DataLoader for training is defined as train_loader
i = 0
for images, labels in train_loader:
    #outputs = model(images)  # Get the feature maps
    print (labels[0])
    break
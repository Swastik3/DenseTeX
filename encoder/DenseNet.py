import torch
import torch.nn as nn
from torchvision import models

# Define the DenseNet169 model
model = models.densenet169(pretrained=True)

#print(*list(model.children()))

# Remove the final fully connected layer to get the final feature maps
model = nn.Sequential(*list(model.children())[:-1])

# Assuming the DataLoader for training is defined as train_loader
for images, labels in train_loader:
    outputs = model(images)  # Get the feature maps

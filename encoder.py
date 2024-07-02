import torch
import torch.nn as nn
from torchvision import models
from DataLoader import get_dataloader
from DenseNet import PositionalEncoding2D, InputEmbeddings

# Define the DenseNet169 model
model = models.densenet169(pretrained=True)
#print(*list(model.children()))

# Remove the final fully connected layer to get the final feature maps
model = nn.Sequential(*list(model.children())[:-1])
model.add_module('PositionalEncoding2D', PositionalEncoding2D(1664,800,400)) # hardcoded this based on denseNet output size
model.add_module('InputEmbeddings', InputEmbeddings(1664,768))
#yet to automate that, the 7x7 is dependent on the input size of the model, it can be anything. Still need to automate this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Assuming the DataLoader for training is defined as train_loader
train_loader = get_dataloader(batch_size=10)

def get_batch(data_loader):
    for images, labels in data_loader:
        model(images)
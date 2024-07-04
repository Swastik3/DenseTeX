from DataLoader import CustomDataset
import random
import torch
from torch.utils.data import Subset, DataLoader



batch_size = 10
subset_size = 128


def get_subset_dataloader(batch_size=batch_size, image_dir = './data/UniMER-1M/images/', label_file = './data/UniMER-1M/train.txt', subset_size = subset_size) :

    full_dataset = CustomDataset(image_dir = image_dir, label_file = label_file, transform = None, cache_file = 'valid_indices_cache.pkl')

    # random select a subset of indices
    subset_indices = random.sample(range(len(full_dataset)), subset_size)
    #create a subset dataset
    subset_dataset = Subset(full_dataset, subset_indices)
    # create a dataloader for the subset
    subset_dataloader = DataLoader(subset_dataset, batch_size = batch_size, shuffle = True)

    return subset_dataloader


x = get_subset_dataloader()

for i, (image, label) in enumerate(x):
    print(i, image.shape)

    print ('\n')
    print(label)
    print ('\n')

    if i == 5:
        break   
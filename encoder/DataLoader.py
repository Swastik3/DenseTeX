import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, cache_file='valid_indices_cache.pkl'):
        self.image_dir = image_dir
        self.transform = transform
        self.cache_file = cache_file
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = f.readlines()
        self.valid_indices = self._load_or_create_valid_indices()

    def _load_or_create_valid_indices(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                valid_indices = pickle.load(f)
        else:
            valid_indices = self._get_valid_indices()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(valid_indices, f)
        return valid_indices

    def _get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.labels)):
            img_name = os.path.join(self.image_dir, f"{idx:07d}.png")
            if os.path.exists(img_name):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_name = os.path.join(self.image_dir, f"{actual_idx:07d}.png")
        image = Image.open(img_name).convert('RGB')
        label = self.labels[actual_idx].strip()
        
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            image = transform(image)
        
        return image, label

def get_dataloader(batch_size, image_dir='../../UniMER-1M/images/', label_file='../../UniMER-1M/train.txt', transform=None, cache_file='valid_indices_cache.pkl'):
    dataset = CustomDataset(image_dir=image_dir, label_file=label_file, transform=transform, cache_file=cache_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, cache_file='valid_indices_cache.pkl', shuffle=True):
        self.image_dir = image_dir
        self.transform = transform
        self.cache_file = cache_file
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = f.readlines()
        self.valid_indices = self._load_or_create_valid_indices()
        self.shuffle = True

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
        image = self.resize_and_pad(image, 800, 400)
        
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image = transform(image)
        
        return image, label

    def resize_and_pad(self, image, target_width, target_height):
        # Calculate the ratio to maintain the aspect ratio
        original_width, original_height = image.size
        ratio = min(target_width / original_width, target_height / original_height)
    
        # Resize the image while maintaining the aspect ratio
        new_size = (int(original_width * ratio), int(original_height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)
    
        # Create a new image with the specified target size and a white background
        new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    
        # Calculate the position to paste the resized image on the white background
        paste_x = (target_width - new_size[0]) // 2
        paste_y = (target_height - new_size[1]) // 2
    
        # Paste the resized image onto the white background
        new_image.paste(resized_image, (paste_x, paste_y))
    
        return new_image


def distributed_sampler(rank, world_size, image_dir, label_file, transform=None, cache_file='valid_indices_cache.pkl'):
    """ Create a distributed sampler for the dataset 
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        dataset: The dataset to be distributed
    """
    dataset = CustomDataset(image_dir=image_dir, label_file=label_file, transform=transform, cache_file=cache_file)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset = dataset,
        num_replicas= world_size,
        rank=rank,
        shuffle=True
    )
    return dist_sampler


def get_dataloader(batch_size, image_dir='../../UniMER-1M/images/', label_file='../../UniMER-1M/train.txt', transform=None, cache_file='valid_indices_cache.pkl', num_workers=1, sampler=None):
    dataset = CustomDataset(image_dir=image_dir, label_file=label_file, transform=transform, cache_file=cache_file)

    if sampler is not None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


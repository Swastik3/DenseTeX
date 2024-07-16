import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from torchvision import transforms
from PIL import Image
import pickle
import random

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



class CustomDataLoader:
    def __init__(self, image_dir, label_file, process_rank, num_processes, transform=None, cache_file='valid_indices_cache.pkl', shuffle=True, batch_size=1, num_workers=1, sampler=None):
        self.dataset = CustomDataset(image_dir=image_dir, label_file=label_file, transform=transform, cache_file=cache_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.process_rank = process_rank
        self.num_processes = num_processes
        # Create a DistributedSampler for multi-GPU training
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            num_replicas=self.num_processes,
            rank=self.process_rank,
            shuffle=self.shuffle)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.sampler is not None:
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=self.sampler)
        else:
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return iter(dataloader)
    
    def get_epoch(self):
        return self.sampler.epoch
    
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

class SubsetCustomDataLoader:
    def __init__(self, image_dir, label_file, process_rank, num_processes, subset_size, 
                 transform=None, cache_file='valid_indices_cache.pkl', shuffle=True, 
                 batch_size=1, num_workers=1, seed=42):
        # Initialize the full dataset
        self.full_dataset = CustomDataset(image_dir=image_dir, label_file=label_file, 
                                          transform=transform, cache_file=cache_file)
        
        # Create a subset of the dataset
        total_size = len(self.full_dataset)
        subset_size = min(subset_size, total_size)
        
        # Use a fixed seed for reproducibility
        random.seed(seed)
        subset_indices = random.sample(range(total_size), subset_size)
        
        self.dataset = Subset(self.full_dataset, subset_indices)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # Create a DistributedSampler for multi-GPU training
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.num_processes,
            rank=self.process_rank,
            shuffle=self.shuffle
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # Shuffle is handled by DistributedSampler
            num_workers=self.num_workers, 
            sampler=self.sampler,
            pin_memory=True  # This can speed up data transfer to GPU
        )
        return iter(dataloader)

    def get_epoch(self):
        return self.sampler.epoch

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
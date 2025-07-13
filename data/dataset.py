import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import albumentations as A
import numpy as np
import cv2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=256):
        self.root_dir = root_dir
        
        # Now images are directly in the root_dir (train or test)
        self.real_images = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # Augmentations to simulate manipulations
        self.fake_augment = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
            ], p=1.0),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.MultiplicativeNoise(p=0.5),
                A.ISONoise(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.ColorJitter(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Blur(blur_limit=7, p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.MedianBlur(blur_limit=7, p=0.5),
            ], p=0.5),
        ])
            
    def __len__(self):
        return len(self.real_images)
    
    def __getitem__(self, idx):
        # Load real image
        real_img_path = os.path.join(self.root_dir, self.real_images[idx])
        real_image = Image.open(real_img_path).convert('RGB')
        
        # Create fake version through augmentation
        real_np = np.array(real_image)
        fake_np = self.fake_augment(image=real_np)['image']
        fake_image = Image.fromarray(fake_np)
        
        # Apply transformations
        if self.transform:
            real_image = self.transform(real_image)
            fake_image = self.transform(fake_image)
        
        return real_image, fake_image

def create_dataloaders(config):
    # Create datasets
    train_dataset = DeepfakeDataset(
        root_dir=config.data.train_path,
        image_size=config.data.image_size
    )
    
    val_dataset = DeepfakeDataset(
        root_dir=config.data.val_path,
        image_size=config.data.image_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 
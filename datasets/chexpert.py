import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class CheXpert(Dataset):
    def __init__(self, root_dir, image_size, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        print(f"Debug: Looking for images in {self.root_dir}")
        
        # Get all image paths recursively
        if split == 'calculate':
            split = 'valid'
            self.sample_mode = True
        else:
            self.sample_mode = False
            
        self.root_dir = os.path.join(root_dir, split)
        print(f"Debug: Looking for images in {self.root_dir}")
        
        # Get all image paths recursively
        self.image_paths = []
        patient_count = 0
        study_count = 0
        image_count = 0
        
        # Walk through patient folders
        for patient in os.listdir(self.root_dir):
            patient_path = os.path.join(self.root_dir, patient)
            if not os.path.isdir(patient_path):
                continue
            patient_count += 1
                
            # Walk through study folders
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue
                study_count += 1
                
                # Get images in study folder
                for file in os.listdir(study_path):
                    if not file.startswith('.') and file.endswith('.jpg'):
                        full_path = os.path.join(study_path, file)
                        self.image_paths.append(full_path)
                        image_count += 1
                        # If in sample mode, only get first image
                        if self.sample_mode:
                            break
                    if self.sample_mode and len(self.image_paths) > 0:
                        break
                if self.sample_mode and len(self.image_paths) > 0:
                    break
                        
        print(f"Debug: Found {patient_count} patients")
        print(f"Debug: Found {study_count} studies")
        print(f"Debug: Found {image_count} images")
        if len(self.image_paths) > 0:
            print(f"Debug: Example path: {self.image_paths[0]}")
        else:
            raise RuntimeError(f"No images found in {self.root_dir}")
            
        # Basic transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        image = self.transform(image)
        
        # Create noisy version
        noise = torch.randn_like(image)
        noisy_image = image + 0.1 * noise
        noisy_image = torch.clamp(noisy_image, -1, 1)
        
        return {
            'LD': noisy_image,
            'FD': image,
            'case_name': os.path.basename(img_path),
            'path': img_path
        }
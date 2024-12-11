import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        self.categorical_columns = ['Sex', 'Frontal/Lateral', 'AP/PA']
        
        self.label_columns = [col for col in self.data.columns[1:] if col not in self.categorical_columns]

        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].fillna('missing'))  # Encode 'missing' values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0] 
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        label = self.data.iloc[idx, 1:].values
        label = [float(l) if isinstance(l, (int, float)) else float('nan') for l in label]
        
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

train_dataset = CheXpertDataset(csv_file='CheXpert-v1.0-small/train.csv', transform=transform)
valid_dataset = CheXpertDataset(csv_file='CheXpert-v1.0-small/valid.csv', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True)

from diffusers import DiffusionPipeline, DDPMScheduler
from torch.optim import Adam
import torch
from accelerate import Accelerator
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm 

accelerator = Accelerator()

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipe = accelerator.prepare(pipe)
pipe = pipe.to(accelerator.device)

optimizer = Adam(pipe.unet.parameters(), lr=1e-5)
epochs = 5

scheduler = DDPMScheduler()

#Load model checkpoint to continue training from saved epoch 5
checkpoint_epoch = 0  # Start from epoch 1 or any other epoch you saved
checkpoint_path_unet = f"unet_epoch_{checkpoint_epoch}.pth"
checkpoint_path_vae = f"vae_epoch_{checkpoint_epoch}.pth"

if os.path.exists(checkpoint_path_unet) and os.path.exists(checkpoint_path_vae):
    print(f"Loading checkpoint for epoch {checkpoint_epoch}...")
    pipe.unet.load_state_dict(torch.load(checkpoint_path_unet, weights_only=True))
    pipe.vae.load_state_dict(torch.load(checkpoint_path_vae, weights_only=True))
else:
    print("No checkpoint found, starting from scratch.")
    checkpoint_epoch = 0
# Training loop
for epoch in range(checkpoint_epoch, epochs):
    pipe.unet.train() 
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for step, (images, _) in enumerate(pbar):
            images = accelerator.prepare(images)

            images = images.repeat(1, 3, 1, 1) 

            images = (images * 2) - 1 

            pipe.vae = pipe.vae.to(accelerator.device) 
            images = images.to(pipe.vae.device)

            latents = pipe.vae.encode(images).latent_dist.sample() 

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.size(0),), device=latents.device)

            noise = torch.randn_like(latents) 

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = torch.zeros(images.size(0), 77, 768, device=latents.device)

            optimizer.zero_grad()

            output = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states)[0]
            loss = ((output - latents) ** 2).mean()

            loss.backward() 
            optimizer.step() 

            running_loss += loss.item()

            pbar.set_postfix({'Loss': loss.item(), 'Running Loss': running_loss})
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {running_loss/len(train_loader):.4f}")

    # Validation phase
    pipe.unet.eval()
    pipe.vae.eval() 

    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            images, _ = batch
            
            images = accelerator.prepare(images)
            
            images = images.repeat(1, 3, 1, 1)

            images = (images * 2) - 1 

            pipe.vae = pipe.vae.to(accelerator.device) 

            images = images.to(pipe.vae.device)

            latents = pipe.vae.encode(images).latent_dist.sample()

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.size(0),), device=latents.device)

            noise = torch.randn_like(latents) 

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = torch.zeros(images.size(0), 77, 768, device=latents.device)

            output = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states)[0]

            loss = ((output - latents) ** 2).mean()

            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(valid_loader)}")

    # Save model checkpoint after each epoch
    torch.save(pipe.unet.state_dict(), f"unet_epoch_{epoch+1}.pth")
    torch.save(pipe.vae.state_dict(), f"vae_epoch_{epoch+1}.pth")

from torchvision import transforms

def generate_and_save_images(pipe, epoch, num_images):
    pipe.unet.eval()
    pipe.vae.eval()
    
    with torch.no_grad():
        latents = torch.randn(num_images, pipe.unet.in_channels, 390, 320).to(pipe.device)

        decoded_output = pipe.vae.decode(latents)

        images = (decoded_output.sample + 1) / 2

        images = torch.clamp(images, 0.0, 1.0)

        for i, image in enumerate(images):
            image_pil = transforms.ToPILImage()(image.cpu())
            image_pil.save(f"generated_image3_epoch{epoch}_img{i}.jpg")

generate_and_save_images(pipe, epochs, num_images=1) 

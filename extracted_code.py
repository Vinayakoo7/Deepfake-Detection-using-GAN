import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

#Constants and hyperparameters for the model, training process, and dataset are defined.
CUDA = True
DATASET_PATH = './faces'  # Path to your local dataset of face images
BATCH_SIZE = 32  # Reduced batch size for larger images
IMAGE_CHANNEL = 3  # RGB images
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 128  # Target image size 128x128
D_HIDDEN = 64
EPOCH_NUM = 10
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1

if not torch.cuda.is_available():
    CUDA = False
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    # Enable deterministic behavior for reproducibility
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

print(f"PyTorch version: {torch.__version__}")
if CUDA:
    print(f"CUDA version: {torch.version.cuda}\n")
print(f"Using device: {device}")

# Custom Dataset for loading face images
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                           if os.path.isfile(os.path.join(root_dir, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as dummy label

# Image transformations - resize from 256x256 to 128x128
transform = transforms.Compose([
    transforms.Resize(X_DIM),
    transforms.CenterCrop(X_DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

# Create the dataset and dataloader
try:
    dataset = FaceDataset(root_dir=DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True if CUDA else False)
    print(f"Dataset loaded with {len(dataset)} images")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure the directory {DATASET_PATH} exists and contains image files")
    sys.exit(1)

# Display: Visualize a batch of real images
try:
    real_batch = next(iter(dataloader))[0]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig("real_samples.png")
except Exception as e:
    print(f"Error displaying images: {e}")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Generator for 128x128 RGB images
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input layer: From latent vector to a small "feature map"
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 16),
            nn.ReLU(True),
            # 4x4

            # Upsampling layers
            nn.ConvTranspose2d(G_HIDDEN * 16, G_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 8x8

            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 16x16

            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 32x32

            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # 64x64

            # Final layer to get to 128x128
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 128x128
        )

    def forward(self, input):
        return self.main(input)

# Discriminator for 128x128 RGB images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 128x128
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64

            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32

            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16

            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8

            nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4

            nn.Conv2d(D_HIDDEN * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 1x1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Create the generator
netG = Generator().to(device)
netG.apply(weights_init)
print("Generator Architecture:")
print(netG)

# Create the discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)
print("Discriminator Architecture:")
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors for visualization
viz_noise = torch.randn(16, Z_DIM, 1, 1, device=device)  # Reduced to 16 for larger images

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(EPOCH_NUM):
    with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch+1}/{EPOCH_NUM}", dynamic_ncols=True) as tepoch:
        for i, data in enumerate(tepoch):
            real_images = data[0].to(device, non_blocking=True)
            b_size = real_images.size(0)
            
            # Train Discriminator with real images
            netD.zero_grad()
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train Discriminator with fake images
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(FAKE_LABEL)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(REAL_LABEL)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Update progress bar
            tepoch.set_postfix(Loss_D=errD.item(), Loss_G=errG.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)

            # Save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Save generated images periodically
            if (iters % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(viz_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                # Save current generator output
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title(f"Generated Images (Epoch {epoch+1}, Iter {iters})")
                plt.imshow(np.transpose(img_list[-1], (1,2,0)))
                plt.savefig(f"generated_e{epoch+1}_i{iters}.png")
                plt.close()

            iters += 1

# Save the trained models
torch.save(netG.state_dict(), "generator.pth")
torch.save(netD.state_dict(), "discriminator.pth")

# Plot loss curves
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curves.png")

# Final visualization: Real vs Generated
# Get a batch of real images
real_batch = next(iter(dataloader))[0][:16]

# Generate a batch of fake images
with torch.no_grad():
    fake_batch = netG(viz_noise).detach().cpu()

# Plot real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch, padding=5, normalize=True).cpu(),(1,2,0)))

# Plot fake images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(fake_batch, padding=5, normalize=True),(1,2,0)))
plt.savefig("final_comparison.png")
plt.show()

# Create animation of the generator's progress
import matplotlib.animation as animation

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# Save the animation as a gif
ani.save("face_gan_training.gif", writer='pillow', fps=4)

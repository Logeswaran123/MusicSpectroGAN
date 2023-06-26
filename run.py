import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from dcgan_model import DCMusicSpectroGAN
from train import train
from utils import *

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    device = "cuda"
    dc_msgan = DCMusicSpectroGAN(device)

    # Hyperparameters
    batch_size = 64
    image_size = (64, 64)
    nc = 1    # number of channels
    nz = 100  # Size of the latent vector
    ngf = 64  # Number of generator filters
    ndf = 64  # Number of discriminator filters
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5

    netG, netD = dc_msgan.model(nz, ngf, nc, ndf)

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageFolder(root=f"{os.getcwd()}\spectogram_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train
    train(device, nz, lr, beta1, netD, netG, dataloader, num_epochs)


if __name__ == "__main__":
    main()
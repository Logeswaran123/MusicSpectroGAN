import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from utils import *


def train(device, nz, lr, beta1, netD, netG, dataloader, num_epochs):
    device = torch.device(device)

    path = os.getcwd() + "\generated_images"
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # Define the size of the fixed noise vector
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Define lists to store generator and discriminator losses
    G_losses = []
    D_losses = []

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator network
            netD.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), 1.0, device=device)

            output = netD(real_images).view(-1)
            errD_real = F.binary_cross_entropy(output, label) 
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0.0)

            output = netD(fake_images.detach()).view(-1)
            errD_fake = F.binary_cross_entropy(output, label) 
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator network
            netG.zero_grad()
            label.fill_(1.0)

            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Store generator and discriminator losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Print training statistics
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} "
                    f"Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} "
                    f"D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # Save generated images
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            save_image(fake, fr"{path}\epoch_{epoch + 1}.png", normalize=True)

    # Generate and save final images
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        save_image(fake, fr"{path}\final_result.png", normalize=True)

    # Plot and save the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}\loss_curves.png")
    plt.show()
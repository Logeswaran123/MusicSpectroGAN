import os

from utils import *


def train_dcgan(device, nz, lr, beta1, netD, netG, dataloader, num_epochs):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torchvision.utils import save_image

    torch.manual_seed(42)
    device = torch.device(device)

    path = os.getcwd() + "\generated_images"
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
    plt.close()


def train_cgan(spectrograms, labels, latent_dim, num_classes, epochs, batch_size, discriminator, generator):
    from tensorflow import keras    

    save_path = os.getcwd() + "\generated_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Compile the discriminator
    discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=["accuracy"])

    # Compile the generator
    generator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    discriminator.summary()
    generator.summary()

    # Combine the generator and discriminator into a single model
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_labels = keras.Input(shape=(num_classes,))
    gan_output = discriminator([generator([gan_input, gan_labels]), gan_labels])
    gan = keras.Model([gan_input, gan_labels], gan_output)
    gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    # Training loop
    num_batches = spectrograms.shape[0] // batch_size
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))

        for batch in range(num_batches):
            # Sample real spectrograms and labels
            real_spectrograms = spectrograms[batch * batch_size : (batch + 1) * batch_size]
            real_class_labels = labels[batch * batch_size : (batch + 1) * batch_size]

            # Generate random latent vectors and labels
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            random_class_labels = np.random.randint(0, num_classes, size=(batch_size, num_classes))

            # Generate fake spectrograms using the generator
            generated_spectrograms = generator.predict([random_latent_vectors, random_class_labels])

            # Concatenate real and fake spectrograms and labels
            combined_spectrograms = np.concatenate([real_spectrograms, generated_spectrograms])
            combined_class_labels = np.concatenate([real_class_labels, random_class_labels])

            # Create labels for real and fake spectrograms
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            combined_labels = np.concatenate([real_labels, fake_labels])

            # Train the discriminator
            discriminator_loss = discriminator.train_on_batch([combined_spectrograms, combined_class_labels], combined_labels)

            # Train the generator
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            random_labels = np.random.randint(0, num_classes, size=(batch_size, num_classes))
            generator_loss = gan.train_on_batch([random_latent_vectors, random_labels], real_labels)

            # Print the progress
            print("Batch {}/{} | D loss: {:.4f} | G loss: {:.4f}".format(
                batch + 1, num_batches, discriminator_loss[0], generator_loss))

        # Generate and save example spectrograms
        generate_and_save_images(save_path, epoch, latent_dim, num_classes, generator)
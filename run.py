import argparse
import os

from dcgan_model import DCMusicSpectroGAN
from cgan_model import CMusicSpectroGAN
from train import train_dcgan, train_cgan
from utils import *

# Set random seed for reproducibility
np.random.seed(42)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gan', "--gan", required=True, type=str, choices=["dcgan", "cgan"],
                                        help="Specify GAN model architecture.")
    return parser


def main():
    args = argparser().parse_args()

    if args.gan == "dcgan":
        device = "cuda"

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

        # Load the dataset
        dataloader = pt_load_dataset(f"{os.getcwd()}\spectogram_images", image_size, batch_size)

        dc_msgan = DCMusicSpectroGAN(device)
        netG, netD = dc_msgan.model(nz, ngf, nc, ndf)

        # Train
        train_dcgan(device, nz, lr, beta1, netD, netG, dataloader, num_epochs)

    if args.gan == "cgan":
        # Hyperparameters
        latent_dim = 100  # Dimension of the random latent vector
        num_epochs = 10  # Number of training epochs
        batch_size = 32  # Batch size for training

        # Load the dataset
        spectrograms, labels, num_classes = load_dataset(f"{os.getcwd()}\spectogram_images", batch_size, 64, 64)
        image_size = spectrograms[0].shape

        c_msgan = CMusicSpectroGAN()

        discriminator = c_msgan.discriminator(image_size, num_classes)
        generator = c_msgan.generator(latent_dim, num_classes, (image_size[0]//4, image_size[1]//4))

        # Train
        train_cgan(spectrograms, labels, latent_dim, num_classes, num_epochs, batch_size, discriminator, generator)


if __name__ == "__main__":
    main()
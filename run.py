

from model import cMusicSpectroGAN
from train import train


def main():
    music_spectro_gan = cMusicSpectroGAN()

    # Instantiate
    latent_dim=100 # Our latent space has 100 dimensions. We can change it to any number
    gen_model = music_spectro_gan._generator(latent_dim)

    # Show model summary and plot model diagram
    gen_model.summary()

    # Instantiate
    dis_model = music_spectro_gan._discriminator()

    # Show model summary and plot model diagram
    dis_model.summary()

    # Instantiate
    gan_model = music_spectro_gan.model(gen_model, dis_model)

    # Show model summary and plot model diagram
    gan_model.summary()


if __name__ == "__main__":
    main()
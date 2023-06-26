import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_path = "spectogram_images"


def load_dataset(batch_size = 32, spectrogram_height = 512, spectrogram_width = 512):
    datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize

    data_generator = datagen.flow_from_directory(
        dataset_path,
        color_mode="grayscale",
        target_size=(spectrogram_height, spectrogram_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    num_classes = data_generator.num_classes
    spectrograms = []
    labels = []

    for batch in data_generator:
        spectrograms.extend(batch[0])
        labels.extend(batch[1])

        # Break the loop when all data is extracted
        if len(spectrograms) >= data_generator.n:
            break

    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    return spectrograms, labels, num_classes


def build_generator(latent_dim, num_classes, in_shape):
    inputs = keras.Input(shape=(latent_dim,))
    labels = keras.Input(shape=(num_classes,))

    x = layers.Concatenate()([inputs, labels])

    x = layers.Dense(128 * in_shape[0] * in_shape[1], activation="relu")(x)
    x = layers.Reshape((in_shape[0], in_shape[1], 128))(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(1, 7, padding="same", activation="tanh")(x)

    model = keras.Model([inputs, labels], x)
    return model


def build_discriminator(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    labels = keras.Input(shape=(num_classes,))

    x = layers.Conv2D(64, 3, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    x = layers.Concatenate()([x, labels])

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model([inputs, labels], outputs)
    return model


def train_cgan(spectrograms, labels, latent_dim, num_classes, epochs, batch_size):
    # Build and compile the discriminator
    image_size = spectrograms[0].shape
    discriminator = build_discriminator(image_size, num_classes)
    discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=["accuracy"])

    # Build and compile the generator
    generator = build_generator(latent_dim, num_classes, (image_size[0]//4, image_size[1]//4))
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

    # Generate and save example spectrograms during training
    def generate_and_save_images(epoch, latent_dim, num_classes):
        random_latent_vectors = np.random.normal(size=(10, latent_dim))
        random_labels = np.random.randint(0, num_classes, size=(10, num_classes))
        generated_spectrograms = generator.predict([random_latent_vectors, random_labels])

        fig, axs = plt.subplots(2, 5, figsize=(12, 6))
        count = 0
        for i in range(2):
            for j in range(5):
                axs[i, j].imshow(generated_spectrograms[count, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                count += 1
        plt.tight_layout()
        plt.savefig("generated_spectrograms_epoch_{:04d}.png".format(epoch))
        plt.close()

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
        generate_and_save_images(epoch, latent_dim, num_classes)


latent_dim = 100  # Dimension of the random latent vector
epochs = 10  # Number of training epochs
batch_size = 1  # Batch size for training

spectrograms, labels, num_classes = load_dataset(batch_size, 64, 64)

train_cgan(spectrograms, labels, latent_dim, num_classes, epochs, batch_size)
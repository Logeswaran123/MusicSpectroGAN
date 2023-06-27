from tensorflow import keras
from tensorflow.keras import layers


class CMusicSpectroGAN():
    def __init__(self) -> None:
        pass

    def generator(self, latent_dim, num_classes, in_shape):
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

    def discriminator(self, input_shape, num_classes):
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
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



class cMusicSpectroGAN():
    def __init__(self, ) -> None:
        pass

    def model(self, generator, discriminator):
        # We don't want to train the weights of discriminator at this stage. Hence, make it not trainable
        discriminator.trainable = False

        # Get Generator inputs / outputs
        gen_latent, gen_label = generator.input # Latent and label inputs from the generator
        gen_output = generator.output # Generator output image

        # Connect image and label from the generator to use as input into the discriminator
        gan_output = discriminator([gen_output, gen_label])

        # Define GAN model
        model = Model([gen_latent, gen_label], gan_output, name="cDCGAN")

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model

    def _generator(self, latent_dim, in_shape=(7,7,1), n_cats=10):
        # Label Inputs
        in_label = Input(shape=(1,), name='Generator-Label-Input-Layer') # Input Layer
        lbls = Embedding(n_cats, 50, name='Generator-Label-Embedding-Layer')(in_label) # Embed label to vector

        # Scale up to image dimensions
        n_nodes = in_shape[0] * in_shape[1] 
        lbls = Dense(n_nodes, name='Generator-Label-Dense-Layer')(lbls)
        lbls = Reshape((in_shape[0], in_shape[1], 1), name='Generator-Label-Reshape-Layer')(lbls) # New shape

        # Generator Inputs (latent vector)
        in_latent = Input(shape=latent_dim, name='Generator-Latent-Input-Layer')

        # Image Foundation 
        n_nodes = 7 * 7 * 128 # number of nodes in the initial layer
        g = Dense(n_nodes, name='Generator-Foundation-Layer')(in_latent)
        g = ReLU(name='Generator-Foundation-Layer-Activation-1')(g)
        g = Reshape((in_shape[0], in_shape[1], 128), name='Generator-Foundation-Layer-Reshape-1')(g)

        # Combine both inputs so it has two channels
        concat = Concatenate(name='Generator-Combine-Layer')([g, lbls])

        # Hidden Layer 1
        g = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-Hidden-Layer-1')(concat)
        g = ReLU(name='Generator-Hidden-Layer-Activation-1')(g)

        # Hidden Layer 2
        g = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-Hidden-Layer-2')(g)
        g = ReLU(name='Generator-Hidden-Layer-Activation-2')(g)

        # Output Layer (Note, we use only one filter because we have a greysclae image. Color image would have three
        output_layer = Conv2D(filters=1, kernel_size=(7,7), activation='tanh', padding='same', name='Generator-Output-Layer')(g)

        # Define model
        model = Model([in_latent, in_label], output_layer, name='Generator')
        return model

    def _discriminator(self, in_shape=(28,28,1), n_cats=10):
        # Label Inputs
        in_label = Input(shape=(1,), name='Discriminator-Label-Input-Layer') # Input Layer
        lbls = Embedding(n_cats, 50, name='Discriminator-Label-Embedding-Layer')(in_label) # Embed label to vector

        # Scale up to image dimensions
        n_nodes = in_shape[0] * in_shape[1] 
        lbls = Dense(n_nodes, name='Discriminator-Label-Dense-Layer')(lbls)
        lbls = Reshape((in_shape[0], in_shape[1], 1), name='Discriminator-Label-Reshape-Layer')(lbls) # New shape

        # Image Inputs
        in_image = Input(shape=in_shape, name='Discriminator-Image-Input-Layer')

        # Combine both inputs so it has two channels
        concat = Concatenate(name='Discriminator-Combine-Layer')([in_image, lbls])

        # Hidden Layer 1
        h = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', name='Discriminator-Hidden-Layer-1')(concat)
        h = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1')(h)

        # Hidden Layer 2
        h = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', name='Discriminator-Hidden-Layer-2')(h)
        h = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2')(h)
        h = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid', name='Discriminator-MaxPool-Layer-2')(h) # Max Pool

        # Flatten and Output Layers
        h = Flatten(name='Discriminator-Flatten-Layer')(h) # Flatten the shape
        h = Dropout(0.2, name='Discriminator-Flatten-Layer-Dropout')(h) # Randomly drop some connections for better generalization

        output_layer = Dense(1, activation='sigmoid', name='Discriminator-Output-Layer')(h) # Output Layer

        # Define model
        model = Model([in_image, in_label], output_layer, name='Discriminator')

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
        return model
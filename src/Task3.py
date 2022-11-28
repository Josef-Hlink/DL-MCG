# %% [markdown]
# # Task 3: Generative Models

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Imports

# %%
# stdlib
import os

# pip
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape
import matplotlib.pyplot as plt

# local
from utils import get_dirs

# %% [markdown]
# ### Fix directories

# %%
DIRS = get_dirs(os.path.abspath('') + os.sep + 'Task3.ipynb')
print('\033[1m' + 'Directories:' + '\033[0m')
for dir_name, path in DIRS.items():
    print(f'{dir_name:<7} {path}')

# %% [markdown]
# ### Fix random seed
# 
# Setting a random seed to make sure the results can be replicated.

# %%
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% [markdown]
# ### Dataset

# %%
def convert_to_npy(path: str) -> np.ndarray:
    """
    Converts a directory of images to a numpy array.
    Returns numpy array of shape (N, H, W, C) where N is the number of images,
    H and W are the height and width of the images, and C is the number of channels.
    """

    images = []
    for file in os.listdir(path):
        if not file.endswith('.png'):
            continue
        image = PIL.Image.open(os.path.join(path, file))
        image = np.array(image)
        images.append(image)
    return np.array(images)

def load_from_npy(path: str, scale: bool = False) -> np.ndarray:
    """
    Loads a numpy array from a file.
    scale: If False, channels are in the range [0, 1]. If True, channels are in range [-1, 1]
    """

    X = np.load(path)
    if scale:
        X = (X - 127.5) * 2
    return X / 255.0

# %% [markdown]
# The data should be present (unzipped) in the `data` folder.
# Because this data is a collection of multiple sources, we want to remove duplicates when loading it for the first time.
# The pruned dataset will be saved as a NumPy array in the `data` folder (cats.npy).
# We are working with a generative model, so we will not need to use any labels.

# %%
if not os.path.exists(DIRS['data'] + 'cats.npy'):
    images = convert_to_npy(path=DIRS['data'] + 'cats/')
    # remove duplicates
    samples_before = images.shape[0]
    images = np.unique(images, axis=0)
    samples_after = images.shape[0]
    print(f'Removed {samples_before - samples_after} duplicates')
    np.save(DIRS['data'] + 'cats.npy', images)
else:
    images = load_from_npy(DIRS['data'] + 'cats.npy')

dataset = images
print(f'dataset shape: {dataset.shape}, min: {dataset.min()}, max: {dataset.max()}')

# %% [markdown]
# This function will be called later on, but we also call it once to show that it works. It is used to plot the images in a 3 x 3 grid. 

# %%
def grid_plot(
    images: np.ndarray,
    epoch: int = 0,
    name: str = '',
    n: int = 3,
    save: bool = False,
    scale: bool = False
    ) -> None:
    """
    Plot a grid of n*n images, note that images.shape[0] must equal n*n.
    """

    if scale:
        images = (images + 1) / 2.0
    
    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))
    for i in range(n * n):
        ax = axes[i // n, i % n]
        ax.imshow(images[i])
        ax.axis('off')
    fig.suptitle(f'{name} {epoch}', fontsize=14)
    
    if save:
        fig.savefig(DIRS['figs'] + f'generated_plot_e{epoch+1}03d_f.png')
        plt.close(fig)
    else:
        plt.show()


ri_mask = np.random.choice(range(0, dataset.shape[0]), 9, replace=False)
random_images = dataset[ri_mask]
grid_plot(random_images, name='Cats dataset (64x64x3)', n=3)

# %% [markdown]
# ## 2.1. Convolutional & De-convolutional
# 
# Here are the convolutional and de-convolutional neural networks. These function as the basis for the encoding and de-encoding networks of the VAE and GAN. 
#  
# 
# 
# #### Code for building these components:

# %%
def build_conv_net(
    in_shape: tuple = (64, 64, 3),
    out_shape: int = 512,
    n_downsampling_layers: int = 4,
    filters: int = 128,
    out_activation: str = 'sigmoid'
    ) -> tf.keras.Sequential:
    """
    Build a basic convolutional network
    """
    
    model = tf.keras.Sequential()
    default_args = dict(
        kernel_size = (3, 3),
        strides = (2, 2),
        padding = 'same',
        activation = 'relu'
    )

    model.add(Conv2D(
        input_shape = in_shape,
        **default_args,
        filters = filters
    ))

    for _ in range(n_downsampling_layers):
        model.add(Conv2D(
            **default_args,
            filters = filters
        ))

    model.add(Flatten())
    model.add(Dense(out_shape, activation=out_activation) )
    model.summary()
    
    return model


def build_deconv_net(
    latent_dim: int = 512,
    n_upsampling_layers = 4,
    filters = 128,
    activation_out = 'sigmoid'
    ) -> tf.keras.Sequential:
    """
    Build a deconvolutional network for decoding/upscaling latent vectors
    """

    model = tf.keras.Sequential()
    model.add(Dense(4 * 4 * 64, input_dim=latent_dim)) 
    model.add(Reshape((4, 4, 64))) # This matches the output size of the downsampling architecture
    default_args = dict(
        kernel_size = (3, 3),
        strides = (2, 2),
        padding = 'same',
        activation = 'relu'
    )
    
    for _ in range(n_upsampling_layers):
        model.add(Conv2DTranspose(**default_args, filters=filters))

    # This last convolutional layer converts back to 3 channel RGB image
    model.add(Conv2D(filters=3, kernel_size=(3,3), activation=activation_out, padding='same'))
    model.summary()
    
    return model

# %% [markdown]
# --- 
# ---
# 
# 
# ## 2. 2. Variational Autoencoders (VAEs)
# 

# %%
class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder
    It takes two vectors as input - one for means and other for variances of the latent variables described by a multimodal gaussian
    Its output is a latent vector randomly sampled from this distribution
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon

def build_vae(data_shape, latent_dim, filters=128):

    # Building the encoder - starts with a simple downsampling convolutional network  
    encoder = build_conv_net(data_shape, latent_dim*2, filters=filters)
        
    
    # Adding special sampling layer that uses the reparametrization trick 
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var])
    
    # Connecting the two encoder parts
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Defining the decoder which is a regular upsampling deconvolutional network
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))
    
    # Adding the special loss term
    kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
    vae.add_loss(kl_loss/tf.cast(tf.keras.backend.prod(data_shape), tf.float32))

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae


# %%
# Training the VAE model saving the interpolated vectors
epochs = 10
latent_dim = 32
encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, filters=128)
all_vectors = np.zeros((10,9,32))

# Generate random vectors that we will use to sample our latent space. The output is a grid image of the normal latent vectors.
for epoch in range(epochs):

    latent_vectors = np.random.randn(9, latent_dim)
    vae.fit(x=dataset, y=dataset, epochs=1, batch_size=8, verbose=0)
    ip_vectors = np.zeros((len(latent_vectors), latent_dim))
    
    for i in range(9):
        if i != 8:
            ip_vectors[i][:] = (latent_vectors[i] + latent_vectors[i + 1]) / 2
        else:
            ip_vectors[i][:] = (latent_vectors[i] + latent_vectors[0]) / 2
    
    all_vectors[epoch] = ip_vectors
 
    images = decoder(latent_vectors)
    grid_plot(images, epoch, name='VAE generated images interpolated', n=3, save=False)

# %% [markdown]
# Display the different interpolated images, these are saved during fitting and decoded after the fitting has ended. 

# %%
for i in range(all_vectors.shape[0]):
    images = decoder(all_vectors[i])
    grid_plot(images, i, name='VAE generated images interpolated', n=3, save=False)


# %% [markdown]
# ---
# 
# ## 2.3 Generative Adversarial Networks (GANs)
# 
# 
# 

# %%
def build_gan(
    data_shape,
    latent_dim,
    filters = 128,
    lr = 0.0002,
    beta_1 = 0.5
    ):
    optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta_1)

    # Usually thew GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, activation_out='tanh', filters=filters)
    
    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1, filters=filters) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN = tf.keras.models.Sequential([generator, discriminator])
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator, generator, GAN

# %% [markdown]
# ### Definining custom functions for training your GANs
# 
# ---
# 
# 
# 

# %%
def run_generator(generator, n_samples=100):
    """
    Run the generator model and generate n samples of synthetic images using random latent vectors
    """
    latent_dim = generator.layers[0].input_shape[-1]
    generator_input = np.random.randn(n_samples, latent_dim)

    return generator.predict(generator_input)

def get_batch(generator, dataset, batch_size=64):
    """
    Gets a single batch of samples (X) and labels (y) for the training the discriminator.
    One half from the real dataset (labeled as 1s), the other created by the generator model (labeled as 0s).
    """
    batch_size //= 2 # Split evenly among fake and real samples

    fake_data = run_generator(generator, n_samples=batch_size)
    real_data = dataset[np.random.randint(0, dataset.shape[0], batch_size)]

    X = np.concatenate([fake_data, real_data], axis=0)
    y = np.concatenate([np.zeros([batch_size, 1]), np.ones([batch_size, 1])], axis=0)

    return X, y

def train_gan_interpolation(generator, discriminator, gan, dataset, latent_dim, n_epochs=10, batch_size=64):

    df = pd.DataFrame(columns=['discriminator_loss', 'generator_loss'])
    batches_per_epoch = int(dataset.shape[0] / batch_size / 2)
    all_noises = np.empty((n_epochs, 16, 256))
    for epoch in range(n_epochs):
        d_loss, g_loss = np.zeros(batches_per_epoch), np.zeros(batches_per_epoch)
        for batch in range(batches_per_epoch):
            
            # 1) Train discriminator both on real and synthesized images
            X, y = get_batch(generator, dataset, batch_size=batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y)
            d_loss[batch] = discriminator_loss

            # 2) Train generator 
            X_gan = np.random.randn(batch_size, latent_dim)
            y_gan = np.ones([batch_size, 1])
            generator_loss = gan.train_on_batch(X_gan, y_gan)
            g_loss[batch] = generator_loss

        mean_d_loss, mean_g_loss = np.mean(d_loss), np.mean(g_loss)
        print(mean_d_loss, mean_g_loss)
        df.loc[epoch] = [mean_d_loss, mean_g_loss]
        noise = np.random.randn(16, latent_dim)
        it_noises = np.zeros((len(noise), latent_dim))

        for i in range(16):
            if i != 15:
                it_noises[i][:] = (noise[i] + noise[i + 1]) / 2
            else:
                it_noises[i][:] = (noise[i] + noise[0]) / 2
        
        all_noises[epoch] = it_noises
        images = generator.predict(noise)
        grid_plot(images, epoch, name='GAN generated images', n=3, save=False, scale=True)

    return df, all_noises, generator

# %%
## Build and train the model (need around 20 epochs to start seeing some results)

latent_dim = 256
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = dataset * 2 - 1  # Scale to [-1, 1] for tanh activation function

loss_df_i, all_noises, gen = train_gan_interpolation(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=20)

# %% [markdown]
# Plot the loss of the generator and discriminator. 

# %%
fig, ax = plt.subplots()

ax.plot(loss_df_i['discriminator_loss'], label='Discriminator')
ax.plot(loss_df_i['generator_loss'], label='Generator')
ax.set_title('GAN loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.tight_layout()
fig.savefig(DIRS['plots'] + 'gan_loss.png')

# %% [markdown]
# Plot the interpolated vectors, using the same technique as before.
# 

# %%
for i in range(20):
    images =  generator.predict(all_noises[i])
    grid_plot(images, i, name='GAN generated images interpolated', n=3, save=False, scale=True)



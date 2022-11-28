# %% [markdown]
# # Task 1: Learn the basics of Keras API for TensorFlow

# %% [markdown]
# Start with reading the section “Implementing MLPs with Keras” from _Chapter 10 of Geron’s text-book (pages 292-325)_.
# Then install `TensorFlow 2.0+` and experiment with the code included in this section.
# Additionally, study the official documentation (https://keras.io/) and get an idea of the numerous options offered by Keras (layers, loss functions, metrics, optimizers, activations, initializers, regularizers).
# Don’t get overwhelmed with the number of options – you will frequently return to this site in the coming months.

# %% [markdown]
# ### Imports

# %%
# stdlib
import os
from itertools import product
from time import perf_counter
from typing import Callable

# pip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.datasets import fashion_mnist, cifar10

# local
from utils import get_dirs

# %%
DIRS = get_dirs(os.path.abspath('') + os.sep + 'Task1.ipynb')
print('\033[1m' + 'Directories:' + '\033[0m')
for dir_name, path in DIRS.items():
    print(f'{dir_name:<7} {path}')

# %% [markdown]
# ---
# ## Part 1

# %% [markdown]
# See the report

# %% [markdown]
# ---
# 
# ## Part 2

# %% [markdown]
# We will first test multiple sets of hyperparameters on the MNIST dataset using the MLP and the CNN model from the book. We will extract the top 3 best performing sets from this experiment and use them to train a MLP and a CNN on the CIFAR-10 dataset, to have a look how well these hyperparameters generalize to a different dataset.

# %% [markdown]
# Firstly, load FMNIST dataset

# %%
(X_train_f, y_train_f), (X_test_f, y_test_f) = fashion_mnist.load_data()

X_train_f = X_train_f.astype('float32') / 255
X_test_f = X_test_f.astype('float32') / 255

X_train_f.shape, X_test_f.shape

# %% [markdown]
# Define functions for running the hyperparameter exploration experiments

# %%
def build_default_MLP(input_shape, activation, optimizer, lr) -> Sequential:
    """
    Returns a compiled default MLP classifier architecture with
    a given input shape, activation function, optimizer and learning rate.
    """
    model = Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(300, activation=activation),
        layers.Dense(100, activation=activation),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer = optimizer(learning_rate=lr),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

# %%
def build_default_CNN(input_shape, activation, optimizer, lr) -> Sequential:
    """
    Returns a compiled default CNN classifier architecture with
    a given input shape, activation function, optimizer and learning rate.
    """
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation=activation, input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation),
        layers.Flatten(),
        layers.Dense(64, activation=activation),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer = optimizer(learning_rate=lr),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

# %%
def run_experiment(model_constructor: Callable, datasets: tuple, configs: list) -> pd.DataFrame:
    """
    Params:
        model_constructor (function) - build_default_MLP or build_default_CNN
        datasets (tuple) - (X_train, y_train, X_test, y_test)
        configs (list) - list of tuples of (optimizer, lr, activation)
    """
    np.random.seed(42)
    X_train, y_train, X_test, y_test = datasets
    if 'CNN' in model_constructor.__name__:
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
    
    df = pd.DataFrame(columns=['optimizer', 'lr', 'activation', 'loss', 'accuracy', 'traintime'])
    run = 1
    for optimizer, lr, activation in configs:
        losses, accuracies, traintimes = [], [], []
        for _ in range(3):
            print(f'\r{run}/{len(configs)*3}', end='')
            
            model = model_constructor(
                input_shape = X_train.shape[1:],
                activation = activation,
                optimizer = optimizer,
                lr = lr
            )
            
            tic = perf_counter()
            model.fit(
                x = X_train,
                y = y_train,
                epochs = 5,
                batch_size = 64,
                verbose = 0
            )
            toc = perf_counter()

            test_loss, test_acc = model.evaluate(
                x = X_test,
                y = y_test,
                verbose = 0
            )
            
            losses.append(test_loss)
            accuracies.append(test_acc)
            traintimes.append(toc-tic)
            run += 1

        df.loc[f'{optimizer.__name__}-{activation}-{lr}'] = [
            optimizer.__name__,
            lr,
            activation,
            np.mean(losses),
            np.mean(accuracies),
            np.mean(traintimes)
        ]
    return df

# %%
def create_plots(df, optimizers, activations, lrs, title) -> plt.Figure:
    """
    Creates a 3x3 grid of plots for the given optimizers, activations and learning rates.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, optimizer in enumerate([opt.__name__ for opt in optimizers]):
        for j, metric in enumerate(['accuracy', 'loss', 'traintime']):
            ax = axes[i,j]
            for activation in activations:
                df[(df.optimizer == optimizer) & (df.activation == activation)].plot(
                    x = 'lr',
                    y = metric,
                    ax = ax,
                    label = activation
                )
            ax.set_xlabel('')
            ax.set_ylabel(optimizer) if j == 0 else ax.set_ylabel('')
            ax.set_xticks(lrs, fontsize=3)
            ax.get_legend().remove()
            ax.set_title(metric) if i == 0 else ax.set_title('')

    # set global legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=3, fontsize=14)
    fig.suptitle(title, fontsize=20, weight='bold')
    fig.tight_layout()
    return fig

# %% [markdown]
# Test different hyperparameters for a 2-hidden-layer MLP as defined in chapter 10 of the book (FMNIST dataset)

# %%
optimizers = [keras.optimizers.Adam, keras.optimizers.SGD, keras.optimizers.RMSprop]
lrs = [1e-3, 5e-3, 1e-2]
activations = ['relu', 'sigmoid', 'tanh']
configs = list(product(optimizers, lrs, activations))

df_mf = run_experiment(build_default_MLP, (X_train_f, y_train_f, X_test_f, y_test_f), configs)
df_mf.to_csv(DIRS['csv'] + 'mlp_fmnist.csv', index=False)
fig_mf = create_plots(df_mf, optimizers, activations, lrs, 'Fashion MNIST MLP')
fig_mf.savefig(DIRS['plots'] + 'mlp_fmnist.png', dpi=300)

# %% [markdown]
# Do the same for a 3-hidden-layer CNN as defined in chapter 14 of the book with some modifications to save on runtime (FMNIST dataset)

# %%
# we use the same configs list as for the MLP

df_cf = run_experiment(build_default_CNN, (X_train_f, y_train_f, X_test_f, y_test_f), configs)
df_cf.to_csv(DIRS['csv'] + 'cnn_fmnist.csv', index=False)
fig_cf = create_plots(df_cf, optimizers, activations, lrs, 'Fashion MNIST CNN')
fig_cf.savefig(DIRS['plots'] + 'cnn_fmnist.png', dpi=300)

# %% [markdown]
# ---

# %% [markdown]
# ### CIFAR-10

# %% [markdown]
# We now look at the 3 best performing hyperparameter sets that are described in the tables in the report and train a MLP and a CNN (the same models as used before) on the CIFAR-10 dataset.

# %% [markdown]
# Remove the FMNIST dataset from memory if needed

# %%
try:
    del X_train_f, y_train_f, X_test_f, y_test_f
except NameError:
    pass

# %% [markdown]
# Load in the CIFAR-10 dataset.

# %%
(X_train_c, y_train_c), (X_test_c, y_test_c) = cifar10.load_data()

X_train_c = X_train_c.astype('float32') / 255
X_test_c = X_test_c.astype('float32') / 255

X_train_c.shape, X_test_c.shape

# %% [markdown]
# Define the models for the 3 best performing hyperparameter sets.

# %%
MLPmodel1 = build_default_MLP(X_train_c.shape[1:], 'sigmoid', keras.optimizers.Adam, 0.005)
MLPmodel2 = build_default_MLP(X_train_c.shape[1:], 'relu', keras.optimizers.RMSprop, 0.001)
MLPmodel3 = build_default_MLP(X_train_c.shape[1:], 'tanh', keras.optimizers.SGD, 0.01)

CNNmodel1 = build_default_CNN(X_train_c.shape[1:], 'relu', keras.optimizers.RMSprop, 0.001)
CNNmodel2 = build_default_CNN(X_train_c.shape[1:], 'tanh', keras.optimizers.Adam, 0.001)
CNNmodel3 = build_default_CNN(X_train_c.shape[1:], 'tanh', keras.optimizers.SGD, 0.01)

MLPmodels = [MLPmodel1, MLPmodel2, MLPmodel3]
CNNmodels = [CNNmodel1, CNNmodel2, CNNmodel3]

# %% [markdown]
# Train the models and save the results.

# %%
def trainModel(model):

    for x in range(5):
        losses, accuracies, traintimes = [], [], []
        historyarr = []
        tic = perf_counter()
        history = model.fit(
            x = X_train_c,
            y = y_train_c,
            epochs = 5,
            batch_size = 64,
            verbose = 0
        )
        toc = perf_counter()

        test_loss, test_acc = model.evaluate(
            x = X_test_c,
            y = y_test_c,
            verbose = 1
        )

        losses.append(test_loss)
        accuracies.append(test_acc)
        traintimes.append(toc-tic)
        historyarr.append(history)

        return historyarr, np.mean(losses), np.mean(accuracies), np.mean(traintimes)
        
        

# %% [markdown]
# Run the models

# %%
historydictMLP = {}
for model in MLPmodels:

    history = trainModel(model)
    historydictMLP[model] = history

historydictCNN = {}
for model in CNNmodels:

    history = trainModel(model)
    historydictCNN[model] = history

# %% [markdown]
# Plotting all the results

# %%
fig, ax = plt.subplots(3, 2, figsize=(10, 10))

#add the dictioarys together
for i, model in enumerate(historydictMLP):

    ax[i,0].plot(historydictMLP[model][0][0].history['loss'], label='loss')
    twax = ax[i,0].twinx()
    twax.plot(historydictMLP[model][0][0].history['accuracy'], color='orange', label='accuracy')
    ax[i,0].set_title('MLP model ' + str(i+1))
    ax[i,0].set_xlabel('Epoch')
    ax[i,0].set_xticks(np.arange(0, 5, 1))
    ax[i,0].set_ylabel('Loss')
    twax.set_ylabel('Accuracy')
    ax[i,0].legend(loc='upper left')
    twax.legend(loc='upper right')

for i, model in enumerate(historydictCNN):

    ax[i,1].plot(historydictCNN[model][0][0].history['loss'], label='loss')
    twax = ax[i,1].twinx()
    twax.plot(historydictCNN[model][0][0].history['accuracy'], color='orange', label='accuracy')
    ax[i,1].set_title('CNN model ' + str(i+1))
    ax[i,1].set_xlabel('Epoch')
    ax[i,1].set_xticks(np.arange(0, 5, 1), np.arange(0, 5, 1))
    ax[i,1].set_ylabel('Loss')
    ax[i,1].set_yticks(np.arange(0, 2, 0.5), np.arange(0, 2, 0.5))
    twax.set_ylabel('Accuracy')
    ax[i,1].legend(loc='upper left')
    twax.legend(loc='upper right')

fig.tight_layout()
fig.savefig(DIRS['plots'] + 'cifar10.png', dpi=500)



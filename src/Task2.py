# %% [markdown]
# # Task 2: Develop a "Tell-the-time" network

# %% [markdown]
# ### Setup

# %% [markdown]
# Imports

# %%
# stdlib
import os
from functools import partial

# pip
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, Model
from keras import layers
from keras import backend as K

# local
from utils import get_dirs, train_test_split, ProgressBar

# %% [markdown]
# Fix directories

# %%
DIRS = get_dirs(os.path.abspath('') + os.sep + 'Task2.ipynb')
print('\033[1m' + 'Directories:' + '\033[0m')
for dir_name, path in DIRS.items():
    print(f'{dir_name:<7} {path}')

# %% [markdown]
# Load and split data

# %%
images = np.load(DIRS['data'] + 'images.npy')
labels = np.load(DIRS['data'] + 'labels.npy')

images = images.astype('float32') / 255
labels = labels.astype('int32')

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

del images, labels

print('\033[1m' + 'Data:' + '\033[0m')
print('  name  |        shape      | dtype')
print('--------+-------------------+-------')
for name, arr in zip(['X_train', 'X_test', 'y_train', 'y_test'], [X_train, X_test, y_train, y_test]):
    print(f'{name:<7} | {str(arr.shape):<17} | {arr.dtype}')

# %% [markdown]
# Confirm that the data is sufficiently shuffled

# %%
size_tr, size_te = len(y_train) // 5, len(y_test) // 5
print('\033[1m' + 'Average labels:' + '\033[0m')
print('train:', '\t'.join([f'{np.mean(y_train[i*size_tr:(i+1)*size_tr]):.3f}' for i in range(5)]))
print('test: ', '\t'.join([f'{np.mean(y_test[i*size_te:(i+1)*size_te]):.3f}' for i in range(5)]))

# %% [markdown]
# Define functions to get specific versions of the labels

# %%
def get_regression_labels(y_train, y_test):
    """
    Hours and minutes are merged into a single float value:
        * (1, 30) -> 1.5
        * (11, 15) -> 11.25
    """
    reg_y_train = (y_train[:, 0] + y_train[:, 1] / 60).astype('float32')
    reg_y_test  = (y_test[:, 0] + y_test[:, 1] / 60).astype('float32')
    return reg_y_train, reg_y_test

def get_class_24_labels(y_train, y_test):
    """
    12 hours are split into 24 half-hour bins:
        * (1, 30) -> 3 (one-hot vector)
        * (11, 15) -> 22 (one-hot vector)
    """
    class_y_train = (y_train[:, 0] * 2 + y_train[:, 1] // 30).astype('float32')
    class_y_test  = (y_test[:, 0] * 2 + y_test[:, 1] // 30).astype('float32')
    class_y_train = keras.utils.to_categorical(class_y_train, num_classes=24)
    class_y_test = keras.utils.to_categorical(class_y_test, num_classes=24)
    return class_y_train, class_y_test

def get_class_48_labels(y_train, y_test):
    """
    12 hours are split into 48 quarter-hour bins:
        * (1, 30) -> 6 (one-hot vector)
        * (11, 15) -> 45 (one-hot vector)
    """
    class_y_train = (y_train[:, 0] * 4 + y_train[:, 1] // 15).astype('float32')
    class_y_test  = (y_test[:, 0] * 4 + y_test[:, 1] // 15).astype('float32')
    class_y_train = keras.utils.to_categorical(class_y_train, num_classes=48)
    class_y_test = keras.utils.to_categorical(class_y_test, num_classes=48)
    return class_y_train, class_y_test

def get_multiclass_labels(y_train, y_test):
    """
    Hours are returned as one-hot vectors and minutes are binned into 12 classes.
    This results in 12 * 12 = 144 classes.
    """
    hours_train = keras.utils.to_categorical(y_train[:, 0], num_classes=12)
    hours_test  = keras.utils.to_categorical(y_test[:, 0], num_classes=12)
    minutes_train = keras.utils.to_categorical(y_train[:, 1] // 5, num_classes=12)
    minutes_test  = keras.utils.to_categorical(y_test[:, 1] // 5, num_classes=12)
    class_y_train = np.array((hours_train, minutes_train))
    class_y_test  = np.array((hours_test, minutes_test))
    return class_y_train, class_y_test

# %% [markdown]
# Define custom loss functions where 0:00 and 11:55 are just 5 minutes apart

# %%
def cs_loss_1(y_true, y_pred):
    """Basic form of common sense loss"""
    return K.abs(K.minimum(K.abs(y_true - y_pred), K.ones_like(y_true) * 12 - K.abs(y_true - y_pred)))

def cs_loss_2(y_true, y_pred):
    """Common sense loss with fixed penalty for predictions <0"""
    # punish negative predictions
    penalty = 12 * K.sum(K.cast(K.less(y_pred, 0), 'float32'))
    loss = K.abs(K.minimum(K.abs(y_true - y_pred), K.ones_like(y_true) * 12 - K.abs(y_true - y_pred)))
    return loss + penalty

def cs_loss_3(y_true, y_pred):
    """Common sense loss with scaled penalty for predictions <0 and >12"""
    # punish negative predictions and predictions greater than 12
    penalty = K.maximum(0.0, -y_pred) + K.maximum(0.0, y_pred - 12)
    loss = K.minimum(K.abs(y_true - y_pred), K.ones_like(y_true) * 12 - K.abs(y_true - y_pred))
    return loss + penalty

# %%
def test_loss_fn(loss_fn):
    tests = [[1.0, 1.0], [1.5, 1.25], [11.75, 0.25], [2.0, -10.0], [1.0, -1.0], [3.0, 15.0]]
    print('\033[1m' + f'{loss_fn.__name__}:' + '\033[0m')
    print(' true | pred  | loss')
    print('------+-------+-----')
    for test in tests:
        print(f'{test[0]:<5} | {test[1]:<5} | {loss_fn(test[0], test[1])}')

for loss_fn in [cs_loss_1, cs_loss_2, cs_loss_3]:
    test_loss_fn(loss_fn)

# %% [markdown]
# Define custom activation function that maps -1 to 11, 13 to 1, etc.

# %%
def custom_activation(x):
    x = K.switch(K.less(x, 0), 12 + x, x)
    x = K.switch(K.greater(x, 12), x - 12, x)
    return x

for x in [1, 13, -1, 0, -10]:
    print(f'{x:^3} -> {custom_activation(x)}')

# %% [markdown]
# ---

# %% [markdown]
# ### Heavy functions

# %% [markdown]
# Define function that can build models for all of the different approaches to the problem

# %%
def build_model(problem_type: str) -> Model:
    """
    Build and compile a model for the given problem type.
    
    Options:
        * 'nrg' | naive regression
        * 'cs1' | common sense regression with cs_loss_1
        * 'cs2' | common sense regression with cs_loss_2
        * 'cs3' | common sense regression with cs_loss_3
        * 'cs4' | common sense regression with cs_loss_1 & custom_activation
        * 'c24' | classification into 24 half-hour bins
        * 'c48' | classification into 48 quarter-hour bins
        * 'mhc' | multi-head classification: 12 hour bins, 12 (five) minute bins
    """
    
    if problem_type == 'nrg':
        final_layer = layers.Dense(1)
        loss, metrics = 'mse', ['mae']
    elif problem_type in ['cs1', 'cs2', 'cs3']:
        final_layer = layers.Dense(1)
        loss = {'cs1': cs_loss_1, 'cs2': cs_loss_2, 'cs3': cs_loss_3}[problem_type]
        metrics = ['mae']
    elif problem_type == 'cs4':
        final_layer = layers.Dense(1, activation=custom_activation)
        loss = cs_loss_1
        metrics = ['mae']
    elif problem_type in ['c24', 'c48']:
        final_layer = layers.Dense(24 if problem_type == 'c24' else 48, activation='softmax')
        loss, metrics = 'categorical_crossentropy', ['accuracy']
    elif problem_type == 'mhc':
        final_layer = layers.Flatten()  # dummy, because we will add two heads later on
        loss, metrics = ['categorical_crossentropy', 'categorical_crossentropy'], ['accuracy']
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

    DefaultConv2D = partial(layers.Conv2D, kernel_initializer='he_normal', kernel_size=3, activation='elu', padding='SAME')
    
    model = Sequential()
    model.add(DefaultConv2D(filters=32, kernel_size=5, strides=(3, 3), input_shape=(150, 150, 1)))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=64))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=128))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(final_layer)

    if problem_type == 'mhc':
        output_hrs = layers.Dense(12, activation='softmax', name='hrs')(model.output)
        output_min = layers.Dense(12, activation='softmax', name='min')(model.output)
        model = Model(inputs=model.input, outputs=[output_hrs, output_min])

    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    
    return model

# %% [markdown]
# Define a function that generates one batch of data at a time to save on RAM

# %%
def batch_generator(X_train, X_test, y_train, y_test, multiclass: bool = False, folds: int = 5):
    """
    Generator function to yield data in batches that divide all data into n batches.
    If multiclass is True, we first need to split y into y_hrs and y_min, because y.shape[0] = 2
    """
    batch_size_train = X_train.shape[0] // folds
    batch_size_test = X_test.shape[0] // folds
    while True:
        for i in range(folds):
            _X_train = X_train[i*batch_size_train:(i+1)*batch_size_train]
            _X_test = X_test[i*batch_size_test:(i+1)*batch_size_test]
            if multiclass:
                _y_train = [y_train[0][i*batch_size_train:(i+1)*batch_size_train], y_train[1][i*batch_size_train:(i+1)*batch_size_train]]
                _y_test = [y_test[0][i*batch_size_test:(i+1)*batch_size_test], y_test[1][i*batch_size_test:(i+1)*batch_size_test]]
            else:
                _y_train = y_train[i*batch_size_train:(i+1)*batch_size_train]
                _y_test = y_test[i*batch_size_test:(i+1)*batch_size_test]
            yield _X_train, _y_train, _X_test, _y_test

# %% [markdown]
# The "main" function

# %%
def run_experiment(
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    passes: int = 5,
    epochs: int = 10,
    save_model: bool = False,
    prog_bar: bool = False,
    ):
    """
    Run an experiment for the given problem type.
    
    Options for problem_type:
        * 'nrg' | naive regression
        * 'cs1' | common sense regression with cs_loss_1
        * 'cs2' | common sense regression with cs_loss_2
        * 'cs3' | common sense regression with cs_loss_3
        * 'cs4' | common sense regression with cs_loss_1 & custom_activation
        * 'c24' | classification into 24 half-hour bins
        * 'c48' | classification into 48 quarter-hour bins
        * 'mhc' | multi-head classification: 12 hour bins, 12 (five) minute bins

    Returns:
        * loss history (train)
        * metric history (train)
        * test results

    For the multi-head model, five dataframes are returned.

    If save_model is True, the model is saved to a `<problem_type>.h5` file.
    """
    
    if problem_type in ['nrg', 'cs1', 'cs2', 'cs3', 'cs4']:
        y_train, y_test = get_regression_labels(y_train, y_test)
    elif problem_type == 'c24':
        y_train, y_test = get_class_24_labels(y_train, y_test)
    elif problem_type == 'c48':
        y_train, y_test = get_class_48_labels(y_train, y_test)
    elif problem_type == 'mhc':
        y_train, y_test = get_multiclass_labels(y_train, y_test)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

    if problem_type == 'mhc':
        # df_test will have each row representing a fold, with the columns being loss and metric
        df_test = pd.DataFrame(columns=['total_loss', 'hrs_loss', 'min_loss', 'hrs_acc', 'min_acc'])
        # df_train_... will be a list with each row representing an epoch, and each column representing a fold
        df_train_loss_hrs = pd.DataFrame()
        df_train_metric_hrs = pd.DataFrame()
        df_train_loss_min = pd.DataFrame()
        df_train_metric_min = pd.DataFrame()
    else:
        df_test = pd.DataFrame(columns=['loss', 'metric'])
        df_train_loss = pd.DataFrame()
        df_train_metric = pd.DataFrame()

    model = build_model(problem_type)
    folds = 5

    get_batch = batch_generator(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        multiclass = True if problem_type=='mhc' else False,
        folds = folds
    )

    progress_bar = ProgressBar(passes*folds, problem_type) if prog_bar else None
    tf_verbose = 0 if prog_bar else 1

    for p in range(passes):
        if not prog_bar: print('\n' + '\033[1m' + '-' * 50 + '\n' + f'Pass {p+1}/{passes}' + '\033[0m')
        for fold in range(folds):
            if not prog_bar: print('\n---\n' + '\033[1m' + f'Fold {fold+1}/{folds}' + '\033[0m')
            _X_train, _y_train, _X_test, _y_test = next(get_batch)

            # train the model
            if not prog_bar: print('\033[1m' + 'Train' + '\033[0m')
            history = model.fit(_X_train, _y_train, epochs=epochs, verbose=tf_verbose)

            # evaluate the model
            if not prog_bar: print('\033[1m' + 'Test' + '\033[0m')
            results = model.evaluate(_X_test, _y_test, verbose=tf_verbose)

            # save the results
            df_test.loc[p*folds + fold] = results
            if problem_type != 'mhc':
                df_train_loss[p*folds + fold] = history.history['loss']
                df_train_metric[p*folds + fold] = history.history[list(history.history.keys())[1]]
            else:
                df_train_loss_hrs[p*folds + fold] = history.history['hrs_loss']
                df_train_metric_hrs[p*folds + fold] = history.history['hrs_accuracy']
                df_train_loss_min[p*folds + fold] = history.history['min_loss']
                df_train_metric_min[p*folds + fold] = history.history['min_accuracy']
            
            if prog_bar: progress_bar(p*folds + fold)

        # save the model after each pass
        if save_model:
            model.save(DIRS['models'] + f'{problem_type}.h5')
    del model

    if problem_type == 'mhc':
        # df_train_metric will be empty
        return df_test, df_train_loss_hrs, df_train_metric_hrs, df_train_loss_min, df_train_metric_min
    return df_test, df_train_loss, df_train_metric

# %% [markdown]
# ---

# %% [markdown]
# ### Running the experiments

# %%
default_experiment = partial(
    run_experiment,
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    passes = 5,
    epochs = 10,
    save_model = True,
    prog_bar = True
)

# %% [markdown]
# First we run the naive regression model

# %%
target = DIRS['csv'] + os.sep + 'naive' + os.sep
if not os.path.exists(target):
    os.makedirs(target)

df_test_nrg, df_train_loss_nrg, df_train_metric_nrg = default_experiment(problem_type='nrg')
for df, name in zip(
        [df_test_nrg, df_train_loss_nrg, df_train_metric_nrg],
        ['test', 'train_loss', 'train_metric']
    ):
    df.to_csv(target + f'nrg_{name}.csv')

# %% [markdown]
# Then we run the common sense regression models

# %%
target = DIRS['csv'] + os.sep + 'commonsense' + os.sep
if not os.path.exists(target):
    os.makedirs(target)
for i in range(1, 5):
    df_test_cs, df_train_loss_cs, df_train_metric_cs = default_experiment(problem_type=f'cs{i}')
    for df, name in zip(
            [df_test_cs, df_train_loss_cs, df_train_metric_cs],
            ['test', 'train_loss', 'train_metric']
        ):
        df.to_csv(target + f'cs{i}_{name}.csv')

# %% [markdown]
# Now we move on to the classifiers

# %%
target = DIRS['csv'] + os.sep + 'classify' + os.sep
if not os.path.exists(target):
    os.makedirs(target)
for i in [24, 48]:
    df_test_c, df_train_loss_c, df_train_metric_c = default_experiment(problem_type=f'c{i}')
    for df, name in zip(
            [df_test_c, df_train_loss_c, df_train_metric_c],
            ['test', 'train_loss', 'train_metric']
        ):
        df.to_csv(target + f'c{i}_{name}.csv')

# %% [markdown]
# last but not least, we run the multi-head classifier

# %%
target = DIRS['csv'] + os.sep + 'multihead' + os.sep
if not os.path.exists(target):
    os.makedirs(target)
df_test_mhc, df_train_loss_hrs_mhc, df_train_metric_hrs_mhc, df_train_loss_min_mhc, df_train_metric_min_mhc = default_experiment(problem_type='mhc')
for df, name in zip(
        [df_test_mhc, df_train_loss_hrs_mhc, df_train_metric_hrs_mhc, df_train_loss_min_mhc, df_train_metric_min_mhc],
        ['test', 'train_loss_hrs', 'train_metric_hrs', 'train_loss_min', 'train_metric_min']
    ):
    df.to_csv(target + f'mhc_{name}.csv')

# %% [markdown]
# ---

# %% [markdown]
# ### Regression plots

# %% [markdown]
# Define functions that load a trained model and return the predictions for the test set as a dataframe

# %%
def get_reg_predictions(
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ) -> tuple[pd.DataFrame, float]:
    """Get predictions and calculate MAE for the test set."""
    
    custom_objects = None
    if model_name in ['cs1', 'cs2', 'cs3']:
        custom_objects = {f'cs_loss_{model_name[2]}': {'cs1': cs_loss_1, 'cs2': cs_loss_2, 'cs3': cs_loss_3}[model_name]}
    elif model_name == 'cs4':
        custom_objects = {'cs_loss_1': cs_loss_1, 'custom_activation': custom_activation}
    
    model = keras.models.load_model(DIRS['models'] + model_name + '.h5', custom_objects=custom_objects)
    df = pd.DataFrame(columns=['true', 'pred'])
    y_pred = model.predict(X_test, verbose=0)
    df['true'] = y_test
    df['pred'] = y_pred
    mae = df['true'].sub(df['pred']).abs().mean()
    del model
    return df, mae

# %%
def get_class_predictions(
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ) -> tuple[pd.DataFrame, float]:
    """Get predictions and calculate accuracy for the test set."""
    
    model = keras.models.load_model(DIRS['models'] + model_name + '.h5')
    y_pred = model.predict(X_test, verbose=0)
    if model_name != 'mhc':
        df = pd.DataFrame(columns=['true', 'pred'])
        df['true'], df['pred'] = y_test.argmax(axis=1), y_pred.argmax(axis=1)
        acc = (df['true'] == df['pred']).mean()
    else:
        df = pd.DataFrame(columns=['true_hrs', 'pred_hrs', 'true_min', 'pred_min'])
        df['true_hrs'], df['pred_hrs'] = y_test[0].argmax(axis=1), y_pred[0].argmax(axis=1)
        df['true_min'], df['pred_min'] = y_test[1].argmax(axis=1), y_pred[1].argmax(axis=1)
        acc = (
            (df['true_hrs'] == df['pred_hrs']).mean()
            + (df['true_min'] == df['pred_min']).mean()
        ) / 2
    del model
    return df, acc

# %% [markdown]
# Define function that clusters the prediction data and fits one or two regression models

# %%
def get_clusters(df: pd.DataFrame, method: str = 'km') -> tuple[list[pd.DataFrame], list[float]]:
    """
    Divide the dataframe into clusters and also add regression data to the returned dataframes.
    Options for method: 'kmeans' (with k=2), 'dbscan'.
    """
    df = df.copy()
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(df)
        df['cluster'] = kmeans.labels_
        # calculate linear regression for both clusters separately
        df1 = df[df['cluster'] == 0].copy()
        df2 = df[df['cluster'] == 1].copy()
        lr1 = LinearRegression().fit(df1['true'].values.reshape(-1, 1), df1['pred'].values.reshape(-1, 1))
        lr2 = LinearRegression().fit(df2['true'].values.reshape(-1, 1), df2['pred'].values.reshape(-1, 1))
        # predict
        df1['reg'] = lr1.predict(df1['true'].values.reshape(-1, 1))
        df2['reg'] = lr2.predict(df2['true'].values.reshape(-1, 1))
        return [df1, df2], [lr1.coef_[0][0], lr2.coef_[0][0]]
    elif method == 'dbscan':
        dbscan = DBSCAN()
        dbscan.fit(df)
        df['cluster'] = dbscan.labels_
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        dfs = [df[df['cluster'] == i].copy() for i in range(n_clusters)]
        # biggest cluster
        dfs.sort(key=lambda x: len(x), reverse=True)
        main_cluster = dfs[0]
        # calculate linear regression for main cluster
        lr = LinearRegression().fit(main_cluster['true'].values.reshape(-1, 1), main_cluster['pred'].values.reshape(-1, 1))
        main_cluster['reg'] = lr.predict(main_cluster['true'].values.reshape(-1, 1))
        dfs[0] = main_cluster
        return dfs, [lr.coef_[0][0]]
    else:
        raise ValueError('Invalid method')

# %% [markdown]
# Define a function that creates an actual scatterplot for a regression model's predictions

# %%
def get_scatterplot(clusters: list[pd.DataFrame], slopes: list, title: str = None) -> plt.Figure:
    """Get a scatterplot with regression lines for each cluster."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, cluster in enumerate(clusters):
        color = 'tab:blue' if i == 0 else 'tab:green'
        ax.plot(cluster['true'], cluster['pred'], 'o', color=color, alpha=0.1, zorder=1)
        reg_color = 'tab:red' if i == 0 else 'tab:orange'
        try:  # only plot if we have a regression line for this cluster
            ax.plot(cluster['true'], cluster['reg'], color=reg_color, label=f'${slopes[i]:.2f}$', zorder=2)
        except KeyError:
            pass
    ax.plot([0, 12], [0, 12], '--', color='tab:gray', zorder=0)
    ax.hlines(0, 0, 12, color='tab:gray', zorder=0)
    ax.hlines(12, 0, 12, color='tab:gray', zorder=0)
    ax.set_xlabel('true')
    ax.set_ylabel('predicted')
    ax.legend(title='slope')
    if title is not None:
        ax.set_title(title, weight='bold')
    fig.tight_layout()
    return fig

# %% [markdown]
# First we want to visualize the naive regression predictions

# %%
_, reg_y_test = get_regression_labels(y_train, y_test)
del _
naive_pred, naive_mae = get_reg_predictions('nrg', X_test, reg_y_test)

x, y = naive_pred['true'], naive_pred['pred']
p = np.poly1d(np.polyfit(x, y, 3))

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, y, 'o', color='tab:blue', alpha=0.1, zorder=1)
ax.scatter(x, p(x), s=1, c='tab:red', zorder=2)
ax.plot([0, 12], [0, 12], '--', color='tab:gray', zorder=0)
ax.hlines(0, 0, 12, color='tab:gray', zorder=0)
ax.hlines(12, 0, 12, color='tab:gray', zorder=0)
ax.set_xlabel('true')
ax.set_ylabel('predicted')
ax.set_title(f'Naive Approach (mae = {naive_mae:.3f})', weight='bold')
fig.tight_layout()
fig.savefig(DIRS['plots']+ 'nrg_acc.png', dpi=500)

# %% [markdown]
# And now we also want to visualize how our different custom loss functions perform

# %%
_, reg_y_test = get_regression_labels(y_train, y_test)
del _
for model_name in ['cs1', 'cs2', 'cs3', 'cs4']:
    pred, mae = get_reg_predictions(model_name, X_test, reg_y_test)
    method = 'kmeans' if model_name == 'cs1' else 'dbscan'
    clusters, slopes = get_clusters(pred, method=method)
    fig = get_scatterplot(clusters, slopes, title=f'Common Sense Loss {model_name[2]} (mae = {mae:.3f})')
    fig.savefig(DIRS['plots'] + f'{model_name}_acc.png', dpi=500)

# %% [markdown]
# ---

# %% [markdown]
# ### Classification plots

# %% [markdown]
# Load classification predictions

# %%
_, class_24_y_test = get_class_24_labels(y_train, y_test)
pred_24, acc_24 = get_class_predictions('c24', X_test, class_24_y_test)
del _, class_24_y_test

_, class_48_y_test = get_class_48_labels(y_train, y_test)
pred_48, acc_48 = get_class_predictions('c48', X_test, class_48_y_test)
del _, class_48_y_test

_, mhc_y_test = get_multiclass_labels(y_train, y_test)
pred_mhc, acc_mhc = get_class_predictions('mhc', X_test, mhc_y_test)
del _, mhc_y_test

# %% [markdown]
# Visualize the simple classifier results

# %%
for pred, acc in zip([pred_24, pred_48], [acc_24, acc_48]):
    cm = confusion_matrix(pred['pred'], pred['true'])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap='Reds')
    
    r = cm.shape[0]
    ax.set_xticks(np.arange(0, r, r//12))
    ax.set_yticks(np.arange(0, r, r//12))
    ax.set_xticklabels([f'{i:02d}:00' for i in range(0, 12)])
    ax.set_yticklabels([f'{i:02d}:00' for i in range(0, 12)])
    plt.setp(ax.get_xticklabels(), rotation=-45, ha='left', rotation_mode='anchor')
    
    ax.set_xlabel('true')
    ax.set_ylabel('predicted')
    ax.set_title(f'{r} Classes (acc. = {acc*100:.1f}%)', weight='bold')
    fig.tight_layout()
    fig.savefig(DIRS['plots'] + f'c{r}_acc.png', dpi=500)

del pred_24, pred_48

# %% [markdown]
# And finally the multihead classifier results

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
cm_hrs = confusion_matrix(pred_mhc['pred_hrs'], pred_mhc['true_hrs'])
cm_min = confusion_matrix(pred_mhc['pred_min'], pred_mhc['true_min'])

ax1.imshow(cm_hrs, cmap='Reds')
ax1.set_xticks(np.arange(0, 12)); ax1.set_yticks(np.arange(0, 12))
ax1.set_title('Hours')

ax2.imshow(cm_min, cmap='Reds')
ax2.set_xticks(np.arange(0, 12)); ax2.set_yticks(np.arange(0, 12))
ax2.set_xticklabels(np.arange(0, 60, 5)); ax2.set_yticklabels(np.arange(0, 60, 5))
ax2.set_title('Minutes')
fig.supxlabel('true')
fig.supylabel('predicted')
fig.suptitle(f'Multihead (acc. = {acc_mhc*100:.1f}%)', weight='bold')
fig.tight_layout()
fig.savefig(DIRS['plots'] + 'mhc_acc.png', dpi=500)

del pred_mhc

# %% [markdown]
# ---

# %% [markdown]
# ### History plots

# %%
def load_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both train and test histories (loss+metrics) from csv files and returns it as two DataFrames."""
    train_loss = pd.read_csv(DIRS['csv'] + f'{model_name}_train_loss.csv', index_col=0)
    train_metric = pd.read_csv(DIRS['csv'] + f'{model_name}_train_metric.csv', index_col=0)
    df_test = pd.read_csv(DIRS['csv'] + f'{model_name}_test.csv', index_col=0)
    # train_loss and train_metric have n_folds*n_passes columns and n_epochs rows
    # we want to have n_folds*n_passes*n_epochs rows and 2 columns
    train_loss = train_loss.melt(var_name='leg', value_name='loss')
    train_metric = train_metric.melt(var_name='leg', value_name='metric')
    # merge the two DataFrames so we get loss, metric as columns, we don't care about the leg column
    df_train = pd.merge(train_loss, train_metric, left_index=True, right_index=True).drop(columns=['leg_x', 'leg_y'])
    return df_train, df_test

# %%
def get_history_plot(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    title: str,
    metric: str
    ) -> plt.Figure:
    """Returns a figure with the train and test loss and metric plots."""
    test_locs = [9 + i for i in range(0, 250, 10)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
    
    ax1.plot(df_train['loss'], 'o', alpha=0.25, zorder=1, color='tab:blue', label='train')
    ax1.plot(df_train['loss'].rolling(10).mean(), zorder=2, color='tab:blue', label='train (smoothed)')
    ax1.scatter(test_locs, df_test['loss'], marker='x', zorder=3, color='tab:red', label='test')
    ax1.set_ylabel('loss')

    ax2.plot(df_train['metric'], 'o', alpha=0.25, zorder=1, color='tab:blue', label='train')
    ax2.plot(df_train['metric'].rolling(10).mean(), zorder=2, color='tab:blue', label='train (smoothed)')
    ax2.scatter(test_locs, df_test['metric'], marker='x', zorder=3, color='tab:red', label='test')
    ax2.set_ylabel(metric)

    fig.supxlabel('epoch')
    fig.tight_layout()
    fig.suptitle(title, weight='bold')
    fig.tight_layout()
    return fig

# %% [markdown]
# Naive regression

# %%
nrg_df_train, nrg_df_test = load_data('naive' + os.sep + 'nrg')
nrg_fig = get_history_plot(nrg_df_train, nrg_df_test, 'Naive Regression', 'mae')
nrg_fig.savefig(DIRS['plots'] + 'nrg_history.png', dpi=500)

# %% [markdown]
# Common sense regression

# %%
for i in range(1, 5):
    model_name = f'cs{i}'
    df_train, df_test = load_data('commonsense' + os.sep + model_name)
    fig = get_history_plot(df_train, df_test, title=f'Common Sense {i}', metric='mae')
    fig.savefig(DIRS['plots'] + f'{model_name}_history.png', dpi=500)

# %% [markdown]
# Classifiers

# %%
for i in [24, 48]:
    model_name = f'c{i}'
    df_train, df_test = load_data('classify' + os.sep + model_name)
    fig = get_history_plot(df_train, df_test, title=f'{i} Classes', metric='acc')
    fig.savefig(DIRS['plots'] + f'{model_name}_history.png', dpi=500)

# %% [markdown]
# The multihead model results are stored in a different way, so we need to load them differently

# %%
csv_loc = DIRS['csv'] + 'multihead' + os.sep
mhc_df_train_loss_hrs = pd.read_csv(csv_loc + 'mhc_train_loss_hrs.csv', index_col=0)
mhc_df_train_loss_hrs = mhc_df_train_loss_hrs.melt(var_name='leg', value_name='loss')
mhc_df_train_metric_hrs = pd.read_csv(csv_loc + 'mhc_train_metric_hrs.csv', index_col=0)
mhc_df_train_metric_hrs = mhc_df_train_metric_hrs.melt(var_name='leg', value_name='metric')
mhc_df_train_hrs = pd.merge(mhc_df_train_loss_hrs, mhc_df_train_metric_hrs, left_index=True, right_index=True).drop(columns=['leg_x', 'leg_y'])

mhc_df_train_loss_min = pd.read_csv(csv_loc + 'mhc_train_loss_min.csv', index_col=0)
mhc_df_train_loss_min = mhc_df_train_loss_min.melt(var_name='leg', value_name='loss')
mhc_df_train_metric_min = pd.read_csv(csv_loc + 'mhc_train_metric_min.csv', index_col=0)
mhc_df_train_metric_min = mhc_df_train_metric_min.melt(var_name='leg', value_name='metric')
mhc_df_train_min = pd.merge(mhc_df_train_loss_min, mhc_df_train_metric_min, left_index=True, right_index=True).drop(columns=['leg_x', 'leg_y'])

mhc_df_test = pd.read_csv(csv_loc + 'mhc_test.csv', index_col=0)
mhc_df_test_hrs = mhc_df_test[[col for col in mhc_df_test.columns if 'hrs' in col]].rename(columns=lambda x: x.replace('acc', 'metric').replace('hrs_', ''))
mhc_df_test_min = mhc_df_test[[col for col in mhc_df_test.columns if 'min' in col]].rename(columns=lambda x: x.replace('acc', 'metric').replace('min_', ''))

mhc_fig_hrs = get_history_plot(mhc_df_train_hrs, mhc_df_test_hrs, title='Multi-Head (Hours)', metric='acc')
mhc_fig_hrs.savefig(DIRS['plots'] + 'mhc_hrs_history.png', dpi=500)
mhc_fig_min = get_history_plot(mhc_df_train_min, mhc_df_test_min, title='Multi-Head (Minutes)', metric='acc')
mhc_fig_min.savefig(DIRS['plots'] + 'mhc_min_history.png', dpi=500)



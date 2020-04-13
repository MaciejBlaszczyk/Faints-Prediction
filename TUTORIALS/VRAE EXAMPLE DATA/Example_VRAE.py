#!/usr/bin/env python
# coding: utf-8

# ## Timeseries clustering
# 
# Time series clustering is to partition time series data into groups based on similarity or distance, so that time series in the same cluster are similar.
# 
# Methodology followed:
# * Use Variational Recurrent AutoEncoder (VRAE) for dimensionality reduction of the timeseries
# * To visualize the clusters, PCA and t-sne are used
# 
# Paper:
# https://arxiv.org/pdf/1412.6581.pdf

# #### Contents
# 
# 0. [Load data and preprocess](#Load-data-and-preprocess)
# 1. [Initialize VRAE object](#Initialize-VRAE-object)
# 2. [Fit the model onto dataset](#Fit-the-model-onto-dataset)
# 3. [Transform the input timeseries to encoded latent vectors](#Transform-the-input-timeseries-to-encoded-latent-vectors)
# 4. [Save the model to be fetched later](#Save-the-model-to-be-fetched-later)
# 5. [Visualize using PCA and tSNE](#Visualize-using-PCA-and-tSNE)


import torch
from torch.utils.data import TensorDataset

from VAE.vrae.utils import *
from VAE.vrae.vrae import VRAE

dload = './model_dir'  # download directory

hidden_size = 90
hidden_layer_depth = 1
latent_length = 2
batch_size = 32
learning_rate = 0.0005
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam'  # options: ADAM, SGD
cuda = False  # options: True, False
print_every = 30
clip = True  # options: True, False
max_grad_norm = 5
loss = 'MSELoss'  # options: SmoothL1Loss, MSELoss
block = 'LSTM'  # options: LSTM, GRU

X_train, X_val, y_train, y_val = open_data('data', ratio_train=0.9)

num_classes = len(np.unique(y_train))
base = np.min(y_train)  # Check if data is 0-based
if base != 0:
    y_train -= base
y_val -= base

train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))

sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]


vrae = VRAE(sequence_length=sequence_length,
            number_of_features=number_of_features,
            hidden_size=hidden_size,
            hidden_layer_depth=hidden_layer_depth,
            latent_length=latent_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            cuda=cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss=loss,
            block=block,
            dload=dload)

# vrae.fit(train_dataset)
# vrae.fit(dataset, save = True)
# z_run = vrae.transform(dataset, save = True)
# vrae.save('vrae.pth')
vrae.load('vrae.pth')
z_run = vrae.transform(test_dataset)

# ### Visualize using PCA and tSNE

# In[13]:
plt.figure()
def choose_color(label):
    if label == 0.0:
        return 'r'
    elif label == 1.0:
        return 'b'
    elif label == 2.0:
        return 'g'
    elif label == 3.0:
        return 'c'
    else:
        return 'm'

colors = [choose_color(y) for y in y_val[:z_run.shape[0]]]
plt.scatter(z_run[:, 0], z_run[:, 1], c=colors, s=2)
plt.show()
# plot_clustering(z_run, y_val, engine='matplotlib', download=False)

# If plotly to be used as rendering engine, uncomment below line
# plot_clustering(z_run, y_val, engine='plotly', download = False)

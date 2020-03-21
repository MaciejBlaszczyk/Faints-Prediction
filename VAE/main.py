from vrae.vrae import VRAE
from vrae.utils import *
import numpy as np
import torch

import plotly
from torch.utils.data import DataLoader, TensorDataset
plotly.offline.init_notebook_mode()

dload = './model_dir' #download directory

hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 32

learning_rate = 0.0005
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU

# X_train, X_val, y_train, y_val = open_data('data', ratio_train=0.8)
X_train, y_train = load_datasets("data/training_set.txt")
X_val, y_val = load_datasets("data/test_set.txt")

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
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

vrae.fit(train_dataset)
z_run = vrae.transform(test_dataset)
plot_clustering(z_run, y_val, engine='matplotlib', download = False)
# If the model has to be saved, with the learnt parameters use:
# vrae.fit(dataset, save = True)
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Skalowanie daje slabe wyniki, bez skalowania jest duzo lepiej
#batch normalization polepszylo znaczaco wyniki
#


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from tensorflow.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# In[3]:


tf.get_logger().setLevel('INFO')
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, RNN, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, MSE
# import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from datetime import datetime
# from keras import backend as K
from tensorflow.keras.utils import plot_model
from mpl_toolkits.mplot3d import Axes3D
# import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# # Generate data

# In[4]:


def generate_lines(n_train, dim, noise):    
    rising_line_1 = np.array(list(range(dim)))
    rising_line_2 = np.array(list(range(dim))) / 2
    falling_line_1 = np.array(list(reversed(list(range(dim)))))
    falling_line_2 = (np.array(list(reversed(list(range(dim))))) / 2) + (2 * np.mean(rising_line_1))
    straight_line_1 = np.array(dim * [np.mean(rising_line_1)])
    straight_line_2 = np.array(dim * [2 * np.mean(rising_line_1)])

    rising_lines_1 = [rising_line_1 + np.random.normal(0,noise,dim) for _ in range(int(n_train/6))]
    rising_lines_2 = [rising_line_2 + np.random.normal(0,noise,dim) for _ in range(int(n_train/6))]
    falling_lines_1 = [falling_line_1 + np.random.normal(0,noise,dim) for _ in range(int(n_train/6))]
    falling_lines_2 = [falling_line_2 + np.random.normal(0,noise,dim) for _ in range(int(n_train/6))]
    straight_lines_1 = [straight_line_1 + np.random.normal(0,noise,dim) for _ in range(int(n_train/6))]
    straight_lines_2 = [straight_line_2 + np.random.normal(0,noise,dim) for _ in range(int(n_train/6))]
    
    lines = rising_lines_1 + rising_lines_2 + falling_lines_1 + falling_lines_2 + straight_lines_1 + straight_lines_2
    labels = ['r' for _ in range(int(n_train/6))] + ['m' for _ in range(int(n_train/6))] + ['b' for _ in range(int(n_train/6))] + ['c' for _ in range(int(n_train/6))] + ['g' for _ in range(int(n_train/6))] + ['y' for _ in range(int(n_train/6))]
    
    return np.array(lines), np.array(labels)
    


# In[5]:


x_train, y_train = generate_lines(60000, 32, 1)


# # Scale data

# In[6]:


# x_train = (x_train - np.amin(x_train)) / (np.amax(x_train) - np.amin(x_train))


# In[7]:


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)


# # Plot data

# In[8]:


plot_lines = np.concatenate((x_train[1000:1010], x_train[11000:11010], x_train[21000:21010], x_train[31000:31010],x_train[41000:41010],x_train[51000:51010])) 
for l in plot_lines:
    plt.plot(l)


# # Create model

# In[9]:


original_dim = 32
input_shape = (original_dim, )
intermediate_dim = 16
batch_size = 128
latent_dim = 8
epochs = 1

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim), stddev=0.1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[10]:


inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
x = BatchNormalization()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)

z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
x = BatchNormalization()(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


# In[11]:


models = (encoder, decoder)
reconstruction_loss = MSE(inputs, outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + (1 * kl_loss))
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
# vae.summary()


# # Learn data

# In[12]:


vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size)


# In[13]:


z_mean, _, _ = encoder.predict(x_train, batch_size=batch_size)
z_mean = PCA(n_components=3).fit_transform(z_mean)

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2],  c=y_train)


# In[14]:


res = vae.predict(x_train, batch_size=128)
res_lines = np.concatenate((res[1000:1010], 
                            res[11000:11010], 
                            res[21000:21010], 
                            res[31000:31010],
                            res[41000:41010],
                            res[51000:51010])) 
res_colors = np.concatenate((y_train[1000:1010], 
                             y_train[11000:11010], 
                             y_train[21000:21010], 
                             y_train[31000:31010],
                             y_train[41000:41010],
                             y_train[51000:51010]))

plt.figure(figsize=(20,12))
for (l, c) in zip(res_lines, res_colors):
    plt.plot(l, c=c)


# In[ ]:





# In[ ]:





# In[15]:


low = 0
high = 1000
d = 10000

for i in range(6):
    res_lines = res[low:high]
    res_colors = y_train[low:high]
    plt.figure(figsize=(20,12))
    for (l, c) in zip(res_lines, res_colors):
        plt.plot(l, c=c)
    low += d
    high += d


# In[ ]:





# In[ ]:





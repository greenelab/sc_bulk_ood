# general imports
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Softmax, ReLU, ELU, LeakyReLU
from keras.layers.merge import concatenate as concat
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, KLDivergence
from tensorflow.keras.datasets import mnist
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from tqdm import tnrange, tqdm_notebook

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# programming stuff
import time
import os
import pickle
from pathlib import Path

# disable eager execution
# https://github.com/tensorflow/tensorflow/issues/47311#issuecomment-786116401
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def fit_model(known_prop_vae, unknown_prop_vae, X_unknown_prop, 
              label_unknown_prop, X_known_prop, Y_known_prop, 
              label_known_prop, epochs, batch_size):
    assert len(X_known_prop) % len(X_unknown_prop) == 0, \
            (len(X_unknown_prop), batch_size, len(X_known_prop))
    start = time.time()
    history = []
    
    for epoch in range(epochs):
        labeled_index = np.arange(len(X_known_prop))
        np.random.shuffle(labeled_index)

        # Repeat the unlabeled data to match length of labeled data
        unlabeled_index = []
        for i in range(len(X_known_prop) // len(X_unknown_prop)):
            l = np.arange(len(X_unknown_prop))
            np.random.shuffle(l)
            unlabeled_index.append(l)
        unlabeled_index = np.concatenate(unlabeled_index)
        
        batches = len(X_unknown_prop) // batch_size
        for i in range(batches):
            # Labeled
            index_range =  labeled_index[i * batch_size:(i+1) * batch_size]
            loss = known_prop_vae.train_on_batch(X_known_prop[index_range], 
                                                    [X_known_prop[index_range], Y_known_prop[index_range], label_known_prop[index_range]])
            
            # Unlabeled
            y_shuffle = np.identity(10, dtype=np.float32)
            for idx in range(0, 49):
                y_shuffle = np.vstack((y_shuffle, np.identity(10, dtype=np.float32)))
            #np.random.shuffle(y_shuffle)
            y_shuffle = np.zeros((batch_size, 10))
            y_shuffle[:,0] = 1
            index_range =  unlabeled_index[i * batch_size:(i+1) * batch_size]
            loss += [unknown_prop_vae.train_on_batch(X_unknown_prop[index_range], 
                                                        [X_unknown_prop[index_range], y_shuffle, label_unknown_prop[index_range]])]


            history.append(loss)
            
    
   
    done = time.time()
    elapsed = done - start
    print("Elapsed: ", elapsed)
    
    return history

def instantiate_model(n_x,
                    n_y,
                    n_label,
                    n_z,
                    decoder_out_dim,
                    n_label_z = 64,
                    encoder_dim = 512,
                    decoder_dim = 512,
                    batch_size = 500,
                    n_epoch = 100,
                    alpha_rot = 1000000,
                    alpha_prop = 100,
                    beta_kl_slack = 10,
                    beta_kl_rot = 100,
                    beta_kl_prop = 10,
                    activ = 'relu',
                    optim = Adam(learning_rate=0.001)):


    def null_f(args):
        return args

    # helper methods for evaluation
    def sum_abs_error(y_pred, y_true):
      return sum(abs(y_pred - y_true))

    def mean_abs_error(y_pred, y_true):
      return np.mean(abs(y_pred - y_true))

    def mean_sqr_error(y_pred, y_true):
      return np.mean((y_pred - y_true)**2)

    # declare the Keras tensor we will use as input to the encoder
    X = Input(shape=(n_x,))
    label = Input(shape=(n_label,))
    props = Input(shape=(n_y,))

    inputs = X

    # set up encoder network
    # this is an encoder with encoder_dim hidden layer
    encoder_s = Dense(encoder_dim, activation=activ, name="encoder_slack")(inputs)
    encoder_p = Dense(encoder_dim, activation=activ, name="encoder_prop")(inputs)
    encoder_r = Dense(encoder_dim, activation=activ, name="encoder_rot")(inputs)

    # now from the hidden layer, you get the mu and sigma for 
    # the latent space which is divided into three sections
    # slack, proportions, rotation (domain)
    mu_slack = Dense(n_z, activation='linear', name = "mu_slack")(encoder_s)
    l_sigma_slack = Dense(n_z, activation='linear', name = "sigma_slack")(encoder_s)

    mu_prop = Dense(n_label_z, activation='linear', name = "mu_prop")(encoder_p)
    l_sigma_prop = Dense(n_label_z, activation='linear', name = "sigma_prop")(encoder_p)

    mu_rot = Dense(n_label_z, activation='linear', name = "mu_rot")(encoder_r)
    l_sigma_rot = Dense(n_label_z, activation='linear', name = "sigma_rot")(encoder_r)
    

    # sampler from mu and sigma
    def sample_z(args):
        mu, l_sigma, n_z = args
        eps = K.random_normal(shape=(batch_size, n_z), mean=0., stddev=1.)
        return mu + K.exp(l_sigma / 2) * eps


    # Sampling latent space
    z_slack = Lambda(sample_z, output_shape = (n_z, ), name="z_samp_slack")([mu_slack, l_sigma_slack, n_z])
    z_prop = Lambda(sample_z, output_shape = (n_label_z, ), name="z_samp_prop")([mu_prop, l_sigma_prop, n_label_z])
    z_rot = Lambda(sample_z, output_shape = (n_label_z, ), name="z_samp_rot")([mu_rot, l_sigma_rot, n_label_z])

    z_concat = concat([z_slack, z_prop, z_rot])


    ###### DECODER
    # set up decoder network
    # this is a decoder with 512 hidden layer
    decoder_hidden = Dense(decoder_dim, activation=activ, name = "decoder_h1")

    # final reconstruction
    decoder_out = Dense(decoder_out_dim, activation='sigmoid', name = "decoder_out")

    d_in = Input(shape=(n_z+n_label_z+n_label_z,))
    d_h1 = decoder_hidden(d_in)
    d_out = decoder_out(d_h1)

    # set up the decoder part that links to the encoder
    h_p = decoder_hidden(z_concat)
    outputs = decoder_out(h_p)

    ###### Proportions classifier
    # this is the proportions we try to estimate
    prop_h1 = Dense(20, activation='linear', name = "prop_h1")
    prop_h2 = Dense(n_z, activation='linear', name = "prop_h2")
    prop_softmax = Softmax(name = "mu_prop_pred")
    decoder_sigma = Lambda(null_f, name = "l_sigma_prop_pred")

    prop_1_out = prop_h1(z_prop)
    prop_2_out = prop_h2(prop_1_out)
    prop_outputs = prop_softmax(prop_2_out)
    sigma_outputs_p = decoder_sigma(l_sigma_rot)


    ###### Rotations classifier
    # this is the rotation we try to estimate
    rot_h1 = ReLU(name = "rot_h1")
    rot_h2 = Dense(n_label, activation='linear', name = "rot_h2")
    rot_softmax = Softmax(name = "mu_rot_pred")
    decoder_sigma_r = Lambda(null_f, name = "l_sigma_rot_pred")

    rot_1_out = rot_h1(z_rot)
    rot_2_out = rot_h2(rot_1_out)
    rotation_outputs = rot_softmax(rot_2_out)
    sigma_outputs_r = decoder_sigma_r(l_sigma_rot)


    ###### Loss functions where you need access to internal variables
    def vae_loss(y_true, y_pred):
        recon = K.sum(mean_squared_error(y_true, y_pred), axis=-1)
        kl_prop = beta_kl_prop * K.sum(K.exp(l_sigma_prop) + K.square(mu_prop) - 1. - l_sigma_prop, axis=-1)
        kl_rot = beta_kl_rot * K.sum(K.exp(l_sigma_rot) + K.square(mu_rot) - 1. - l_sigma_rot, axis=-1)
        kl_slack = beta_kl_slack * K.sum(K.exp(l_sigma_slack) + K.square(mu_slack) - 1. - l_sigma_slack, axis=-1)
        return recon + kl_prop + kl_rot + kl_slack

    def prop_loss_unknown(y_true, y_pred):
      return K.sum(mean_absolute_error(prop_outputs, y_pred), axis=-1) * alpha_prop

    def prop_loss(y_true, y_pred):
      return K.sum(mean_absolute_error(y_true, y_pred), axis=-1) * alpha_prop

    def class_loss(y_true, y_pred):
        recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)*alpha_rot
        return recon

    ##### link it all together
    known_prop_vae = Model(X, [outputs, prop_outputs, rotation_outputs, sigma_outputs_p])
    unknown_prop_vae = Model(X, [outputs, prop_outputs, rotation_outputs])

    known_prop_vae.compile(optimizer=optim, loss=[vae_loss, prop_loss, class_loss, None])
    unknown_prop_vae.compile(optimizer=optim, loss=[vae_loss, prop_loss_unknown, class_loss])


    encoder = Model(X, [z_slack, mu_slack, l_sigma_slack, mu_prop, l_sigma_prop, prop_outputs, z_rot, mu_rot, l_sigma_rot])

    decoder = Model(d_in, d_out)

    return (known_prop_vae, unknown_prop_vae, encoder, decoder)
# import the VAE code
import sys
sys.path.insert(1, '../../')
from diva import diva
from sc_preprocessing import sc_preprocess

# general imports
import warnings
import numpy as np
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import Adam


# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

# programming stuff
import time
import os
import pickle
from pathlib import Path
from argparse import ArgumentParser

# set seeds
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)



if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-res", "--res_data_path", dest="res_data_path",
                        help="path to write DIVA results")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to augmented GBM28 files")
    parser.add_argument("-n", "--num_genes",
                        dest="num_genes", type=int,
                        help="Number of features (genes) for VAE")

    args = parser.parse_args()

    print('parser', flush=True)


    ##################################################
    #####. set up experiment specific variables
    ##################################################

    # number of patients/domains/samples expected
    idx_range = range(26)
    n_tot_samples = 26

    # number expected cell types
    n_cell_types = 4

    # number of pseudobulks PER patient
    n_train = 1000

    # patient IDs
    # skip "MGH85", "BT1187"
    # MGH122 must be the first one, because this excludes it from the 
    # training 
    patient_ids = ["MGH125", "MGH122", "MGH101", "MGH66", "MGH100",  "MGH102", 
                    "MGH104", "MGH105", "MGH106", "MGH110", 
                    "MGH113", "MGH115", "MGH121", 
                    "MGH124", "MGH128", "MGH129", 
                    "MGH136", "MGH143", "MGH151", "MGH152", 
                    "BT749", "BT771", "BT786", "BT830", 
                    "BT920", "BT1160"]

    ### create the domains label 
    Label_full = np.concatenate([np.full(n_train, i) for i in range(n_tot_samples)], axis=0)
    label_full = to_categorical(Label_full)


    print('finished setting up variables', flush=True)


    ##################################################
    #####. Design the experiment
    ##################################################
    X_train_all = []
    Y_train_all = []
    for curr_id in patient_ids:
        print(f'{curr_id} start \n', flush=True)
        X_train, Y_train, gene_df, _ = sc_preprocess.read_diva_files(args.aug_data_path, None, curr_id)
        Y_train.reindex(columns=['Malignant', 'Macrophage', 'T-cell', 'Oligodendrocyte'], fill_value=0)
        X_train_all.append(X_train)
        Y_train_all.append(Y_train)
        print(f'{curr_id} end \n', flush=True)


    print('1', flush=True)

    X_train = pd.concat(X_train_all, ignore_index=True)
    Y_train = pd.concat(Y_train_all, ignore_index=True)
    X_full = X_train.to_numpy()
    Y_full = Y_train.to_numpy()
    gene_df = gene_df.to_frame(index=False, name='gene_ids')

    print('2', flush=True)

    ## remove genes with no expression
    X_colmean = X_full.mean(axis=0)
    X_full = X_full[:,np.where(X_colmean > 0)[0]]
    gene_df = gene_df.iloc[np.where(X_colmean > 0)[0]]
    print('3', flush=True)

    ## get the top variable genes
    X_colmean = X_full.mean(axis=0)
    X_colvar = X_full.var(axis=0)
    X_CoV = np.array(np.divide(X_colvar, X_colmean))
    idx_top = np.argpartition(X_CoV, -args.num_genes)[-args.num_genes:]
    X_full = X_full[:,idx_top]
    gene_df = gene_df.iloc[idx_top]

    print('filtered genes', flush=True)

    ## normalize within sample
    X_full = scale(X_full, axis=1)
                
    print(X_full.shape)

    print('scaled')

    # indexes for the training, 3 and 0
    # 3 is unlabeled
    # 0 is held out
    idx_train = np.where(np.logical_and(Label_full!=2, Label_full!=0))[0]
    idx_3 = np.where(Label_full==2)[0]
    idx_0 = np.where(Label_full==0)[0]

    # for unknown proportions; i.e. 3 
    X_unkp = X_full[idx_3,]
    label_unkp = label_full[idx_3,]
    y_unkp = Y_full[idx_3,]

    # for known proportions
    X_kp = X_full[idx_train,]
    label_kp = label_full[idx_train,]
    y_kp = Y_full[idx_train,]


    # test
    X_0 = X_full[idx_0,]
    label_0 = label_full[idx_0,]
    y_0 = Y_full[idx_0,]

    print('hyperparameters')


    ##################################################
    #####. Hyperparameters
    ##################################################

    batch_size = 500
    n_epoch = 500 # 100

    alpha_rot = 1000000
    alpha_prop = 100 

    beta_kl_slack = 10
    beta_kl_rot = 100
    beta_kl_prop = 10


    n_x = X_full.shape[1]
    n_y = Y_full.shape[1]
    n_label = n_tot_samples  # 6 "patients" 1 sample augmented into 6 distinct versions
    n_label_z = 64  # 64 dimensional representation of rotation


    # the network dimensions are 784 > 512 > proportion_dim < 512 < 784
    n_z = Y_full.shape[1] # latent space size, one latent dimension PER cell type
    encoder_dim = 512 # dim of encoder hidden layer
    decoder_dim = 512 # dim of encoder hidden layer
    decoder_out_dim = n_x # dim of decoder output layer

    activ = 'relu'
    optim = Adam(learning_rate=0.001)

    print(f"length of X {n_x} and length of y {n_y} and n_label {n_label}")

    print('about to instantiate', flush=True)


    ##################################################
    #####. Train Model
    ##################################################
    known_prop_vae, unknown_prop_vae, encoder, decoder = diva.instantiate_model(n_x=n_x,
                                                            n_y=n_y,
                                                            n_label=n_label,
                                                            n_z=n_z,
                                                            decoder_out_dim = decoder_out_dim,
                                                            n_label_z = n_label_z,
                                                            encoder_dim = encoder_dim,
                                                            decoder_dim = decoder_dim,
                                                            batch_size = batch_size,
                                                            n_epoch = n_epoch,
                                                            alpha_rot = alpha_rot,
                                                            alpha_prop = alpha_prop,
                                                            beta_kl_slack = beta_kl_slack,
                                                            beta_kl_rot = beta_kl_rot,
                                                            beta_kl_prop = beta_kl_prop,
                                                            activ = activ,
                                                            optim = optim)

    X_unkp = np.asarray(X_unkp).astype('float32')
    label_unkp = np.asarray(label_unkp).astype('float32')
    X_kp = np.asarray(X_kp).astype('float32')
    y_kp = np.asarray(y_kp).astype('float32')
    label_kp = np.asarray(label_kp).astype('float32')

    print('about to train', flush=True)

    loss_history = diva.fit_model(known_prop_vae, 
                                    unknown_prop_vae,
                                    X_unkp,
                                    label_unkp,
                                    X_kp, 
                                    y_kp,
                                    label_kp, 
                                    epochs=n_epoch,
                                    batch_size=batch_size)
    print('trained')

    known_prop_vae.save(f"{args.res_data_path}/known_prop_vae")
    unknown_prop_vae.save(f"{args.res_data_path}/unknown_prop_vae")
    encoder.save(f"{args.res_data_path}/encoder")
    decoder.save(f"{args.res_data_path}/decoder")


    print('saved', flush=True)

    # write out the loss for later plotting
    # unpack the loss values
    labeled_total_loss = [item[0] for item in loss_history]
    unlabeled_total_loss = [item[4][0] for item in loss_history]

    labeled_recon_loss = [item[1] for item in loss_history]
    unlabeled_recon_loss = [item[4][1] for item in loss_history]

    labeled_prop_loss = [item[2] for item in loss_history]
    unlabeled_prop_loss = [item[4][2] for item in loss_history]


    labeled_samp_loss = [item[3] for item in loss_history]
    unlabeled_samp_loss = [item[4][3] for item in loss_history]


    # make into a dataframe
    total_loss = labeled_total_loss + unlabeled_total_loss + [a + b for a, b in zip(labeled_total_loss, unlabeled_total_loss)]
    loss_df = pd.DataFrame(data=total_loss, columns=['total_loss'])
    loss_df['type'] = ["labeled"]*len(loss_history) + ["unlabeled"]*len(loss_history) + ["sum"]*len(loss_history)
    loss_df['batch'] = [*range(len(loss_history))] + [*range(len(loss_history))] + [*range(len(loss_history))]

    recon_loss = labeled_recon_loss + unlabeled_recon_loss + [a + b for a, b in zip(labeled_recon_loss, unlabeled_recon_loss)]
    loss_df['recon_loss'] = recon_loss

    prop_loss = labeled_prop_loss + unlabeled_prop_loss + [a + b for a, b in zip(labeled_prop_loss, unlabeled_prop_loss)]
    loss_df['prop_loss'] = prop_loss

    samp_loss = labeled_samp_loss + unlabeled_samp_loss + [a + b for a, b in zip(labeled_samp_loss, unlabeled_samp_loss)]
    loss_df['samp_loss'] = samp_loss

    loss_file = os.path.join(args.res_data_path, f"train-gbm28-DIVA_loss.pkl")
    loss_df.to_pickle(loss_file)


    # write out the features for testing
    gene_file = os.path.join(args.res_data_path, f"train-gbm28-DIVA_features.pkl")
    gene_df.to_pickle(gene_file)

    print('written')

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
                        help="path to write out augmented PBMC files")
    parser.add_argument("-exp", "--exp_id",
                        dest="exp_id",
                        help="ID of GSE experiment to use")
    parser.add_argument("-unlab_exp", "--unlab_exp_id",
                        dest="unlab_exp_id",
                        help="ID of GSE experiment to use as the unlabeled value")
    parser.add_argument("-n", "--num_genes",
                        dest="num_genes", type=int,
                        help="Number of features (genes) for VAE")


    args = parser.parse_args()

    ##################################################
    #####. set up experiment specific variables
    ##################################################

    ## set up flags for needed parameters
    labeled_10x = False
    unlabeled_10x = False

    # number expected cell types
    n_cell_types = 6

    if args.exp_id == "pbmc_rep1_10xV2a_sm2_cells" or args.exp_id == "pbmc_rep2_10xV2_sm2_cells":
        labeled_10x = True
    if args.unlab_exp_id == "pbmc_rep1_10xV2a_sm2_cells" or args.unlab_exp_id == "pbmc_rep2_10xV2_sm2_cells":
        unlabeled_10x = True

    # set up for special case where 10x is used for labeled and unlabeled, and same sample
    if labeled_10x and unlabeled_10x and args.exp_id == args.unlab_exp_id:
        # number of patients/domains/samples expected
        idx_range_lab = range(0, 11)
        idx_range_unlab = range(11, 12)
        n_tot_samples = 12


        # experiment id
        lab_file_name = args.exp_id
        unlab_file_name = args.unlab_exp_id

        # number of pseudobulks PER patient
        n_train = 1000

        ### create the domains label 
        Label_full = np.concatenate([np.full(n_train, 0), np.full(n_train, 1),
                                    np.full(n_train, 2), np.full(n_train, 3),
                                    np.full(n_train, 4), np.full(n_train, 5),
                                    np.full(n_train, 6), np.full(n_train, 7),
                                    np.full(n_train, 8), np.full(n_train, 9),
                                    np.full(n_train, 10), np.full(n_train, 11)], axis=0)
        label_full = to_categorical(Label_full)

        # since the unlabeled idx is the last block of the training data
        # its idx in X_train and Y_train is the max labeled index +1
        unlabeled_idx = max(idx_range_lab)+1

        # indexes for the training, 3 and 0
        # 3 is unlabeled
        # 0 is held out
        idx_train = np.where(np.logical_and(Label_full!=unlabeled_idx, Label_full!=0))[0]
        idx_unlab = np.where(Label_full==unlabeled_idx)[0]
        idx_0 = np.where(Label_full==0)[0]
  
    # set up for case where 10x is used for labeled and unlabeled, and different sample
    elif labeled_10x and unlabeled_10x and args.exp_id != args.unlab_exp_id:
        # number of patients/domains/samples expected
        idx_range_lab = range(0, 12)
        idx_range_unlab = range(0, 12)
        n_tot_samples = 24

        # experiment id
        lab_file_name = args.exp_id
        unlab_file_name = args.unlab_exp_id

        # number of pseudobulks PER patient
        n_train = 1000

        ### create the domains label 
        Label_full = np.concatenate([np.full(n_train, 0), np.full(n_train, 1),
                                    np.full(n_train, 2), np.full(n_train, 3),
                                    np.full(n_train, 4), np.full(n_train, 5),
                                    np.full(n_train, 6), np.full(n_train, 7),
                                    np.full(n_train, 8), np.full(n_train, 9),
                                    np.full(n_train, 10), np.full(n_train, 11)], axis=0)
        Label_full = np.concatenate([Label_full, Label_full+12])
        label_full = to_categorical(Label_full)

        # indexes for the training
        # 12-23 is unlabeled
        # 0 is held out
        idx_train = np.where(np.logical_and(Label_full>0, Label_full<12))[0]
        idx_unlab = np.where(Label_full>=12)[0]
        idx_0 = np.where(Label_full==0)[0]


    # set up for case where 10x is used for labeled and SM2 unlabeled
    elif labeled_10x and not unlabeled_10x:
        # number of patients/domains/samples expected
        idx_range_lab = range(0, 11)
        idx_range_unlab = range(0, 5)
        n_tot_samples = 16

        # experiment id
        lab_file_name = args.exp_id
        unlab_file_name = args.unlab_exp_id

        # number of pseudobulks PER patient
        n_train = 1000

        ### create the domains label 
        Label_full = np.concatenate([np.full(n_train, 0), np.full(n_train, 1),
                                    np.full(n_train, 2), np.full(n_train, 3),
                                    np.full(n_train, 4), np.full(n_train, 5),
                                    np.full(n_train, 6), np.full(n_train, 7),
                                    np.full(n_train, 8), np.full(n_train, 9),
                                    np.full(n_train, 10)], axis=0)
        Label_full_sm2 = np.concatenate([np.full(n_train, 0), np.full(n_train, 1),
                            np.full(n_train, 2), np.full(n_train, 3),
                            np.full(n_train, 4)], axis=0)
        Label_full = np.concatenate([Label_full, Label_full_sm2+11])
        label_full = to_categorical(Label_full)

        # indexes for the training
        # 12-17 is unlabeled
        # 0 is held out
        idx_train = np.where(np.logical_and(Label_full>0, Label_full<13))[0]
        idx_unlab = np.where(Label_full>=13)[0] # 11
        idx_0 = np.where(Label_full==0)[0]

    else:
        sys.exit("Error, undefined training pairs")

    ##################################################
    #####. Design the experiment
    ##################################################

    # read in the labeled data
    X_train, Y_train, gene_df = sc_preprocess.read_all_diva_files(args.aug_data_path, idx_range_lab, lab_file_name)

    # only get genes that are available in both testing and training
    common_genes_file = os.path.join(args.aug_data_path, "intersection_genes.pkl")
    gene_out_path = Path(common_genes_file)
    common_genes = pickle.load(open( gene_out_path, "rb" ))
    X_train = X_train[common_genes]
    X_train.head()

    gene_df = gene_df.loc[gene_df['gene_ids'].isin(common_genes)]


    # read in the unlabeled data
    X_train_unlab, Y_train_unlab, gene_df_unlab = sc_preprocess.read_all_diva_files(args.aug_data_path, idx_range_unlab, unlab_file_name)

    # now we need to ensure the genes are in the same order
    X_train_unlab.columns = gene_df_unlab["gene_ids"]
    X_train_unlab = X_train_unlab.reindex(columns=gene_df["gene_ids"], fill_value=0)

    # we also need to ensure that the cell-types are in the same order
    Y_train_unlab = Y_train_unlab.reindex(columns=Y_train.columns, fill_value=0)

    # convert to data matrices
    X_full = X_train.to_numpy()
    Y_full = Y_train.to_numpy()
    X_train_unlab = X_train_unlab.to_numpy()
    Y_train_unlab = Y_train_unlab.to_numpy()


    # append together the labeled and unlabeled data
    X_full = np.concatenate((X_full, X_train_unlab), axis=0)
    Y_full = np.concatenate((Y_full, Y_train_unlab), axis=0)


    ## get the top variable genes
    X_colmean = X_full.mean(axis=0)
    X_colvar = X_full.var(axis=0)
    X_CoV = np.array(np.divide(X_colvar, X_colmean))
    idx_top = np.argpartition(X_CoV, -args.num_genes)[-args.num_genes:]
    X_full = X_full[:,idx_top]
    gene_df = gene_df.iloc[idx_top]

    ## normalize within sample
    X_full = scale(X_full, axis=1)
                
    print(X_full.shape)



    # for unknown proportions; i.e. 3 
    X_unkp = X_full[idx_unlab,]
    label_unkp = label_full[idx_unlab,]
    y_unkp = Y_full[idx_unlab,]

    # for known proportions
    X_kp = X_full[idx_train,]
    label_kp = label_full[idx_train,]
    y_kp = Y_full[idx_train,]


    # test
    X_0 = X_full[idx_0,]
    label_0 = label_full[idx_0,]
    y_0 = Y_full[idx_0,]


    ##################################################
    #####. Hyperparameters
    ##################################################

    batch_size = 500
    n_epoch = 100 # 500

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

    loss_history = diva.fit_model(known_prop_vae, 
                                    unknown_prop_vae,
                                    X_unkp,
                                    label_unkp,
                                    X_kp, 
                                    y_kp,
                                    label_kp, 
                                    epochs=n_epoch,
                                    batch_size=batch_size)

    known_prop_vae.save(f"{args.res_data_path}/{args.exp_id}_{args.unlab_exp_id}_known_prop_vae")
    unknown_prop_vae.save(f"{args.res_data_path}/{args.exp_id}_{args.unlab_exp_id}_unknown_prop_vae")
    encoder.save(f"{args.res_data_path}/{args.exp_id}_{args.unlab_exp_id}_encoder")
    decoder.save(f"{args.res_data_path}/{args.exp_id}_{args.unlab_exp_id}_decoder")

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

    loss_file = os.path.join(args.res_data_path, f"train-{args.exp_id}-{args.unlab_exp_id}-DIVA_loss.pkl")
    loss_df.to_pickle(loss_file)

    # write out the features for testing
    gene_file = os.path.join(args.res_data_path, f"train-{args.exp_id}-{args.unlab_exp_id}-DIVA_features.pkl")
    gene_df.to_pickle(gene_file)
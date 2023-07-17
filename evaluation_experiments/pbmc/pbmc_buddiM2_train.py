# import the VAE code
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from diva import diva_m2
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
from sklearn.preprocessing import MinMaxScaler

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
    parser.add_argument("-cib_genes", "--cibersort_genes",
                        dest="cibersort_genes",
                        help="Path to cibersort signature matrix")
    parser.add_argument("-n", "--num_genes",
                        dest="num_genes", type=int,
                        help="Number of features (genes) for VAE")


    args = parser.parse_args()

    ##################################################
    #####. set up experiment specific variables
    ##################################################
   
    # number expected cell types
    n_cell_types = 7

    # number of patients/domains/samples expected
    idx_range = range(0, 10)
    n_tot_samples = 10

    # number of drugs (one-hot encoded)
    n_drugs = 2

    # experiment id
    lab_file_name = args.exp_id

    # number of pseudobulks PER patient
    n_train = 1000



    ### create the domains label 
    Label_full = np.concatenate([np.full(n_train, 0), np.full(n_train, 1),
                                np.full(n_train, 2), np.full(n_train, 3),
                                np.full(n_train, 4), np.full(n_train, 5),
                                np.full(n_train, 6), np.full(n_train, 7),
                                np.full(n_train, 8), np.full(n_train, 9)], axis=0)
    label_full = to_categorical(Label_full)

    ### create the drugs label 
    Drug_full = np.concatenate([np.full(n_train, 1), np.full(n_train, 0),
                                np.full(n_train, 0), np.full(n_train, 0),
                                np.full(n_train, 0), np.full(n_train, 1),
                                np.full(n_train, 1), np.full(n_train, 1),
                                np.full(n_train, 0), np.full(n_train, 0)], axis=0)
    drug_full = to_categorical(Drug_full)

    # indexes for the training
    # 1-2 is labeled 
    # 0,3-9 is unlabeled
    idx_train = np.where(np.logical_and(Label_full > 0, Label_full < 3))[0] #np.where(np.logical_and(Label_full > -1, Label_full < 2))[0]
    idx_unlab = np.where(np.logical_or(Label_full == 0, Label_full > 2))[0] #np.where(Label_full >= 2)[0]
    idx_drug = np.where(Drug_full > 0)[0]
    idx_5 = np.where(Label_full == 5)[0]
    idx_0 = np.where(Label_full==0)[0]

    ##################################################
    #####. Design the experiment
    ##################################################

    # read in the labeled data
    X_train, Y_train, gene_df = sc_preprocess.read_all_diva_files(args.aug_data_path, idx_range, lab_file_name)
    X_train.columns = gene_df

    # only get genes that are available in both testing and training
    common_genes_file = os.path.join(args.aug_data_path, "intersection_genes.pkl")
    gene_out_path = Path(common_genes_file)
    common_genes = pickle.load(open( gene_out_path, "rb" ))

    # try using the cibersort genes
    cibersort_df = pd.read_csv(args.cibersort_genes, sep="\t" )
    cibersort_genes = cibersort_df["NAME"].values.tolist()

    common_genes = np.union1d(common_genes, cibersort_genes)

    X_train = X_train[common_genes]

    gene_df = gene_df.loc[gene_df.isin(common_genes)]


    # convert to data matrices
    X_full = X_train.to_numpy()
    Y_full = Y_train.to_numpy()

    ## get the top variable genes
    X_colmean = X_full.mean(axis=0)
    X_colvar = X_full.var(axis=0)
    X_CoV = np.array(np.divide(X_colvar, X_colmean+0.001))
    idx_top = np.argpartition(X_CoV, -args.num_genes)[-args.num_genes:]
    gene_df = gene_df.iloc[idx_top]

    # get both the CoV genes and the cibersort genes
    # use thos genes for the model
    union_genes = np.union1d(gene_df, cibersort_genes)
    X_full = X_train.loc[:,union_genes]


    ## normalize within sample
    clip_upper = np.quantile(X_full, 0.9)
    X_full = np.clip(X_full, 0, clip_upper)
    scaler = MinMaxScaler()
    scaler.fit(X_full)
    X_full = scaler.transform(X_full)



    # for unknown proportions; i.e. 3 
    X_unkp = X_full[idx_unlab,]
    label_unkp = label_full[idx_unlab,]
    drug_unkp = drug_full[idx_unlab,]
    y_unkp = Y_full[idx_unlab,]

    # for known proportions
    X_kp = X_full[idx_train,]
    label_kp = label_full[idx_train,]
    drug_kp = drug_full[idx_train,]
    y_kp = Y_full[idx_train,]



    ##################################################
    #####. Hyperparameters
    ##################################################

    batch_size = 500
    n_epoch = 100 # 500 

    alpha_rot = 1000000 #1000000
    alpha_prop = 100 #100

    beta_kl_slack = 0.01 # 10 ###
    beta_kl_rot = 100 # 100 ###


    n_x = X_full.shape[1]
    n_y = Y_full.shape[1]
    n_label = n_tot_samples  # 6 "patients" 1 sample augmented into 6 distinct versions
    n_drugs = n_drugs  # number of drugs one-hot encoded
    n_label_z = 64  # 64 dimensional representation of rotation


    # the network dimensions are 784 > 512 > proportion_dim < 512 < 784
    n_z = Y_full.shape[1] # latent space size, one latent dimension PER cell type
    encoder_dim = 512 # dim of encoder hidden layer 512 
    decoder_dim = 512 # dim of encoder hidden layer 512 
    decoder_out_dim = n_x # dim of decoder output layer

    # labeled classifier
    class_dim1 = 512 # 512 
    class_dim2 = 256 # 256 


    activ = 'relu'
    optim = Adam(learning_rate=0.0005) #0.001
    print(f"length of X {n_x} and length of y {n_y} n_label {n_label} and n_drugs {n_drugs}")


    ##################################################
    #####. Train Model first pass
    ##################################################
    known_prop_vae, unknown_prop_vae, encoder_unlab, encoder_lab, decoder, classifier = diva_m2.instantiate_model(n_x=n_x,
                                                                                            n_y=n_y,
                                                                                            n_label=n_label,
                                                                                            n_z=n_z,
                                                                                            decoder_out_dim = decoder_out_dim,
                                                                                            n_label_z = n_label_z,
                                                                                            encoder_dim = encoder_dim,
                                                                                            decoder_dim = decoder_dim,
                                                                                            class_dim1 = class_dim1,
                                                                                            class_dim2 = class_dim2,
                                                                                            batch_size = batch_size,
                                                                                            n_epoch = n_epoch,
                                                                                            alpha_rot = alpha_rot,
                                                                                            alpha_prop = alpha_prop,
                                                                                            beta_kl_slack = beta_kl_slack,
                                                                                            beta_kl_rot = beta_kl_rot,
                                                                                            activ = activ,
                                                                                            optim = optim)


    loss_history = diva_m2.fit_model(known_prop_vae, unknown_prop_vae,
                        encoder_unlab, encoder_lab, decoder, classifier,
                        X_unkp, label_unkp,
                        X_kp, y_kp,label_kp, 
                        epochs=n_epoch, batch_size=batch_size)


    known_prop_vae.save(f"{args.res_data_path}/{args.exp_id}_known_prop_vae")
    unknown_prop_vae.save(f"{args.res_data_path}/{args.exp_id}_unknown_prop_vae")
    encoder_unlab.save(f"{args.res_data_path}/{args.exp_id}_encoder_unlab")
    encoder_lab.save(f"{args.res_data_path}/{args.exp_id}_encoder_lab")
    decoder.save(f"{args.res_data_path}/{args.exp_id}_decoder")
    classifier.save(f"{args.res_data_path}/{args.exp_id}_classifier")

    meta_history = loss_history[1]
    loss_history = loss_history[0]

    # write out the loss for later plotting
    # unpack the loss values
    labeled_total_loss = [item[0] for item in loss_history]
    unlabeled_total_loss = [item[4][0] for item in loss_history]

    labeled_recon_loss = [item[1] for item in loss_history]
    unlabeled_recon_loss = [item[4][1] for item in loss_history]

    labeled_prop_loss = [item[2] for item in loss_history]

    labeled_samp_loss = [item[3] for item in loss_history]
    unlabeled_samp_loss = [item[4][2] for item in loss_history]


    # make into a dataframe
    total_loss = labeled_total_loss + unlabeled_total_loss + [a + b for a, b in zip(labeled_total_loss, unlabeled_total_loss)]
    loss_df = pd.DataFrame(data=total_loss, columns=['total_loss'])
    loss_df['type'] = ["labeled"]*len(loss_history) + ["unlabeled"]*len(loss_history) + ["sum"]*len(loss_history)
    loss_df['batch'] = [*range(len(loss_history))] + [*range(len(loss_history))] + [*range(len(loss_history))]

    recon_loss = labeled_recon_loss + unlabeled_recon_loss + [a + b for a, b in zip(labeled_recon_loss, unlabeled_recon_loss)]
    loss_df['recon_loss'] = recon_loss

    prop_loss = labeled_prop_loss + [0]*len(loss_history) + labeled_prop_loss
    loss_df['prop_loss'] = prop_loss

    samp_loss = labeled_samp_loss + unlabeled_samp_loss + [a + b for a, b in zip(labeled_samp_loss, unlabeled_samp_loss)]
    loss_df['samp_loss'] = samp_loss

    # add the log to make it easier to plot
    loss_df["log_total_loss"] = np.log10(loss_df["total_loss"]+1)
    loss_df["log_recon_loss"] = np.log10(loss_df["recon_loss"]+1)
    loss_df["log_samp_loss"] = np.log10(loss_df["samp_loss"]+1)
    loss_df["log_prop_loss"] = np.log10(loss_df["prop_loss"]+1)


    # write loss out
    loss_file = os.path.join(args.res_data_path, f"train-{args.exp_id}-DIVA_loss.pkl")
    loss_df.to_pickle(loss_file)

    # write out the features for testing
    gene_file = os.path.join(args.res_data_path, f"train-{args.exp_id}-DIVA_features.pkl")
    union_genes_df = pd.DataFrame(union_genes)
    union_genes_df.to_pickle(gene_file)
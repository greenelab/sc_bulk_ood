# import the VAE code
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from diva import diva
from sc_preprocessing import sc_preprocess

# general imports
import warnings
import numpy as np
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow   #added these following two imports:
from tensorflow.python.eager import context
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
                        help="path to write out augmented cortex files")  #pbmc to cortex change
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
            
    # number expected cell types
    n_cell_types = 7

    # number of patients/domains/samples expected
    idx_range_lab = range(10)  ##changed from range(0,10)
    #idx_range_unlab = range(0, 10)
    n_tot_samples = 10

    # number of drugs (one-hot encoded)
    #n_drugs = 2

    # experiment id
    lab_file_name = args.exp_id
    #unlab_file_name = args.unlab_exp_id

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
    #Drug_full = np.concatenate([np.full(n_train, 1), np.full(n_train, 0),
    #                            np.full(n_train, 0), np.full(n_train, 0),
    #                            np.full(n_train, 0), np.full(n_train, 1),
    #                            np.full(n_train, 1), np.full(n_train, 1),
    #                            np.full(n_train, 0), np.full(n_train, 0)], axis=0)
    #drug_full = to_categorical(Drug_full)
    
    # indexes for the training
    # 1-9 is labeled training
    # 11-19 is unlabeled
    # 10 is held out to test
    # 0 is held out
    #idx_train = np.where(np.logical_and(Label_full>0, Label_full!=3))[0]
    #idx_unlab = np.where(Label_full == 3)[0]
    #idx_0 = np.where(Label_full==0)[0]

    idx_train = np.where(np.logical_and(Label_full > 0, Label_full < 5))[0]  #changed 5 to 4
    idx_unlab = np.where(Label_full > 5)[0]   #changed 5 to 4
    #idx_drug = np.where(Drug_full > 0)[0]
    idx_4 = np.where(Label_full == 5)[0]  ##changed 5 to 4
    idx_0 = np.where(Label_full==0)[0]

    ##################################################
    #####. Design the experiment
    ##################################################

    # read in the labeled data
    X_train, Y_train, gene_df = sc_preprocess.read_all_diva_files(args.aug_data_path, idx_range_lab, lab_file_name)
    #genes_df = genes_df.T
    X_train.columns = gene_df

    # only get genes that are available in both testing and training
    common_genes_file = os.path.join(args.aug_data_path, "intersection_genes.pkl")
    gene_out_path = Path(common_genes_file)
    common_genes = pickle.load(open( gene_out_path, "rb" ))
    X_train = X_train[common_genes]
    X_train.head()

    gene_df = gene_df.loc[gene_df.isin(common_genes)]

    # read in the unlabeled data
    #X_train_unlab, Y_train_unlab, gene_df_unlab = sc_preprocess.read_all_diva_files(args.aug_data_path, idx_range_unlab, unlab_file_name)

    # now we need to ensure the genes are in the same order
    #X_train_unlab.columns = gene_df_unlab
    #X_train_unlab = X_train_unlab.reindex(columns=gene_df, fill_value=0)

    # we also need to ensure that the cell-types are in the same order
    #Y_train_unlab = Y_train_unlab.reindex(columns=Y_train.columns, fill_value=0)

    # convert to data matrices
    X_full = X_train.to_numpy()
    Y_full = Y_train.to_numpy()
    
    #X_train_unlab = X_train_unlab.to_numpy()
    #Y_train_unlab = Y_train_unlab.to_numpy()


    # append together the labeled and unlabeled data
    #X_full = np.concatenate((X_full, X_train_unlab), axis=0)
    #Y_full = np.concatenate((Y_full, Y_train_unlab), axis=0)

    ## get the top variable genes
    X_colmean = X_full.mean(axis=0)
    X_colvar = X_full.var(axis=0)
    X_CoV = np.array(np.divide(X_colvar, X_colmean+0.001))
    idx_top = np.argpartition(X_CoV, -args.num_genes)[-args.num_genes:]
    gene_df = gene_df.iloc[idx_top]
    X_full = X_train.loc[:,gene_df]
    X_full = X_full.to_numpy()

    ## normalize within sample
    X_full = scale(X_full, axis=1)     
    print(X_full.shape)

    print(np.where(X_colmean == 0)[0].tolist())


    # for unknown proportions; i.e. 3 
    X_unkp = X_full[idx_unlab,]
    label_unkp = label_full[idx_unlab,]
    #drug_unkp = drug_full[idx_unlab,]
    y_unkp = Y_full[idx_unlab,]
    f = open("got_to_line_178.txt", "w")
    # for known proportions
    X_kp = X_full[idx_train,]
    label_kp = label_full[idx_train,]
    #drug_kp = drug_full[idx_train,]
    y_kp = Y_full[idx_train,]

    # test
    X_0 = X_full[idx_0,]
    label_0 = label_full[idx_0,]
    #drug_0 = drug_full[idx_0,]
    y_0 = Y_full[idx_0,]

    ##################################################
    #####. Hyperparameters
    ##################################################

    batch_size = 500
    n_epoch = 500 # 100 

    alpha_rot = 1000000
    alpha_prop = 100
    #alpha_drug = 1000000

    beta_kl_slack = 10
    beta_kl_rot = 100
    beta_kl_prop = 10
    #beta_kl_drug = 1000


    n_x = X_full.shape[1]
    n_y = Y_full.shape[1]
    n_label = n_tot_samples  # 6 "patients" 1 sample augmented into 6 distinct versions
    #n_drugs = n_drugs  # number of drugs one-hot encoded
    n_label_z = 64  # 64 dimensional representation of rotation
    

    # the network dimensions are 784 > 512 > proportion_dim < 512 < 784
    n_z = Y_full.shape[1] # latent space size, one latent dimension PER cell type
    encoder_dim = 512 # dim of encoder hidden layer
    decoder_dim = 512 # dim of encoder hidden layer
    decoder_out_dim = n_x # dim of decoder output layer

    activ = 'relu'
    optim = Adam(learning_rate=0.0005)

    print(f"length of X {n_x} and length of y {n_y} n_label {n_label}") #and n_drugs {n_drugs}")
    ##################################################
    #####. Train Model first pass
    ##################################################
    known_prop_vae, unknown_prop_vae, encoder, decoder = diva.instantiate_model(n_x=n_x,
                                                            n_y=n_y,
                                                            n_label=n_label,
                                                            #n_drugs=n_drugs,
                                                            n_z=n_z,
                                                            decoder_out_dim = decoder_out_dim,
                                                            n_label_z = n_label_z,
                                                            encoder_dim = encoder_dim,
                                                            decoder_dim = decoder_dim,
                                                            batch_size = batch_size,
                                                            n_epoch = n_epoch,
                                                            alpha_rot = alpha_rot,
                                                            #alpha_drug = alpha_drug,
                                                            alpha_prop = alpha_prop,
                                                            alpha_prop_unk = alpha_prop, ## loss for unknown prop is 0 in initialization
                                                            beta_kl_slack = beta_kl_slack,
                                                            beta_kl_rot = beta_kl_rot,
                                                            beta_kl_prop = beta_kl_prop,
                                                            #beta_kl_drug = beta_kl_drug,
                                                            activ = activ,
                                                            optim = optim)

    X_unkp = np.asarray(X_unkp).astype('float32')
    y_unkp = np.asarray(y_unkp).astype('float32')
    label_unkp = np.asarray(label_unkp).astype('float32')
    #drug_unkp = np.asarray(drug_unkp).astype('float32')
    X_kp = np.asarray(X_kp).astype('float32')
    y_kp = np.asarray(y_kp).astype('float32')
    label_kp = np.asarray(label_kp).astype('float32')
    #drug_kp = np.asarray(drug_kp).astype('float32')
    f = open("got_to_line_261.txt", "w")

    # loss_history = diva.fit_model(known_prop_vae, 
    #                                 unknown_prop_vae,
    #                                 X_unkp,
    #                                 y_unkp,
    #                                 label_unkp,
    #                                 drug_unkp,
    #                                 X_kp, 
    #                                 y_kp,
    #                                 label_kp, 
    #                                 drug_kp, 
    #                                 epochs=n_epoch,
    #                                 batch_size=batch_size)

    loss_history = diva.fit_model_supervised(known_prop_vae, 
                                    X_kp, y_kp,
                                    label_kp, n_epoch, batch_size)
    ##################################################
    #####. Train Model second pass
    ##################################################

    idx_second_run = idx_unlab
    X_unkp = X_full[idx_second_run,]
    label_unkp = label_full[idx_second_run,]
    #drug_unkp = drug_full[idx_second_run,]
    z_slack, mu_slack, l_sigma_slack, mu_prop, l_sigma_prop, prop_outputs, z_rot, mu_rot, l_sigma_rot = encoder.predict(X_unkp, batch_size=batch_size) ##z_drug, mu_drug, l_sigma_drug
    y_unkp_rand = prop_outputs

    X_kp = X_full[idx_train,]
    label_kp = label_full[idx_train,]
    #drug_kp = drug_full[idx_train,]
    y_kp = Y_full[idx_train,]



    # known_prop_vae, unknown_prop_vae, encoder, decoder = diva.instantiate_model(n_x=n_x,
    #                                                         n_y=n_y,
    #                                                         n_label=n_label,
    #                                                         n_drugs=n_drugs,
    #                                                         n_z=n_z,
    #                                                         decoder_out_dim = decoder_out_dim,
    #                                                         n_label_z = n_label_z,
    #                                                         encoder_dim = encoder_dim,
    #                                                         decoder_dim = decoder_dim,
    #                                                         batch_size = batch_size,
    #                                                         n_epoch = n_epoch,
    #                                                         alpha_rot = alpha_rot,
    #                                                         alpha_prop = alpha_prop,
    #                                                         alpha_prop_unk = alpha_prop*0.1,
    #                                                         beta_kl_slack = beta_kl_slack,
    #                                                         beta_kl_rot = beta_kl_rot,
    #                                                         beta_kl_prop = beta_kl_prop,
    #                                                         activ = activ,
    #                                                         optim = optim)


    X_unkp = np.asarray(X_unkp).astype('float32')
    y_unkp_rand = np.asarray(y_unkp_rand).astype('float32')
    label_unkp = np.asarray(label_unkp).astype('float32')
    #drug_unkp = np.asarray(drug_unkp).astype('float32')
    X_kp = np.asarray(X_kp).astype('float32')
    y_kp = np.asarray(y_kp).astype('float32')
    label_kp = np.asarray(label_kp).astype('float32')
    #drug_kp = np.asarray(drug_kp).astype('float32')
    #this is all added to make assert in diva.fitmodel code work
    f = open("got_to_line_328.txt", "w")
    print(f"len of X_known_prop = {len(X_kp)} and len of X_unknown_prop = {len(X_unkp)}")
    print(f"shape of X_known_prop = {X_kp.shape} and shape of X_unknown_prop = {X_unkp.shape}")
    #X_kp = X_kp.T
    #print(f"shape of X_known_prop = {X_kp} and shape of X_unknown_prop = {X_unkp.shape}")
    loss_history = diva.fit_model(known_prop_vae, 
                                    unknown_prop_vae,
                                    X_unkp,
                                    #y_unkp_rand,  10 arguments but only takes 9
                                    label_unkp,
                                    X_kp, 
                                    y_kp,
                                    label_kp, 
                                    n_epoch,      #deleted epochs = n_epochs
                                    batch_size)

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
    #unlabeled_prop_loss = [item[4][2] for item in loss_history]

    labeled_samp_loss = [item[3] for item in loss_history]
    unlabeled_samp_loss = [item[4][2] for item in loss_history]
    
    #changed numbers from 5>to 4 above to match index

    #labeled_drug_loss = [item[4] for item in loss_history]
    #unlabeled_drug_loss = [item[5][4] for item in loss_history]

    # make into a dataframe
    total_loss = labeled_total_loss + unlabeled_total_loss + [a + b for a, b in zip(labeled_total_loss, unlabeled_total_loss)]
    loss_df = pd.DataFrame(data=total_loss, columns=['total_loss'])
    loss_df['type'] = ["labeled"]*len(loss_history) + ["unlabeled"]*len(loss_history) + ["sum"]*len(loss_history)
    loss_df['batch'] = [*range(len(loss_history))] + [*range(len(loss_history))] + [*range(len(loss_history))]

    recon_loss = labeled_recon_loss + unlabeled_recon_loss + [a + b for a, b in zip(labeled_recon_loss, unlabeled_recon_loss)]
    loss_df['recon_loss'] = recon_loss

    
    prop_loss = labeled_prop_loss + [a for a in zip(labeled_prop_loss)]
    #prop_loss = labeled_prop_loss + unlabeled_prop_loss + [a + b for a, b in zip(labeled_prop_loss, unlabeled_prop_loss)]
    #prop_loss = prop_loss.to_frame()
    prop_loss = pd.Series(prop_loss)
    loss_df['prop_loss'] = prop_loss

    samp_loss = labeled_samp_loss + unlabeled_samp_loss + [a + b for a, b in zip(labeled_samp_loss, unlabeled_samp_loss)]
    loss_df['samp_loss'] = samp_loss

    #drug_loss = labeled_drug_loss + unlabeled_drug_loss + [a + b for a, b in zip(labeled_drug_loss, unlabeled_drug_loss)]
    #loss_df['drug_loss'] = drug_loss

    loss_file = os.path.join(args.res_data_path, f"train-{args.exp_id}-{args.unlab_exp_id}-DIVA_loss.pkl")
    loss_df.to_pickle(loss_file)

    # write out the features for testing
    gene_file = os.path.join(args.res_data_path, f"train-{args.exp_id}-{args.unlab_exp_id}-DIVA_features.pkl")
    gene_df.to_pickle(gene_file)

    f = open("finished_file.txt", "w")

    loss_history
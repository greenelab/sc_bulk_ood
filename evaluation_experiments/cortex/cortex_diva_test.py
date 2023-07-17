# import the custom code
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from sc_preprocessing import sc_preprocess

# general imports
import warnings
import numpy as np
import keras as K
from scipy.stats import spearmanr, pearsonr

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

# helper methods for evaluation
def sum_abs_error(y_pred, y_true):
    return sum(abs(y_pred - y_true))

def mean_abs_error(y_pred, y_true):
    return np.mean(abs(y_pred - y_true))

def mean_sqr_error(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)



if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-res", "--res_data_path", dest="res_data_path",
                        help="path to read/write DIVA results")
    parser.add_argument("-aug", "--aug_data_path", dest="aug_data_path",
                        help="path to read pseudobulks")
    parser.add_argument("-train", "--train_id",
                        dest="train_id",
                        help="ID of GSE experiment that was used in training")
    parser.add_argument("-unlab_exp", "--unlab_exp_id",
                        dest="unlab_exp_id",
                        help="ID of GSE experiment to use as the unlabeled value")
    parser.add_argument("-test", "--test_id",
                        dest="test_id",
                        help="ID of GSE experiment that was used in testing")

    args = parser.parse_args()

    # read in trained model 
    encoder = K.models.load_model(f"{args.res_data_path}/{args.train_id}_{args.unlab_exp_id}_encoder")

    # read in the test / train data
    _, Y_train, _, _ = sc_preprocess.read_diva_files(args.aug_data_path, 6, args.train_id, use_test=False)
    X_cortex2, Y_cortex2, cortex2_gene_df, _ = sc_preprocess.read_diva_files(args.aug_data_path, 6, args.test_id, use_test=False)
    gene_file = os.path.join(args.res_data_path, f"train-{args.train_id}-{args.unlab_exp_id}-DIVA_features.pkl")
    gene_path = Path(gene_file)
    gene_df_train = pickle.load( open( gene_path, "rb" ) )

    print(f"Encoder files: {args.res_data_path}/{args.train_id}_{args.unlab_exp_id}_encoder")
    print(f"X shape: {X_cortex2.shape}")
    print(f"Y shape:  {Y_cortex2.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    #X_cortex2 = pd.concat([X_cortex2, X_cortex2, X_cortex2, X_cortex2, X_cortex2])
    #Y_cortex2 = pd.concat([Y_cortex2, Y_cortex2, Y_cortex2, Y_cortex2, Y_cortex2])

    print(f"X shape: {X_cortex2.shape}")
    print(f"Y shape:  {Y_cortex2.shape}")
    print(f"Y_train shape: {Y_train.shape}")


    # now we need to ensure the genes are in the same order
    X_cortex2.columns = cortex2_gene_df
    X_cortex2 = X_cortex2.reindex(columns=gene_df_train, fill_value=0)
    X_cortex2 = X_cortex2[gene_df_train]

    # we also need to ensure that the cell-types are in the same order
    ## need to add somthing to fill in missing columns
    Y_cortex2 = Y_cortex2.reindex(columns=Y_train.columns, fill_value=0)
    Y_cortex2 = Y_cortex2[Y_train.columns]

    # cast from DF to NPArray
    X_cortex2 = X_cortex2.to_numpy()
    Y_cortex2 = Y_cortex2.to_numpy()

    # normalilze within sample
    X_cortex2 = scale(X_cortex2, axis=1)
    Y_cortex2 = Y_cortex2/Y_cortex2.sum(axis=1,keepdims=1)

    # now run it
    batch_size = 500

    z_slack, mu_slack, l_sigma_slack, mu_prop, l_sigma_prop, prop_outputs, z_rot, mu_rot, l_sigma_rot= encoder.predict(X_cortex2, batch_size=batch_size)
    z_test = prop_outputs
    encodings = np.asarray(z_test)
    #encodings = encodings.reshape(X_cortex2.shape[0], n_z)


    test_error = [mean_sqr_error(Y_cortex2[idx], encodings[idx]) 
                    for idx in range(0, X_cortex2.shape[0])]

    print(f"MSqE mean: {np.mean(test_error)}, median: {np.median(test_error)}, max: {max(test_error)}")

    test_error = [spearmanr(Y_cortex2[idx].astype(float), encodings[idx].astype(float))[0]
                    for idx in range(0, X_cortex2.shape[0])]
    print(f"Spearman mean: {np.mean(test_error)}, median: {np.median(test_error)}, max: {max(test_error)}")

    test_error = [pearsonr(Y_cortex2[idx].astype(float), encodings[idx].astype(float))[0]
                    for idx in range(0, X_cortex2.shape[0])]
    print(f"Pearson mean: {np.mean(test_error)}, median: {np.median(test_error)}, max: {max(test_error)}")

    # write out the result
    res = pd.DataFrame(encodings)
    res.columns = Y_train.columns

    res_file = os.path.join(args.res_data_path, f"train-{args.train_id}-test-{args.test_id}-unlab-{args.unlab_exp_id}-DIVA.pkl")
    res.to_pickle(res_file)
    
    f = open("finished_file.txt", "w")    

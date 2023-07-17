#!/usr/bin/env python

# import out preprocessing code
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from sc_preprocessing import sc_preprocess

# general imports
import warnings
import numpy as np
import os, sys
import pandas as pd
from argparse import ArgumentParser

import pickle
import gzip
from pathlib import Path



if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-cs", "--cibersort_path", dest="cibersort_path",
                        help="path to folder to output cibersort data")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to read in augmented cortex files")
    parser.add_argument("-exp", "--exp_id",
                        dest="exp_id",
                        help="ID of GSE experiment to use")
    parser.add_argument("-pidx", "--patient_idx",
                        dest="patient_idx",
                        help="ID of simulated patient to use")
    parser.add_argument("--use_test",
                        dest="use_test",
                        help="use the test data, or the training data",
                        action='store_true')
    parser.add_argument("--no_use_test",
                        dest="use_test",
                        help="use the test data, or the training data",
                        action='store_false')
    args = parser.parse_args()

    # get all the augmented data
    X_train, Y_train, gene_df, sig_df = sc_preprocess.read_diva_files(args.aug_data_path, args.patient_idx, args.exp_id, args.use_test)
    X_train.columns = gene_df

    # now we transpose
    X_train = X_train.transpose()
    X_train.columns = range(X_train.shape[1])

    # now write it out
    test_exp_id = f"{args.exp_id}_test"
    if args.use_test:
        sc_preprocess.write_cs_bp_files(args.cibersort_path, test_exp_id, sig_df, X_train, args.patient_idx)
    else:
        sc_preprocess.write_cs_bp_files(args.cibersort_path, args.exp_id, sig_df, X_train, args.patient_idx)


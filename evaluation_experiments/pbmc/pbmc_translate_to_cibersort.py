#!/usr/bin/env python

# import out preprocessing code
import sys
sys.path.insert(1, '../../')
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
    parser.add_argument("-cs", "--cybersort_path", dest="cybersort_path",
                        help="path to folder to output cybersort data")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to read in augmented PBMC files")
    parser.add_argument("-exp", "--exp_id",
                        dest="exp_id",
                        help="ID of GSE experiment to use")
    parser.add_argument("-pidx", "--patient_idx",
                        dest="patient_idx",
                        help="ID of simulated patient to use")

    args = parser.parse_args()

    # get all the augmented data
    X_train, Y_train, gene_df, sig_df = sc_preprocess.read_diva_files(args.aug_data_path, args.patient_idx, args.exp_id)
    X_train.columns = gene_df["gene_ids"]

    # now we transpose
    X_train = X_train.transpose()
    X_train.columns = range(X_train.shape[1])

    # now write it out
    sc_preprocess.write_cs_bp_files(args.cybersort_path, args.exp_id, sig_df, X_train, args.patient_idx)

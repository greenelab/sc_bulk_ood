#!/usr/bin/env python

# import out preprocessing code
import sys
sys.path.insert(1, '../../')
from sc_preprocessing import sc_preprocess

# general imports
import warnings
import numpy as np
import os
import pandas as pd
import scipy as sp
from scipy.sparse import coo_matrix
from argparse import ArgumentParser

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

import pickle
import gzip
from pathlib import Path

# set seeds
from numpy.random import seed
seed(1)


def read_gbm28_input(count_file, meta_file):

    # read in the counts
    count_ptr = gzip.open(count_file, "r")
    count_matr = pd.read_table(count_ptr)

    # read cell_type labels 
    meta_info = pd.read_table(meta_file, skiprows=[1])
    meta_info.rename(columns = {'NAME':'Name', 'CellAssignment':'CellType'}, inplace = True)

    # transpose the count matr
    count_df = count_matr.transpose()
    count_df.columns = count_df.iloc[0]
    count_df = count_df.drop(count_df.index[0])
    expr_col = count_df.columns

    name_idx = count_df.index
    count_df = np.array(count_df,dtype=np.float32)
    count_df = np.ceil(np.exp(count_df)-1)

    # merge
    count_df = pd.DataFrame(count_df)
    count_df.columns = expr_col
    count_df['Name'] = name_idx
    count_meta_df = count_df.merge(meta_info, left_on=["Name"], right_on=["Name"])


    return (count_meta_df, expr_col)


if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-c", "--count_file", dest="count_file",
                        help="path to GBM26 SS2 count file")
    parser.add_argument("-m", "--meta_file",
                        dest="meta_file",
                        help="path to GBM26 SS2 meta file")
    parser.add_argument("-n", "--num_cells",
                        dest="num_cells", type=int, 
                        help="the number of cells to simulate (recc: 100-5000)")
    parser.add_argument("-samp", "--curr_sample",
                        dest="curr_sample",
                        help="Sample ID to generate pseudobulks")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to write out augmented GBM28 files")

    args = parser.parse_args()

    gene_out_file = os.path.join(args.aug_data_path, f"{args.curr_sample}_genes.pkl")
    sig_out_file = os.path.join(args.aug_data_path, f"{args.curr_sample}_sig.pkl")


    # read in the files
    count_meta_df, expr_col = read_gbm28_input(args.count_file, args.meta_file)

    # filter to sample of interest
    # MGH125 is the sample we test on, SO when we aremaking pseuodbulks from it
    # we need to ensure it only has cells from MGH125
    # all other pseudobulks are made up of their specific tumor cells + a mix of normal cells
    if args.curr_sample == "MGH125":
        count_meta_df_regular_cells = count_meta_df[count_meta_df['Sample'] == args.curr_sample]
    else:
        count_meta_df_regular_cells = count_meta_df[count_meta_df['Sample'] != "MGH125"]

    count_meta_df_regular_cells = count_meta_df_regular_cells[count_meta_df_regular_cells['CellType'] != 'Malignant']

    count_meta_df_tumor_cells = count_meta_df[count_meta_df['Sample'] == args.curr_sample]
    count_meta_df_tumor_cells = count_meta_df_tumor_cells[count_meta_df_tumor_cells['CellType'] == 'Malignant']
    count_meta_df_samp = count_meta_df_regular_cells.append(count_meta_df_tumor_cells)


    # write out the gene ids
    gene_out_path = Path(gene_out_file)
    pickle.dump( expr_col, open( gene_out_path, "wb" ) )

    # write out the final signature matrix we are interested in
    sig_out_path = Path(sig_out_file)
    pickle.dump( count_meta_df_samp, open( sig_out_path, "wb" ) )


    pseudobulk_file = os.path.join(args.aug_data_path, f"{args.curr_sample}_pseudo.pkl")
    prop_file = os.path.join(args.aug_data_path, f"{args.curr_sample}_prop.pkl")

    pseudobulk_path = Path(pseudobulk_file)
    prop_path = Path(prop_file)

    if not pseudobulk_path.is_file(): # skip if we already generated it

        # make the pseudobulks
        prop_df, pseudobulks_df = sc_preprocess.make_prop_and_sum(count_meta_df_samp, 
                                                    expr_col, 
                                                    num_samples=1000, 
                                                    num_cells=None,
                                                    use_true_prop=False)

        # make the proportions instead of cell counts
        prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

        pickle.dump( prop_df, open( prop_path, "wb" ) )
        pickle.dump( pseudobulks_df, open( pseudobulk_path, "wb" ) )


        if not np.all(np.isclose(prop_df.sum(axis=1), 1.)):
            assert False, "Proportions do not sum to 1"

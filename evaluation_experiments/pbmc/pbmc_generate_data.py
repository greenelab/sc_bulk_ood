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


def read_gse_input(cell_file, count_file, gene_file, meta_file, meta_tsne_file):
    cell_info = pd.read_table(cell_file, names=["cell_names"], header=None)
    cell_info["idx"] = range(1,cell_info.shape[0]+1) # we need the index for later -- 1 indexed!

    count_ptr = gzip.open(count_file, "r")
    count_matr = pd.read_table(count_ptr, skiprows=2, names=["gene_idx", "cell_idx", "expr"], sep=" ")

    gene_info = pd.read_table(gene_file, names=["gene_ids"], header=None)

    meta_info = pd.read_table(meta_file)

    # merge metadata with cell_type labels 
    meta_tsne = pd.read_table(meta_tsne_file, skiprows=[1])
    meta_tsne.rename(columns = {'NAME':'Name'}, inplace = True)
    meta_info = meta_info.merge(meta_tsne[["Name", "CellType"]], on="Name")

    return (cell_info, count_matr, gene_info, meta_info)

def format_cell_reads_info(meta_info, cell_info, count_matr, num_cells_expected, method_keep, experiment_keep):
    # first lets take a look at the data
    uniq_methods = meta_info["Method"].unique()
    print(f"Methods used: {uniq_methods}")

    uniq_expr = meta_info["Experiment"].unique()
    print(f"Experiment IDs: {uniq_expr}")

    # now lets get the names of the cells of interest
    names_keep = meta_info[(meta_info["Method"].isin(method_keep)) &
                        (meta_info["Experiment"].isin(experiment_keep))]
    names_keep = names_keep["Name"]

    if names_keep.shape[0] != num_cells_expected:
        assert False, "Incorrect number of cells from names_keep."

    # now get the indices of the cells
    cells_keep = cell_info[cell_info["cell_names"].isin(names_keep)]
    if cells_keep.shape[0] != num_cells_expected:
        assert False, "Incorrect number of cells from cells_keep."

    # using this index, get the count matrix using only the cells of interest
    pbmc1_a_mm = count_matr[count_matr["cell_idx"].isin(cells_keep["idx"])]

    # now format it to the dense repr
    pbmc1_a_dense = pd.crosstab(index=pbmc1_a_mm["gene_idx"],
                                columns=pbmc1_a_mm["cell_idx"],
                                values=pbmc1_a_mm["expr"],
                                aggfunc=sum,
                                dropna=True)
    pbmc1_a_dense = pbmc1_a_dense.fillna(0)
    if pbmc1_a_dense.shape[1] != num_cells_expected:
        assert False, "Incorrect number of cells after reshaping."

    return pbmc1_a_dense

def filter_by_expr(pbmc1_a_dense, min_num_cells, gene_info):
    pbmc1_a_expr = pbmc1_a_dense[pbmc1_a_dense.sum(axis=1) > min_num_cells]
    print(f"Filtered table size: {pbmc1_a_expr.shape}")

    # the pbmc1_a_expr is 1-indexed
    gene_pass_idx = pbmc1_a_expr.index.to_numpy()-1

    gene_pass = gene_info.iloc[gene_pass_idx]
    return (pbmc1_a_expr, gene_pass)

def join_metadata(cell_info, meta_info, pbmc1_a_expr, gene_pass):
    cell_meta_info = cell_info.merge(meta_info, left_on=["cell_names"], right_on=["Name"])

    # first transpose
    pbmc1_a_df = pbmc1_a_expr.transpose()
    pbmc1_a_df.columns = gene_pass["gene_ids"]
    expr_col = pbmc1_a_df.columns

    # now merge
    pbmc1_a_df = cell_meta_info.merge(pbmc1_a_df, left_on=["idx"], right_on=["cell_idx"])
    col_interest = ["CellType"]
    col_interest = col_interest + expr_col.tolist()
    pbmc1_a_df = pbmc1_a_df[col_interest]

    return pbmc1_a_df, expr_col

if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-gse", "--GSE_data_path", dest="GSE_data_path",
                        help="path to folder with GSE132044 data")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to write out augmented PBMC files")
    parser.add_argument("-exp", "--exp_id",
                        dest="exp_id",
                        help="ID of GSE experiment to use")

    args = parser.parse_args()

    # these are pre-defined parameters that are inherent
    # to each experiment
    pbmc1_smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                        "Experiment":'pbmc1', 
                        "min_num_cells":[-1], 
                        "num_cells":[253],
                        "file_id":'pbmc_rep1_sm2'})

    pbmc2_smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                        "Experiment":'pbmc2', 
                        "min_num_cells":[-1], 
                        "num_cells":[273],
                        "file_id":'pbmc_rep2_sm2'})

    pbmc1_10x_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                        "Experiment":'pbmc1', 
                        "min_num_cells":[-1], 
                        "num_cells":[3222],
                        "file_id":'pbmc_rep1_10xV2a'})

    pbmc1_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                        "Experiment":'pbmc1', 
                        "min_num_cells":[-1], 
                        "num_cells":[3222],
                        "file_id":'pbmc_rep1_10xV2a_sm2_cells'})

    pbmc2_10x_param = pd.DataFrame({"Method":'10x Chromium V2', 
                        "Experiment":'pbmc2', 
                        "min_num_cells":[-1], 
                        "num_cells":[3362],
                        "file_id":'pbmc_rep2_10xV2'})

    pbmc2_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2', 
                        "Experiment":'pbmc2', 
                        "min_num_cells":[-1], 
                        "num_cells":[3362],
                        "file_id":'pbmc_rep2_10xV2_sm2_cells'})

    sm2_cell_types = ['CD14+ monocyte', 'Cytotoxic T cell',
                    'CD16+ monocyte', 'B cell',
                    'CD4+ T cell', 'Megakaryocyte']

    #####################
    ### set the study ###
    #####################
    cell_file = os.path.join(args.GSE_data_path, "GSE132044_cells_umi_new.txt")
    count_file = os.path.join(args.GSE_data_path, "GSE132044_counts_umi.txt.gz")
    gene_file = os.path.join(args.GSE_data_path, "GSE132044_genes_umi.txt")
    meta_file = os.path.join(args.GSE_data_path, "GSE132044_meta_counts_new.txt")
    meta_tsne_file = os.path.join(args.GSE_data_path, "GSE132044_meta.txt")
    num_cells_vec = [2000, 3000, 1500, 4000, 3500, 2500, 200, 300, 100, 400, 3500, 2500]


    if args.exp_id == "pbmc_rep1_sm2" :
        curr_study = pbmc1_smart_seq2_param
        cell_file = os.path.join(args.GSE_data_path, "GSE132044_cells_read_new.txt")
        count_file = os.path.join(args.GSE_data_path, "GSE132044_counts_read.txt.gz")
        gene_file = os.path.join(args.GSE_data_path, "GSE132044_genes_read.txt")
        num_cells_vec = [200, 500, 1000, 800, 1400, 900]
    elif args.exp_id == "pbmc_rep2_sm2" :
        curr_study = pbmc2_smart_seq2_param
        cell_file = os.path.join(args.GSE_data_path, "GSE132044_cells_read_new.txt")
        count_file = os.path.join(args.GSE_data_path, "GSE132044_counts_read.txt.gz")
        gene_file = os.path.join(args.GSE_data_path, "GSE132044_genes_read.txt")
        num_cells_vec = [200, 500, 1000, 800, 1400, 900]
    elif args.exp_id == "pbmc_rep1_10xV2a":
        curr_study = pbmc1_10x_param
    elif args.exp_id == "pbmc_rep1_10xV2a_sm2_cells":
        curr_study = pbmc1_10x_sm2_cells_param
    elif args.exp_id == "pbmc_rep2_10xV2":
        curr_study = pbmc2_10x_param
    elif args.exp_id == "pbmc_rep2_10xV2_sm2_cells":
        curr_study = pbmc2_10x_sm2_cells_param
    else:
        assert False, "Unrecognized experiment ID"


    # set the study specific parameters
    min_num_cells = pd.to_numeric(curr_study["min_num_cells"][0])
    method_keep = curr_study["Method"].tolist()
    experiment_keep = curr_study["Experiment"].tolist()
    num_cells_expected = pd.to_numeric(curr_study["num_cells"][0])
    out_file_id = curr_study["file_id"][0]
    gene_out_file = os.path.join(args.aug_data_path, f"{out_file_id}_genes.pkl")
    sig_out_file = os.path.join(args.aug_data_path, f"{out_file_id}_sig.pkl")

    ########################
    ### generate samples ###
    ########################
    # now generate the augmented samples
    cell_info, count_matr, gene_info, meta_info = read_gse_input(cell_file, 
                                                                count_file, 
                                                                gene_file, 
                                                                meta_file, 
                                                                meta_tsne_file)

    pbmc1_a_dense = format_cell_reads_info(meta_info, cell_info, count_matr, 
                                            num_cells_expected, method_keep, 
                                            experiment_keep)

    pbmc1_a_expr, gene_pass = filter_by_expr(pbmc1_a_dense, min_num_cells, gene_info)

    pbmc1_a_df, expr_col = join_metadata(cell_info, meta_info, pbmc1_a_expr)
    print(pbmc1_a_df.head)

    # filter to sm2 cell types if needed
    if args.exp_id == "pbmc_rep1_10xV2a_sm2_cells" or args.exp_id == "pbmc_rep2_10xV2_sm2_cells":
        pbmc1_a_df = pbmc1_a_df[pbmc1_a_df['CellType'].isin(sm2_cell_types)]

    # write out the gene ids
    gene_out_path = Path(gene_out_file)
    pickle.dump( gene_pass, open( gene_out_path, "wb" ) )

    # write out the final signature matrix we are interested in
    sig_out_path = Path(sig_out_file)
    pickle.dump( pbmc1_a_df, open( sig_out_path, "wb" ) )

    # simulate different number of cells
    num_samples = 1000
    for idx in range(len(num_cells_vec)):
        print(f"New Domain {idx}")
        pbmc_rep1_pseudobulk_file = os.path.join(args.aug_data_path, f"{out_file_id}_pseudo_{idx}.pkl")
        pbmc_rep1_prop_file = os.path.join(args.aug_data_path, f"{out_file_id}_prop_{idx}.pkl")

        pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
        prop_path = Path(pbmc_rep1_prop_file)

        if not pseudobulk_path.is_file(): # skip if we already generated it

            # make the pseudobulks
            num_cells = None #num_cells_vec[idx]
            prop_df, pseudobulks_df = sc_preprocess.make_prop_and_sum(pbmc1_a_df, 
                                                        expr_col, 
                                                        num_samples, 
                                                        num_cells,
                                                        use_true_prop=False)

            # make the proportions instead of cell counts
            prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

            pickle.dump( prop_df, open( prop_path, "wb" ) )
            pickle.dump( pseudobulks_df, open( pseudobulk_path, "wb" ) )


            if not np.all(np.isclose(prop_df.sum(axis=1), 1.)):
                assert False, "Proportions do not sum to 1"

    # simulate same number of cells and cell-type proportions
    idx = 9999 # this is a tag that its the "true" proportions
    pbmc_rep1_pseudobulk_file = os.path.join(args.aug_data_path, f"{out_file_id}_pseudo_{idx}.pkl")
    pbmc_rep1_prop_file = os.path.join(args.aug_data_path, f"{out_file_id}_prop_{idx}.pkl")

    pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
    prop_path = Path(pbmc_rep1_prop_file)

    if not pseudobulk_path.is_file(): # skip if we already generated it
        # make the pseudobulks
        prop_df, pseudobulks_df = sc_preprocess.make_prop_and_sum(pbmc1_a_df, 
                                                    expr_col, 
                                                    num_samples, 
                                                    num_cells_expected,
                                                    use_true_prop=True)

        # make the proportions instead of cell counts
        prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

        pickle.dump( prop_df, open( prop_path, "wb" ) )
        pickle.dump( pseudobulks_df, open( pseudobulk_path, "wb" ) )          


# general imports
import warnings
import numpy as np
import os, sys
import pandas as pd
import scprep
import scipy as sp
from scipy.sparse import coo_matrix


import pickle
import gzip
from pathlib import Path

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

def format_cell_reads_info(meta_info, cell_info, count_matr):
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

def filter_by_expr(pbmc1_a_dense, min_num_cells):
    pbmc1_a_expr = pbmc1_a_dense[pbmc1_a_dense.sum(axis=1) > min_num_cells]
    print(f"Filtered table size: {pbmc1_a_expr.shape}")

    # the pbmc1_a_expr is 1-indexed
    gene_pass_idx = pbmc1_a_expr.index.to_numpy()-1

    gene_pass = gene_info.iloc[gene_pass_idx]
    return (pbmc1_a_expr, gene_pass)

def join_metadata(cell_info, sm2_cell_types):
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


    # now we transpose
    pbmc1_a_df = pbmc1_a_df.transpose()

    return pbmc1_a_df

def read_diva_files(data_path, file_idx, file_name):
  pbmc_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_pseudo_{file_idx}.pkl")
  pbmc_rep1_prop_file = os.path.join(data_path, f"{file_name}_prop_{file_idx}.pkl")
  pbmc_rep1_gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")

  pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
  prop_path = Path(pbmc_rep1_prop_file)
  gene_path = Path(pbmc_rep1_gene_file)

  prop_df = pickle.load( open( prop_path, "rb" ) )
  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )

  return (pseudobulks_df, prop_df, gene_df)

def write_cs_bp_files(cybersort_path, out_file_id, pbmc1_a_df, X_train):
    # write out the scRNA-seq signature matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_0_cybersort_sig.tsv.gz")
    sig_out_path = Path(sig_out_file)

    pbmc1_a_df.to_csv(sig_out_path, sep='\t',header=False)

    # write out the bulk RNA-seq mixture matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_0_cybersort_mix.tsv.gz")
    sig_out_path = Path(sig_out_file)

    X_train.to_csv(sig_out_path, sep='\t')



# parameters
GSE_data_path = "/content/drive/MyDrive/Colab Notebooks/checkouts/sc_bulk_ood/data/single_cell_data/GSE132044/"
aug_data_path = "/content/drive/MyDrive/Colab Notebooks/checkouts/sc_bulk_ood/data/single_cell_data/augmented_pbmc_data/"
cybersort_path = "/content/drive/MyDrive/Colab Notebooks/checkouts/sc_bulk_ood/data/single_cell_data/cybersort_pbmc/"

smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                    "Experiment":'pbmc1', 
                    "min_num_cells":[-1], 
                    "num_cells":[253],
                    "file_id":'pbmc_rep1_sm2'})

pbmc1_10x_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                    "Experiment":'pbmc1', 
                    "min_num_cells":[-1], 
                    "num_cells":[3222],
                    "file_id":'pbmc_rep1_10xV2a'})

pbmc2_10x_param = pd.DataFrame({"Method":'10x Chromium V2', 
                    "Experiment":'pbmc2', 
                    "min_num_cells":[-1], 
                    "num_cells":[3362],
                    "file_id":'pbmc_rep2_10xV2'})

pbmc1_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                    "Experiment":'pbmc1', 
                    "min_num_cells":[-1], 
                    "num_cells":[3222],
                    "file_id":'pbmc_rep1_10xV2a_sm2_cells'})
sm2_cell_types = ['CD14+ monocyte', 'Cytotoxic T cell',
                    'CD16+ monocyte', 'B cell',
                    'CD4+ T cell', 'Megakaryocyte']
#####################
### set the study ###
#####################
curr_study = pbmc1_10x_sm2_cells_param

# set the parameters
min_num_cells = pd.to_numeric(curr_study["min_num_cells"][0])
method_keep = curr_study["Method"].tolist()
experiment_keep = curr_study["Experiment"].tolist()
num_cells_expected = pd.to_numeric(curr_study["num_cells"][0])
out_file_id = curr_study["file_id"][0]

if method_keep[0] == "Smart-seq2":
  # raw files
  cell_file = os.path.join(GSE_data_path, "GSE132044_cells_read_new.txt")
  count_file = os.path.join(GSE_data_path, "GSE132044_counts_read.txt.gz")
  gene_file = os.path.join(GSE_data_path, "GSE132044_genes_read.txt")

else:
  # raw files
  cell_file = os.path.join(GSE_data_path, "GSE132044_cells_umi_new.txt")
  count_file = os.path.join(GSE_data_path, "GSE132044_counts_umi.txt.gz")
  gene_file = os.path.join(GSE_data_path, "GSE132044_genes_umi.txt")

meta_file = os.path.join(GSE_data_path, "GSE132044_meta_counts_new.txt")
meta_tsne_file = os.path.join(GSE_data_path, "GSE132044_meta.txt")


cell_info, count_matr, gene_info, meta_info = read_gse_input(cell_file, count_file, gene_file, meta_file, meta_tsne_file)
pbmc1_a_dense = format_cell_reads_info(meta_info, cell_info, count_matr)
pbmc1_a_expr, gene_pass = filter_by_expr(pbmc1_a_dense, min_num_cells)

pbmc1_a_df = join_metadata(cell_info, sm2_cell_types)

# fiter out specific cell types if needed
# if we are only using the cell types shared across all data types
if out_file_id == "pbmc_rep1_10xV2a_sm2_cells":
    pbmc1_a_df = pbmc1_a_df[pbmc1_a_df['CellType'].isin(sm2_cell_types)]

# get all the augmented data
X_train, Y_train, gene_df = read_files(aug_data_path, 0, out_file_id)

X_train.columns = gene_df["gene_ids"]

# now we transpose
X_train = X_train.transpose()
X_train.columns = range(X_train.shape[1])

# now write it out
write_cs_bp_files(cybersort_path, out_file_id, pbmc1_a_df, X_train)
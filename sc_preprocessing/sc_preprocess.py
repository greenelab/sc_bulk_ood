#!/usr/bin/env python

# general imports
import warnings
import numpy as np
import os
import pandas as pd
import scipy as sp
from scipy.sparse import coo_matrix

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# file processing
import pickle
import gzip
from pathlib import Path


# cell type specific pseudobulk
def get_cell_type_sum(in_df, cell_type_id, num_samples, expr_col):
  cell_df = in_df[in_df.CellType.isin([cell_type_id])]
  cell_sample = cell_df[expr_col]
  cell_sample = cell_sample.sample(n=num_samples, replace=True)
  return cell_sample.sum(axis=0)

# method to generate a proportion vector
def gen_prop_vec_lognormal(len_vector, num_cells):

  rand_vec = np.random.lognormal(5, np.random.uniform(1,3), len_vector) # 1

  rand_vec = np.round((rand_vec/np.sum(rand_vec))*num_cells)
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

# method to generate true proportion vector
def true_prop_vec(in_df, num_cells):

  rand_vec = in_df['CellType'].value_counts() / in_df.shape[0]
  rand_vec = np.array(rand_vec)

  rand_vec = np.round(rand_vec*num_cells)
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

# total pseudobulk
def make_prop_and_sum(in_df, expr_col, num_samples, num_cells, use_true_prop):
  len_vector = in_df.CellType.unique().shape[0]

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = expr_col)
  total_prop = pd.DataFrame(columns = in_df.CellType.unique())

  # sample specific noise
  sample_noise = np.random.lognormal(0, 0.1, expr_col.shape[0])

  # cell specific noise
  cell_noise = [np.random.lognormal(0, 0.1, expr_col.shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(num_samples):
    if samp_idx % 100 == 0:
      print(samp_idx)

    n_cells = num_cells
    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)

    if use_true_prop:
      props_vec = true_prop_vec(in_df, n_cells)
    else:
      props_vec = gen_prop_vec_lognormal(len_vector, n_cells)
    props = pd.DataFrame(props_vec)
    props = props.transpose()
    props.columns = in_df.CellType.unique()
    total_prop = total_prop.append(props)

    sum_over_cells = np.zeros(expr_col.shape[0])


    #iterate over all the cell types
    for cell_idx in range(len_vector):
      cell_type_id = in_df.CellType.unique()[cell_idx]
      num_cell = props_vec[cell_idx]
      ct_sum = get_cell_type_sum(in_df, cell_type_id, num_cell, expr_col)

      # add noise
      if not use_true_prop:
        ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])
      sum_over_cells = sum_over_cells + ct_sum

    sum_over_cells = pd.DataFrame(sum_over_cells)
    sum_over_cells = sum_over_cells.transpose()
    sum_over_cells.columns = expr_col

    # add sample noise
    if not use_true_prop:
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)

    total_expr = total_expr.append(sum_over_cells)


  return (total_prop, total_expr)


def read_diva_files(data_path, file_idx, file_name):

    if file_idx is not None:
      pbmc_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_pseudo_{file_idx}.pkl")
      pbmc_rep1_prop_file = os.path.join(data_path, f"{file_name}_prop_{file_idx}.pkl")
    else:
      pbmc_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_pseudo.pkl")
      pbmc_rep1_prop_file = os.path.join(data_path, f"{file_name}_prop.pkl")

    pbmc_rep1_gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
    pbmc_rep1_sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

    pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
    prop_path = Path(pbmc_rep1_prop_file)
    gene_path = Path(pbmc_rep1_gene_file)
    sig_path = Path(pbmc_rep1_sig_file)

    prop_df = pickle.load( open( prop_path, "rb" ) )
    pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
    gene_df = pickle.load( open( gene_path, "rb" ) )
    sig_df = pickle.load( open( sig_path, "rb" ) )

    return (pseudobulks_df, prop_df, gene_df, sig_df)

def read_all_diva_files(data_path, idx_range, file_name):

    X_concat = None

    for idx in idx_range:
        X_train, Y_train, gene_df, _ = read_diva_files(data_path, idx, file_name)
        X_train.columns = gene_df["gene_ids"]

        if X_concat is None:
            X_concat, Y_concat = X_train, Y_train
        else:
            X_concat = pd.concat([X_concat, X_train])
            Y_concat = pd.concat([Y_concat, Y_train])

    return (X_concat, Y_concat, gene_df)


def write_cs_bp_files(cybersort_path, out_file_id, pbmc1_a_df, X_train, patient_idx=0):
    # write out the scRNA-seq signature matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_{patient_idx}_cybersort_sig.tsv.gz")
    sig_out_path = Path(sig_out_file)
    pbmc1_a_df = pbmc1_a_df.transpose()

    pbmc1_a_df.to_csv(sig_out_path, sep='\t',header=False)

    # write out the bulk RNA-seq mixture matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_{patient_idx}_cybersort_mix.tsv.gz")
    sig_out_path = Path(sig_out_file)

    X_train.to_csv(sig_out_path, sep='\t',header=True)

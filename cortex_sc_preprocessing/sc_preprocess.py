

#!/usr/bin/env python

# general imports
import warnings
import numpy as np
import os
import pandas as pd
import sklearn as sk
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
def get_cell_type_sum(in_adata, cell_type_id, num_samples):

  # get the expression of the cells of interest
  cell_df = in_adata[in_adata.obs["scpred_CellType"].isin([cell_type_id])]

  # now to the sampling
  cell_sample = sk.utils.resample(cell_df, n_samples = num_samples, replace=True)

  # add  poisson noise
  #dense_X = cell_sample.X.todense()
  #noise_mask = np.random.poisson(dense_X+1)
  #dense_X = dense_X + noise_mask

  sum_per_gene = cell_sample.X.sum(axis=0)


  return sum_per_gene

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
def true_prop_vec(in_adata, num_cells):

  rand_vec = in_adata.obs["scpred_CellType"].value_counts() / in_adata.obs["scpred_CellType"].shape[0]
  rand_vec = np.array(rand_vec)

  rand_vec = np.round(rand_vec*num_cells)
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

# total pseudobulk
def make_prop_and_sum(in_adata, num_samples, num_cells, use_true_prop, cell_noise):
  len_vector = in_adata.obs["scpred_CellType"].unique().shape[0]

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  total_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())

  test_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  test_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())

  # sample specific noise
  sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])  # 0.1


  # cell specific noise, new noise for each sample
  if cell_noise == None:
    cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(num_samples+100):
    if samp_idx % 100 == 0:
      print(samp_idx)

    n_cells = num_cells
    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)

    if use_true_prop:
      props_vec = true_prop_vec(in_adata, n_cells)
    else:
      props_vec = gen_prop_vec_lognormal(len_vector, n_cells)
    props = pd.DataFrame(props_vec)
    props = props.transpose()
    props.columns = in_adata.obs["scpred_CellType"].unique()

    sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

    #iterate over all the cell types
    for cell_idx in range(len_vector):
      cell_type_id = in_adata.obs["scpred_CellType"].unique()[cell_idx]
      num_cell = props_vec[cell_idx]
      ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

      # add noise if we don't want the true proportions
      if not use_true_prop:
        ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

      sum_over_cells = sum_over_cells + ct_sum


    sum_over_cells = pd.DataFrame(sum_over_cells)
    sum_over_cells.columns = in_adata.var['gene_ids']

    # add sample noise
    if not use_true_prop:
      # sample noise
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      sum_over_cells = np.random.poisson(sum_over_cells)[0]


    sum_over_cells = pd.DataFrame(sum_over_cells)
    sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']

    if samp_idx < num_samples:
      total_prop = total_prop.append(props)
      total_expr = total_expr.append(sum_over_cells)
    else:
      test_prop = test_prop.append(props)
      test_expr = test_expr.append(sum_over_cells)


  return (total_prop, total_expr, test_prop, test_expr)


def read_diva_files(data_path, file_idx, file_name, use_test=False):

    pseudo_str = "pseudo"
    prop_str = "prop"
    if use_test:
      pseudo_str = "testpseudo"
      prop_str = "testprop"
     

    if file_idx is not None:
      cortex_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_{pseudo_str}_{file_idx}.pkl")
      cortex_rep1_prop_file = os.path.join(data_path, f"{file_name}_{prop_str}_{file_idx}.pkl")
    else:
      cortex_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_pseudo.pkl")
      cortex_rep1_prop_file = os.path.join(data_path, f"{file_name}_prop.pkl")

    cortex_rep1_gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
    cortex_rep1_sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

    pseudobulk_path = Path(cortex_rep1_pseudobulk_file)
    prop_path = Path(cortex_rep1_prop_file)
    gene_path = Path(cortex_rep1_gene_file)
    sig_path = Path(cortex_rep1_sig_file)

    prop_df = pickle.load( open( prop_path, "rb" ) )
    pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
    gene_df = pickle.load( open( gene_path, "rb" ) )
    sig_df = pickle.load( open( sig_path, "rb" ) )

    return (pseudobulks_df, prop_df, gene_df, sig_df)

def read_all_diva_files(data_path, idx_range, file_name):

    X_concat = None

    for idx in idx_range:
        X_train, Y_train, gene_df, _ = read_diva_files(data_path, idx, file_name)
        X_train.columns = gene_df

        if X_concat is None:
            X_concat, Y_concat = X_train, Y_train
        else:
            X_concat = pd.concat([X_concat, X_train])
            Y_concat = pd.concat([Y_concat, Y_train])
            
      
    return (X_concat, Y_concat, gene_df)


def write_cs_bp_files(cybersort_path, out_file_id, cortex_a_df, X_train, patient_idx=0):
    # write out the scRNA-seq signature matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_{patient_idx}_cybersort_sig.tsv.gz")
    sig_out_path = Path(sig_out_file)
    cortex_a_df = cortex_a_df.transpose()

    # cast from matrix to pd
    cortex_a_df = pd.DataFrame(cortex_a_df)

    cortex_a_df.to_csv(sig_out_path, sep='\t',header=False)

    # write out the bulk RNA-seq mixture matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_{patient_idx}_cybersort_mix.tsv.gz")
    sig_out_path = Path(sig_out_file)

    X_train.to_csv(sig_out_path, sep='\t',header=True)

import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from sc_preprocessing import sc_preprocess


# general imports
import click
import warnings
import numpy as np
import os
import pandas as pd
import scanpy as sc
import pickle
from pathlib import Path





@click.command()
@click.option('--aug_data_path', required=True, help='Directory to put the generated pseudobulk data for BuDDI')
@click.option('--data_path', required=True, help='Directory to where the reference single cell profiles')        
@click.option('--scpred_path', required=True, help='Directory to cell-type labels for the reference single cell profiles')        
@click.option('--num_cells_vec', required=True, help='String that will be split on commas to generate a vector with the number of cells to put in each sample. Length of the vector indicate the number of samples.')        
@click.option('--perturb_vec', required=True, help='String that will be split on commas to generate a vector with the celltypes present in the perturbed samples')        
@click.option('--nonperturb_vec', required=True, help='String that will be split on commas to generate a vector with the celltypes present in the non-perturbed samples')        
@click.option('--res_name', required=True, help='Prefix for the output files')        
@click.option('--in_name', required=True, help='Prefix used for the input files')        
def gen_data(   
                aug_data_path, data_path,
                scpred_path, num_cells_vec, perturb_vec, nonperturb_vec,
                res_name, in_name):

    """Generate data where two closely related cell types are swapped to simulate a perturbation"""
    
    # read in the data
    adata = sc.read_10x_mtx(
                                data_path,                  # the directory with the `.mtx` file
                                var_names='gene_symbols',   # use gene symbols for the variable names (variables-axis index)
                                cache=True)                 # write a cache file for faster subsequent reading
    adata.var_names_make_unique()

    # split the strings into a vector
    num_cells_vec = num_cells_vec.split(',')
    num_cells_vec = [int(i) for i in num_cells_vec]
    
    perturb_vec = perturb_vec.split(',')
    nonperturb_vec = nonperturb_vec.split(',')


    # get the perturbed and non-perturbed cell-types
    perturbed_cell_type = np.setdiff1d(perturb_vec, nonperturb_vec)
    nonperturbed_cell_type = np.setdiff1d(nonperturb_vec, perturb_vec)

    # add metadata
    meta_data = pd.read_csv(f"{scpred_path}/{in_name}_scpred.tsv", sep="\t", index_col='code')
    barcodes = pd.read_csv(f"{data_path}/barcodes.tsv", header=None, names=['code'])
    meta_df = barcodes.join(other=meta_data, on=['code'], how='left', sort=False)
    adata.obs['scpred_CellType'] = meta_df['scpred_prediction'].tolist()

    # filter out cells with less than 200 genes and genes expressed in less than 3 cells
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # remove genes with high mitochondrial content
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # remove cells with more than 2000 genes
    # remove cells with more than 7% MTgenes
    adata = adata[adata.obs.n_genes_by_counts < 2000, :]
    adata = adata[adata.obs.pct_counts_mt < 7, :]


    # normalize to 10K counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # remove cells that are unlabeled or unclassified
    cell_type_id = np.unique(adata.obs["scpred_CellType"].values)
    cell_type_remove = ["unassigned", "unclassified"]
    cell_type_id = set(cell_type_id).difference(set(cell_type_remove))
    adata = adata[adata.obs["scpred_CellType"].isin(cell_type_id)]

    # group together cell types that are not very frequent
    all_vals = adata.obs["scpred_CellType"].to_list()
    all_vals = np.char.replace(all_vals, 'adc', 'other')
    all_vals = np.char.replace(all_vals, 'pdc', 'other')
    all_vals = np.char.replace(all_vals, 'mk', 'other')
    all_vals = np.char.replace(all_vals, 'hsc', 'other')
    adata.obs["scpred_CellType"] = all_vals

    # get the non-perturbed reference profiles
    # we do this for CIBERSORT and BP
    adata_nonperturb = adata[adata.obs["scpred_CellType"].isin(nonperturb_vec)]
    all_vals = adata_nonperturb.obs["scpred_CellType"].to_list()
    all_vals = [(x if x != nonperturbed_cell_type[0] else 'collapsed_celltype') for x in all_vals]
    adata_nonperturb.obs["scpred_CellType"] = all_vals

    # make it dense for BP and CIBERSORT
    dense_matrix = adata_nonperturb.X.todense()

    # write it out
    sc_profile_file = os.path.join(aug_data_path, f"{res_name}_sig.pkl")
    sc_profile_path = Path(sc_profile_file)
    dense_df = pd.DataFrame(dense_matrix, columns = adata_nonperturb.var['gene_ids'])
    dense_df.insert(loc=0, column='scpred_CellType', value=adata_nonperturb.obs["scpred_CellType"].to_list())
    pickle.dump( dense_df, open( sc_profile_path, "wb" ) )

    ########### Make Pseudobulks

    ## set up the cell-noise perturbations
    len_vector = len(perturb_vec)
    cell_noise = [np.random.lognormal(0, 0.1, adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

    ## get the perturbed reference profiles 
    adata_perturb = adata[adata.obs["scpred_CellType"].isin(perturb_vec)]

    all_vals = adata_perturb.obs["scpred_CellType"].to_list()
    all_vals = [(x if x != perturbed_cell_type[0] else 'collapsed_celltype') for x in all_vals]
    adata_perturb.obs["scpred_CellType"] = all_vals

    # write out the gene ids
    gene_pass = adata.var['gene_ids']
    gene_out_file = os.path.join(aug_data_path, f"{res_name}_genes.pkl")
    gene_out_path = Path(gene_out_file)
    pickle.dump( gene_pass, open( gene_out_path, "wb" ) )


    # simulate different number of cells
    num_samples = 1000
    for idx in range(len(num_cells_vec)):
        print(f"New Domain {idx}")
        pbmc_rep1_pseudobulk_file = os.path.join(aug_data_path, f"{res_name}_pseudo_{idx}.pkl")
        pbmc_rep1_prop_file = os.path.join(aug_data_path, f"{res_name}_prop_{idx}.pkl")
        test_pbmc_rep1_pseudobulk_file = os.path.join(aug_data_path, f"{res_name}_testpseudo_{idx}.pkl")
        test_pbmc_rep1_prop_file = os.path.join(aug_data_path, f"{res_name}_testprop_{idx}.pkl")

        # if unlabeled data, we need to have both perturbed and non-perturbed cell types
        # idx 0 is for testing and will be perturbed
        # idx 1-4 are not perturbed because they are labeled
        # idx 5 is ignored for now, we will make it perturbed
        # idx 6-9 are unlabeled and will have be perturbed half not be
        # so this makes idx 5,6,7 perturbed
        # idx 8,9 unperturbed
        # summary:
        # perturbed: 0,5,6,7
        # unperturb: 1,2,3,4,8,9
        # train: 1,2,3,4,6,7,8,9
        # test:  6
        # labeled: 1,2,3,4
        # unlabeled: 6,7,8,9

        perturbed_idx = np.array([0,5,6,7])
        curr_adata = adata_nonperturb
        if idx in perturbed_idx:
            curr_adata = adata_perturb

        pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
        prop_path = Path(pbmc_rep1_prop_file)
        test_pseudobulk_path = Path(test_pbmc_rep1_pseudobulk_file)
        test_prop_path = Path(test_pbmc_rep1_prop_file)

        if not pseudobulk_path.is_file(): # skip if we already generated it

            # make the pseudobulks
            num_cells = num_cells_vec[idx]
            prop_df, pseudobulks_df, test_prop_df, test_pseudobulks_df = sc_preprocess.make_prop_and_sum(curr_adata, 
                                                                                    num_samples, 
                                                                                    num_cells,
                                                                                    use_true_prop=False,
                                                                                    cell_noise=cell_noise)

            # make the proportions instead of cell counts
            prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)
            test_prop_df = test_prop_df.div(test_prop_df.sum(axis=1), axis=0)

            pickle.dump( prop_df, open( prop_path, "wb" ) )
            pickle.dump( pseudobulks_df, open( pseudobulk_path, "wb" ) )

            pickle.dump( test_prop_df, open( test_prop_path, "wb" ) )
            pickle.dump( test_pseudobulks_df, open( test_pseudobulk_path, "wb" ) )


            if not np.all(np.isclose(prop_df.sum(axis=1), 1.)):
                assert False, "Proportions do not sum to 1"




if __name__ == '__main__':
    gen_data()
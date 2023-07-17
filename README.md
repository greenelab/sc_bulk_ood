This project is done as part of a rotation project in the Greene Lab. It uses greenelab/sc_bulk_ood by Natalie Davidson code (created for single cell RNA-seq data from blood) and applies it into cortex tissue. 

Project background:
The aim of this project isto find if BuDDi can predict cell type proportions and reproduce cell type gene expression differences in sc-RNAseq Trisomy 21. We're validating BuDDi's bulk deconvolution in Control (D21) samples by training the variatonal autoencoder to in Ctlr scRNA-seq (pseudobulks), and testing in real Ctrl bulk data. By comparing single cell RNA-seq data to RNA-seq bulk data, we can get an insight into disease-specific heterogeneity observed in T21. 

Project code used:
Within the cortex_sc_preprocessing folder:
-pseudoblks_6k.ipynb to generate "pseudobulks" from scRNAseq cortex data to train BuDDi (autoencoder in diva.py).
-It used cleaningup_intersections.ipynb to create an gene_intersections.pkl file for BuDDi.
-It used sc_preprocess for function definitions used throughout other files.
Within the evalutation_experiments/cortex/ folder:
-cortex_diva_train.py and cortex_diva_test.py to train and test Buddi using the run and test cortex_diva_test shell scripts.
-checking_plots_pseudos* scripts were used to visualize tsne plots, loss curves, etc.


README.md from sc_bulk_ood:
# sc_bulk_ood

This project is still a work in progress. If you are interested in how the model works
on various types of input datasets (MNIST or simulated Bulk RNASeq), please refer to the experiments folder.

Full demos of the model, along with ways to train and test on larger real datasets will be added soon.


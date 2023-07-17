# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

res_path="${work_dir}/../../results/single_cell_data/diva_cortex/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_cortex_data/"
num_genes=5000

py_script="python cortex_diva_train.py -res ${res_path} -aug ${aug_data_path} -n ${num_genes}"

# now run this for all out experimental desires

exp_id="cortex6k"
unlab_exp_id="cortex6k_500epoch_decLR"
lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
bsub -R "rusage[mem=45GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}

# exp_id="pbmc_rep2_10xV2_sm2_cells"
# unlab_exp_id="pbmc_rep2_sm2"
# lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
# bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}

# exp_id="pbmc_rep2_10xV2_sm2_cells"
# unlab_exp_id="pbmc_rep1_sm2"
# lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
# bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}


# exp_id="pbmc_rep2_10xV2_sm2_cells"
# unlab_exp_id="NONE"
# lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
# bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}


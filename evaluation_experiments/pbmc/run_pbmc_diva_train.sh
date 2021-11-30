# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/diva_pbmc/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_pbmc_data/"
num_genes=1000

py_script="python pbmc_diva_train.py -res ${res_path} -aug ${aug_data_path} -n ${num_genes} "

# now run this for all out experimental desires

exp_id="pbmc_rep2_10xV2_sm2_cells"
unlab_exp_id="pbmc_rep2_10xV2_sm2_cells"
lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}

exp_id="pbmc_rep2_10xV2_sm2_cells"
unlab_exp_id="pbmc_rep2_sm2"
lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}

exp_id="pbmc_rep2_10xV2_sm2_cells"
unlab_exp_id="pbmc_rep1_sm2"
lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}


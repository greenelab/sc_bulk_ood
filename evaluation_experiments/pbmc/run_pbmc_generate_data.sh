

# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

GSE_data_path="${work_dir}/../../data/single_cell_data/GSE132044/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_pbmc_data/"

py_script="python get_features.py -gse ${GSE_data_path} -aug ${aug_data_path}"

# write out the features of interest
lsf_file=${aug_data_path}/get_features.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J get_features ${py_script}

py_script="python pbmc_generate_data.py -gse ${GSE_data_path} -aug ${aug_data_path} -exp "

# now run this for all out experimental desires
exp_id="pbmc_rep1_sm2"
lsf_file=${aug_data_path}/${exp_id}_generate_data.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

exp_id="pbmc_rep2_sm2"
lsf_file=${aug_data_path}/${exp_id}_generate_data.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


exp_id="pbmc_rep1_10xV2a"
lsf_file=${aug_data_path}/${exp_id}_generate_data.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

exp_id="pbmc_rep1_10xV2a_sm2_cells"
lsf_file=${aug_data_path}/${exp_id}_generate_data.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


exp_id="pbmc_rep2_10xV2"
lsf_file=${aug_data_path}/${exp_id}_generate_data.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


exp_id="pbmc_rep2_10xV2_sm2_cells"
lsf_file=${aug_data_path}/${exp_id}_generate_data.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}



# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

cybersort_path="${work_dir}/../../data/single_cell_data/cybersort_pbmc/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_pbmc_data/"
py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 0 -exp "

# now run this for all out experimental desires
exp_id="pbmc_rep1_sm2"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

exp_id="pbmc_rep2_sm2"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


exp_id="pbmc_rep1_10xV2a"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

exp_id="pbmc_rep1_10xV2a_sm2_cells"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


exp_id="pbmc_rep2_10xV2"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


exp_id="pbmc_rep2_10xV2_sm2_cells"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


#### a new patient id
py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 6 -exp "

exp_id="pbmc_rep2_10xV2_sm2_cells"
lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

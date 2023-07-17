

# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

cibersort_path="${work_dir}/../../data/single_cell_data/cibersort_cortex/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_cortex_data/"
py_script="python pbmc_translate_to_cibersort.py -cs ${cibersort_path} -aug ${aug_data_path} -pidx 6 --no_use_test  -exp "

# now run this for all out experimental desires


py_script="python pbmc_translate_to_cibersort.py -cs ${cibersort_path} -aug ${aug_data_path} -pidx 1 --no_use_test  -exp "
exp_id="cortex6k"
lsf_file=${cibersort_path}/${exp_id}_1_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}



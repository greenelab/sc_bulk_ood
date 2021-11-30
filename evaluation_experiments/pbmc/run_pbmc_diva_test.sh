# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/diva_pbmc"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_pbmc_data/"

py_script="python pbmc_diva_test.py -res ${res_path} -aug ${aug_data_path}"

# now run this for all our experimental desires

# just noise
train_id="pbmc_rep2_10xV2_sm2_cells"
test_id="pbmc_rep2_10xV2_sm2_cells"
unlab_exp_id="pbmc_rep2_10xV2_sm2_cells"
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=2GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}

# across bioreps
train_id="pbmc_rep2_10xV2_sm2_cells"
test_id="pbmc_rep1_10xV2a_sm2_cells"
unlab_exp_id="pbmc_rep2_10xV2_sm2_cells"
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=2GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}

# across tech and biorep
# unlabeled data is same as labeled
train_id="pbmc_rep2_10xV2_sm2_cells"
test_id="pbmc_rep1_sm2"
unlab_exp_id="pbmc_rep2_10xV2_sm2_cells"
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=2GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}


# unlabeled data is different biorep
train_id="pbmc_rep2_10xV2_sm2_cells"
test_id="pbmc_rep1_sm2"
unlab_exp_id="pbmc_rep2_sm2"
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=2GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}


# unlabeled data is same biorep as target biorep
train_id="pbmc_rep2_10xV2_sm2_cells"
test_id="pbmc_rep1_sm2"
unlab_exp_id="pbmc_rep1_sm2"
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=2GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}

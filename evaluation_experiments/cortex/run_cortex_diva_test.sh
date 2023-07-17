# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/diva_cortex"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_cortex_data/"

py_script="python cortex_diva_test.py -res ${res_path} -aug ${aug_data_path}"

# now run this for all our experimental desires

# just noise
train_id="cortex6k"
test_id="cortex6k"
unlab_exp_id="cortex6k_500epoch_decLR"
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=10GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}

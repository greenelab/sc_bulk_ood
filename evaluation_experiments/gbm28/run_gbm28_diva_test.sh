# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/diva_gbm28"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_gbm28_data/"

py_script="python gbm28_diva_test.py -res ${res_path} -aug ${aug_data_path}"

# now run this for all our experimental desires
test_id="MGH125"
ref_id="MGH125"
curr_py_script="${py_script} -test ${test_id} -ref ${ref_id}"
lsf_file=${res_path}/test-${test_id}.lsf
bsub -R "rusage[mem=6GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${test_id} ${curr_py_script}

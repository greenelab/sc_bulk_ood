# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/diva_gbm28/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_gbm28_data/"
num_genes=1000

py_script="python gbm28_diva_train.py -res ${res_path} -aug ${aug_data_path} -n ${num_genes}"

lsf_file=${res_path}/diva_train.lsf
bsub -R "rusage[mem=100GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J gbm28 ${py_script} 

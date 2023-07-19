# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/buddiM2_kang/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_kang_data/"
cibersort_file_path="${work_dir}/../../results/single_cell_data/cibersort_kang/CIBERSORTx_Job17_kang_1_cybersort_sig_inferred_phenoclasses.CIBERSORTx_Job17_kang_1_cybersort_sig_inferred_refsample.bm.K999.txt"

num_genes=7000

py_script="python kang_buddiM2_train.py -res ${res_path} -aug ${aug_data_path} -n ${num_genes} -cib_genes ${cibersort_file_path}"

# now run this for all out experimental desires

exp_id="kang"
lsf_file=${res_path}/${exp_id}_buddiM2_train.lsf
bsub -R "rusage[mem=30GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id}

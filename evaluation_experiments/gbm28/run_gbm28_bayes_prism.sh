
# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}
r_script="Rscript ${work_dir}/../../bayesprism/bayes_prism_gbm28_test.R "
num_samp=20 #1000
ncores=20

in_dir=${work_dir}/../../data/single_cell_data/cybersort_gbm28/
out_dir=${work_dir}/../../results/single_cell_data/bp_gbm28_mini/


file_id_test="MGH125_0"
run_id="bp_gbm28"
lsf_file=${out_dir}/${file_id_test}_final.lsf
bsub -R "rusage[mem=64GB]" -W 24:00 -n 20 -q "normal" -o ${lsf_file} -J ${run_id} ${r_script} ${in_dir} ${out_dir} ${file_id_test} ${num_samp} ${ncores}


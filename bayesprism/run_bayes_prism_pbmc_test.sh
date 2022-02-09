
work_dir=${PWD}
r_script="Rscript ${work_dir}/bayesprism/bayes_prism_pbmc_test.R "

in_dir=${work_dir}/data/single_cell_data/cybersort_pbmc/
out_dir=${work_dir}/results/single_cell_data/bp_pbmc/


file_id_train=pbmc_rep2_10xV2
file_id_test=pbmc_rep2_10xV2
run_id="same_samp"
lsf_file=${out_dir}/${file_id_train}_${file_id_test}_final.lsf
bsub -R "rusage[mem=16GB]" -W 24:00 -n 10 -q "normal" -o ${lsf_file} -J ${run_id} ${r_script} ${in_dir} ${out_dir} ${file_id_train} ${file_id_test} ${num_samp} ${ncores}


file_id_train=pbmc_rep2_10xV2
file_id_test=pbmc_rep1_10xV2a
run_id="diff_samp"
lsf_file=${out_dir}/${file_id_train}_${file_id_test}_final.lsf
bsub -R "rusage[mem=16GB]" -W 24:00 -n 10 -q "normal" -o ${lsf_file} -J ${run_id} ${r_script} ${in_dir} ${out_dir} ${file_id_train} ${file_id_test} ${num_samp} ${ncores}


file_id_train=pbmc_rep2_10xV2
file_id_test=pbmc_rep1_10xV2a
run_id="diff_tech"
lsf_file=${out_dir}/${file_id_train}_${file_id_test}_final.lsf
bsub -R "rusage[mem=16GB]" -W 24:00 -n 10 -q "normal" -o ${lsf_file} -J ${run_id} ${r_script} ${in_dir} ${out_dir} ${file_id_train} ${file_id_test} ${num_samp} ${ncores}


file_id_train=pbmc_rep2_10xV2
file_id_test=pbmc_rep1_10xV2a
run_id="diff_tech_6cells"
lsf_file=${out_dir}/${file_id_train}_${file_id_test}_final.lsf
bsub -R "rusage[mem=16GB]" -W 24:00 -n 10 -q "normal" -o ${lsf_file} -J ${run_id} ${r_script} ${in_dir} ${out_dir} ${file_id_train} ${file_id_test} ${num_samp} ${ncores}

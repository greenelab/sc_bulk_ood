
work_dir=${PWD}
r_script="Rscript ${work_dir}/bayesprism/bayes_prism_pbmc_test.R "

in_dir=${work_dir}/data/single_cell_data/cybersort_pbmc/
out_dir=${work_dir}/results/single_cell_data/bp_pbmc/
file_id_train=pbmc_rep2_10xV2
file_id_test=pbmc_rep2_10xV2
samp_array=(10 25 50 100 250 500 1000)


# using a 10 core -- 64 G
samp_array=(10 50 100 250 500 1000)
ncores=10
for num_samp in "${samp_array[@]}"
do
    lsf_file=${out_dir}/${file_id_train}_${file_id_test}_${num_samp}_${ncores}_16G_10LSFcores.lsf
    bsub -R "rusage[mem=64GB]" -W 24:00 -n 10 -q "normal" -o ${lsf_file} -J ${num_samp} ${r_script} ${in_dir} ${out_dir} ${file_id_train} ${file_id_test} ${num_samp} ${ncores}
done


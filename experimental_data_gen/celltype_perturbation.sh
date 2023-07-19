# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../results/single_cell_data/diva_pbmc/"
aug_data_path="${work_dir}/../data/single_cell_data/augmented_pbmc_data/"
data_path="${work_dir}/../data/single_cell_data/pbmc6k/hg19/"
scpred_path="${work_dir}/../results/single_cell_data/pbmc_cell_labels/"
cybersort_path="${work_dir}/../data/single_cell_data/cybersort_pbmc/"
bp_path="${work_dir}/../results/single_cell_data/bp_pbmc/"

num_cells_vec="5000,5000,5000,5000,5000,5000,5000,5000,5000,5000"


num_genes=5000



##########################################
####### MONOCYTE EXPERIMENT #########
##########################################

mono_perturb_vec="other,mono16,b,cd4_cd8_naive,cd4,cd8,nk"
mono_nonperturb_vec="other,mono14,b,cd4_cd8_naive,cd4,cd8,nk"
res_name="pbmc6k-mono"
in_name="pbmc6k"
test_id="pbmc6k-mono_6"


##########################################
####### TCELL EXPERIMENT #########
##########################################

mono_perturb_vec="other,mono14,mono16,b,cd4_cd8_naive,cd8,nk"
mono_nonperturb_vec="other,mono14,mono16,b,cd4_cd8_naive,cd4,nk"
res_name="pbmc6k-tcell"
in_name="pbmc6k"
test_id="pbmc6k-tcell_6"


##########################################
####### NK EXPERIMENT #########
##########################################

mono_perturb_vec="other,mono14,mono16,b,cd4_cd8_naive,cd4,cd8"
mono_nonperturb_vec="other,mono14,mono16,b,cd4_cd8_naive,cd4,nk"
res_name="pbmc6k-nk"
in_name="pbmc6k"
test_id="pbmc6k-nk_6"



#######
## simulate pseudobulks
#######
py_script="python celltype_perturbation.py --aug_data_path ${aug_data_path} --data_path ${data_path} --scpred_path ${scpred_path} --num_cells_vec ${num_cells_vec}"


lsf_file=${aug_data_path}/${res_name}_simdata.lsf
bsub -R "rusage[mem=15GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${res_name} ${py_script} --perturb_vec ${mono_perturb_vec} --nonperturb_vec ${mono_nonperturb_vec} --res_name ${res_name} --in_name ${in_name}

#######
## format data for BayesPrism and CIBERSORT
#######
py_script="python ../evaluation_experiments/pbmc/pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path}  -exp ${res_name}  --no_use_test -pidx "
for pidx in `seq 0 9`;
do
    lsf_file=${cybersort_path}/${res_name}_${pidx}_translate_to_cibersort.lsf
    bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${res_name} ${py_script} ${pidx}
done
py_script="python ../evaluation_experiments/pbmc/pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path}  -exp ${res_name}  --use_test -pidx 6"
lsf_file=${cybersort_path}/${res_name}_6_test_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${res_name} ${py_script}

#######
## Train BuDDI
#######

py_script="python ../evaluation_experiments/pbmc/pbmc_diva_train_iter.py -res ${res_path} -aug ${aug_data_path} -n ${num_genes} "

exp_id=${res_name}
unlab_exp_id=${res_name}
lsf_file=${res_path}/${exp_id}_${unlab_exp_id}_diva_train.lsf
bsub -R "rusage[mem=15GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J buddi_${exp_id} ${py_script} -exp ${exp_id} -unlab_exp ${unlab_exp_id}

#######
## Test BuDDI
#######
py_script="python ../evaluation_experiments/pbmc/pbmc_diva_test.py -res ${res_path} -aug ${aug_data_path}"

train_id=${res_name}
test_id=${res_name}
unlab_exp_id=${res_name}
curr_py_script="${py_script} -train ${train_id} -test ${test_id} -unlab_exp ${unlab_exp_id}"
lsf_file=${res_path}/train-${train_id}-test-${test_id}-unlab-${unlab_exp_id}.lsf
bsub -R "rusage[mem=10GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${train_id} ${curr_py_script}


#######
## Run BayesPrism 
#######
r_script="Rscript ../bayesprism/bayes_prism_pbmc_test_multi.R "
num_samp=8100
ncores=20

file_id_train=${res_name}
file_id_test=${res_name}
run_id=${res_name}
lsf_file=${bp_path}/${file_id_train}_${file_id_test}_final.lsf
bsub -R "rusage[mem=48GB]" -W 48:00 -n 20 -q "normal" -o ${lsf_file} -J bp_${run_id} ${r_script} ${cybersort_path} ${bp_path} ${file_id_train} ${file_id_test} ${num_samp} ${ncores}



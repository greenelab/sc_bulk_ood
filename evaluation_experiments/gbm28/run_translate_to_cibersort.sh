

# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

cybersort_path="${work_dir}/../../data/single_cell_data/cybersort_gbm28/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_gbm28_data/"
py_script="python gbm28_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -exp "

# skip "BT1187"
# skip MGH85

ELEMS=(
    "MGH66 436"
    "MGH100 209"
    "MGH101 200"
    "MGH102 312"
    "MGH104 355"
    "MGH105 634"
    "MGH106 200"
    "MGH110 359"
    "MGH113 260"
    "MGH115 165"
    "MGH121 297"
    "MGH122 273"
    "MGH124 370"
    "MGH125 399"
    "MGH128 184"
    "MGH129 192"
    "MGH136 231"
    "MGH143 274"
    "MGH151 163"
    "MGH152 229"
    "BT749 317"
    "BT771 339"
    "BT786 212"
    "BT830 220"
    "BT920 301"
    "BT1160 335"
)

for line in "${ELEMS[@]}"; do
    # now run this for all out experimental desires
    parts=($line)
    exp_id="${parts[0]}"
    lsf_file=${cybersort_path}/${exp_id}_translate_to_cibersort.lsf
    bsub -R "rusage[mem=5GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}
done

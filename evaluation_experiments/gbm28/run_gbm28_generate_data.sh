

# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

count_file="${work_dir}/../../data/single_cell_data/gbm28/expression/IDHwtGBM.processed.SS2.logTPM.txt.gz"
meta_file="${work_dir}/../../data/single_cell_data/gbm28/metadata/IDHwt.GBM.Metadata.SS2.tsv"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_gbm28_data/"
py_script="python gbm28_generate_data.py -c ${count_file} -m ${meta_file} -aug ${aug_data_path}"

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
    "MGH85 245"
    "BT749 317"
    "BT771 339"
    "BT786 212"
    "BT830 220"
    "BT920 301"
    "BT1160 335"
    "BT1187 219"
)

for line in "${ELEMS[@]}"; do
    # now run this for all out experimental desires
    parts=($line)
    samp="${parts[0]}"
    num_cells="${parts[1]}"
    lsf_file=${aug_data_path}/${samp}_generate_data.lsf
    py_script_curr="${py_script} -n ${num_cells} -samp ${samp}"
    bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${samp} ${py_script_curr}
done

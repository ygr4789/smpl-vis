#!/bin/bash

BASE_DIR="/data2/mochi_demo/mochi_obj_retarget"
FOLDER_LIST=("47_0" "48_6")
# FOLDER_LIST=("selected_vis_pred")
PICKLE_SUFFIX="_vis.pkl"
# C_VALUES=(0 1 3 5)

for folder in "${FOLDER_LIST[@]}"; do
    INPUT_DIR="${BASE_DIR}/${folder}/"

    for pkl_path in "$INPUT_DIR"/*"$PICKLE_SUFFIX"; do
        [ -f "$pkl_path" ] || continue
        echo "Running main.py -i $pkl_path -c 0 -p"
        python main.py -i "$pkl_path" -c 0 -p
        # for c_val in "${C_VALUES[@]}"; do
            # python main.py -i "$pkl_path" -c "$c_val" -p
            # python main.py -i "$pkl_path" -c "$c_val" 
        # done
    done
done

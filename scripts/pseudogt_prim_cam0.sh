#!/bin/bash

BASE_DIR="/data2/mochi_demo"
FOLDER_LIST=("selected_vis_pseudo_gt")
PICKLE_SUFFIX=".pkl"

for folder in "${FOLDER_LIST[@]}"; do
    INPUT_DIR="${BASE_DIR}/${folder}"

    for pkl_path in "$INPUT_DIR"/*"$PICKLE_SUFFIX"; do
        [ -f "$pkl_path" ] || continue
        python main.py -i "$pkl_path" -c 0 -p -g
        # python main.py -i "$pkl_path" 
    done
done

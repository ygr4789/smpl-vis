#!/bin/bash

ROOT_DIR="./video/prim_pred"

# Iterate over all subdirectories
for SUBDIR in "$ROOT_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        # Remove the "./video/" prefix
        REL_PATH="${SUBDIR#./video/}"
        echo "Processing: $REL_PATH"
        python scripts/omomo_concat.py --folder "$REL_PATH"
    fi
done

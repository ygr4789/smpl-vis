#!/bin/bash

ROOT_DIR="./video/prim_pseudogt"

# Iterate over all subdirectories
for SUBDIR in "$ROOT_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        # Remove the "./video/" prefix
        REL_PATH="${SUBDIR#./video/}"
        echo "Processing: $REL_PATH"
        python scripts/pseudogt_concat.py --folder "$REL_PATH"
    fi
done

# Script paths
MAIN_SCRIPT := main.py

# Directory paths
DATA_DIR := data
OUTPUT_DIR := output
CACHE_DIR := cache
VIDEO_DIR := video

# Your command here
make:
	python ${MAIN_SCRIPT} -i ${DATA_DIR}/sample.pkl
	
clean-output:
	rm -rf $(OUTPUT_DIR)/*

clean-cache:
	rm -rf $(CACHE_DIR)/*

clean-video:	
	rm -rf $(VIDEO_DIR)/*

mrproper: clean-output clean-cache clean-video
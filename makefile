# Script paths
MAIN_SCRIPT := main.py
RENDER_SMPL_SCRIPT := blender/render_smpl.py
RENDER_PRIM_SCRIPT := blender/render_prim.py

# Directory paths
DATA_DIR := data
OUTPUT_DIR := output
CACHE_DIR := cache
VIDEO_DIR := video

make:
	# python main.py -i $(DATA_DIR)/2_2 -a -c 1 -sc 1
	python main.py -i $(DATA_DIR)/pred_sample_10_seq_5_eval_vis_0_90.pkl -c -1 -sc 2 -q
	# python main.py -i $(DATA_DIR)/pseudo_gt_sample_54_seq_4_eval_vis_0_58.pkl -c 1 -g -sc 3
	# python main.py -i $(DATA_DIR)/2_2 -a -c 1 -p -sc 0
	# python main.py -i $(DATA_DIR)/pred_sample_10_seq_5_eval_vis_0_90.pkl -c 1 -p -sc 1
	# python main.py -i $(DATA_DIR)/pseudo_gt_sample_54_seq_4_eval_vis_0_58.pkl -c 1 -g -p -sc 4
	
clean-output:
	rm -rf $(OUTPUT_DIR)/*

clean-cache:
	rm -rf $(CACHE_DIR)/*

clean-video:	
	rm -rf $(VIDEO_DIR)/*

mrproper: clean-output clean-cache clean-video
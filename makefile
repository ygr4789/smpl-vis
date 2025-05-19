# Script paths
MAIN_SCRIPT := main.py
RENDER_SMPL_SCRIPT := blender/render_smpl.py
RENDER_PRIM_SCRIPT := blender/render_prim.py

# Directory paths
DATA_DIR := data
OUTPUT_DIR := output
CACHE_DIR := cache
RESULT_DIR := video

make:
	# python main.py -i $(DATA_DIR)/2_2 -a -c 1
	python main.py -i $(DATA_DIR)/pred_sample_10_seq_5_eval_vis_0_90.pkl -c 1
	python main.py -i $(DATA_DIR)/pseudo_gt_sample_54_seq_4_eval_vis_0_58.pkl -c 1 -g
	python main.py -i $(DATA_DIR)/2_2 -a -c 1 -p
	python main.py -i $(DATA_DIR)/pred_sample_10_seq_5_eval_vis_0_90.pkl -c 1 -p
	python main.py -i $(DATA_DIR)/pseudo_gt_sample_54_seq_4_eval_vis_0_58.pkl -c 1 -g -p
	
clean-output:
	rm -rf $(OUTPUT_DIR)/*

clean-cache:
	rm -rf $(CACHE_DIR)/*

clean-result:	
	rm -rf $(RESULT_DIR)/*

mrproper: clean-output clean-cache clean-result
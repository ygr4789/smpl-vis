SMPL_SCRIPT := main.py
RENDER_SCRIPT := blender/seq.py
DATA_DIR := data
OUTPUT_DIR := output
CACHE_DIR := cache
RESULT_DIR := video

# Extract base names from .pkl files
BASE_NAMES := $(basename $(notdir $(wildcard $(DATA_DIR)/*.pkl)))

# Targets like output/sample1/.rendered
RENDERED := $(addprefix $(OUTPUT_DIR)/, $(addsuffix /.rendered, $(BASE_NAMES)))

# Default target
all: $(RENDERED)

# Rule: run main.py if .pkl is newer than output dir
$(OUTPUT_DIR)/%: $(DATA_DIR)/%.pkl
	@echo "Generating $@ from $<"
	python $(SMPL_SCRIPT) $<

# Rule: run rendering only if previous step succeeded 
$(OUTPUT_DIR)/%/.rendered: $(OUTPUT_DIR)/%
	@echo "Rendering $*"
	export PYTHONPATH=$PWD
	blender --background --python $(RENDER_SCRIPT) -- -t 0 $(OUTPUT_DIR)/$*
	blender --background --python $(RENDER_SCRIPT) -- -t 1 $(OUTPUT_DIR)/$*
	blender --background --python $(RENDER_SCRIPT) -- -t 2 $(OUTPUT_DIR)/$*
	touch $@
	@echo "--------------------------------"
	@echo "Video rendering completed for $* !"
	@echo "--------------------------------"
	@echo

.PRECIOUS: $(OUTPUT_DIR)/%

# Clean all outputs
clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(CACHE_DIR)/*
	
# Clean only .rendered files
clean-rendered:
	rm -f $(OUTPUT_DIR)/*/.rendered
	
mrproper: clean
	rm -rf $(RESULT_DIR)/*
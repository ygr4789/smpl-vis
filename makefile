# Script paths
SMPL_SCRIPT := main.py
RENDER_SCRIPT := blender/seq.py

# Directory paths
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
	@echo "Generating .obj files from $<"
	python $(SMPL_SCRIPT) $<

# Render meshes from different camera angles
define render_sequence
	PYTHONPATH=$$PWD blender --background --python $(RENDER_SCRIPT) -- -t $(1) -c 0 $(OUTPUT_DIR)/$(2)
endef

# Render meshes and mark as complete
$(OUTPUT_DIR)/%/.rendered: $(OUTPUT_DIR)/%
	@echo "Rendering $*"
	# $(call render_sequence,0,$*)
	$(call render_sequence,1,$*)
	$(call render_sequence,2,$*)
	@touch $@
	@echo "--------------------------------"
	@echo "Video rendering completed for $* !"
	@echo "--------------------------------"
	@echo

.PRECIOUS: $(OUTPUT_DIR)/%

clean: clean-cache clean-output clean-rendered

clean-cache:
	rm -rf $(CACHE_DIR)/*

clean-objs:
	rm -rf $(OUTPUT_DIR)/*

clean-rendered:
	rm -f $(OUTPUT_DIR)/*/.rendered
	
mrproper: clean
	rm -rf $(RESULT_DIR)/*
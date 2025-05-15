# Script paths
SMPL_SCRIPT := main.py
PRIM_SCRIPT := pkl2npz.py
RENDER_SMPL_SCRIPT := blender/render_smpl.py
RENDER_PRIM_SCRIPT := blender/render_prim.py

# Directory paths
DATA_DIR := data
OUTPUT_DIR := output
CACHE_DIR := cache
RESULT_DIR := video

NPZ_SUFFIX := .npz
PKL_SUFFIX := .pkl
RENDERED_SUFFIX := .rendered

# Extract base names from .pkl files
BASE_NAMES := $(basename $(notdir $(wildcard $(DATA_DIR)/*$(PKL_SUFFIX))))

# Targets like output/sample1/.rendered
RENDERED := $(addprefix $(OUTPUT_DIR)/, $(addsuffix $(RENDERED_SUFFIX), $(BASE_NAMES)))

# Default target
all: $(RENDERED)

# Rule: run main.py if .pkl is newer than output dir
$(OUTPUT_DIR)/%: $(DATA_DIR)/%.pkl
	@echo "Generating .obj files from $<"
	python $(SMPL_SCRIPT) $<
	python $(PRIM_SCRIPT) $<

# Render meshes from different camera angles
define render_sequence_smpl
	PYTHONPATH=$$PWD blender --background --python $(RENDER_SMPL_SCRIPT) -- -t $(1) $(OUTPUT_DIR)/$(2)
endef

define render_sequence_prim
	PYTHONPATH=$$PWD blender --background --python $(RENDER_PRIM_SCRIPT) -- -t $(1) $(CACHE_DIR)/$(2)$(NPZ_SUFFIX)
endef

# Render meshes and mark as complete
$(OUTPUT_DIR)/%$(RENDERED_SUFFIX): $(OUTPUT_DIR)/%
	@echo "Rendering $*"
	$(call render_sequence_prim,0,$*)
	$(call render_sequence_prim,1,$*)
	$(call render_sequence_prim,2,$*)
	@touch $@
	@echo "--------------------------------"
	@echo "Video rendering completed for $* !"
	@echo "--------------------------------"
	@echo

.PRECIOUS: $(OUTPUT_DIR)/%

clean: clean-cache clean-objs clean-rendered

clean-cache:
	rm -rf $(CACHE_DIR)/*

clean-objs:
	rm -rf $(OUTPUT_DIR)/*

clean-rendered:
	rm -f $(OUTPUT_DIR)/*$(RENDERED_SUFFIX)
	
mrproper: clean
	rm -rf $(RESULT_DIR)/*
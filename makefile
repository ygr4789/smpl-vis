# Define paths
DATA_DIR := data
OUTPUT_DIR := output
SCRIPT := main.py
RENDER_SCRIPT := blender/seq.py

# Get list of base names (sample1, sample2)
BASE_NAMES := $(basename $(notdir $(wildcard $(DATA_DIR)/*.pkl)))

# Final target list
RENDERED_TARGETS := $(addprefix $(OUTPUT_DIR)/, $(addsuffix /.rendered, $(BASE_NAMES)))

# Default goal
all: $(RENDERED_TARGETS)

# Rule to run main.py and create output directory
$(OUTPUT_DIR)/%/.rendered:
	@name=$*; \
	echo "Running python $(SCRIPT) $(DATA_DIR)/$$name.pkl"; \
	python $(SCRIPT) $(DATA_DIR)/$$name.pkl; \
	echo "Rendering with python $(RENDER_SCRIPT) $(OUTPUT_DIR)/$$name"; \
	python $(RENDER_SCRIPT) $(OUTPUT_DIR)/$$name -l; \
	touch $@

# Clean all generated output
clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(CACHE_DIR)/*
	
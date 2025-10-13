# Root-level Makefile to forward to gta1_serve targets

SERVE_DIR ?= gta1_serve

# Only pass through variables that are set (avoid clobbering defaults)
PASS_VARS := TAG PLAT PUSH_EXPORTER COMPRESSION FORCE_COMPRESSION PUSH_CACHE
PASS_ARGS := $(foreach v,$(PASS_VARS),$(if $($(v)),$(v)=$($(v))))

.PHONY: build build-local push push-build login builder-init push-latest print-tag clean help

build build-local push push-build login builder-init push-latest print-tag clean:
	@$(MAKE) -C $(SERVE_DIR) $@ $(PASS_ARGS)

help:
	@$(MAKE) -C $(SERVE_DIR) help


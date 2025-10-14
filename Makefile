# Root-level Makefile to forward to gta1_serve targets

SERVE_DIR ?= gta1_serve

# Only pass through variables that are set (avoid clobbering defaults)
PASS_VARS := TAG PLAT PUSH_EXPORTER COMPRESSION FORCE_COMPRESSION PUSH_CACHE
PASS_ARGS := $(foreach v,$(PASS_VARS),$(if $($(v)),$(v)=$($(v))))

.PHONY: build build-local push push-build login builder-init push-latest print-tag clean help \
	ray-build ray-push-build ray-push ray-login ray-builder-init ray-print-tag ray-clean ray-help

build build-local push push-build login builder-init push-latest print-tag clean:
	@$(MAKE) -C $(SERVE_DIR) $@ $(PASS_ARGS)

help:
	@$(MAKE) -C $(SERVE_DIR) help

# Ray wrappers (use alternate Makefile.ray)
ray-build ray-push ray-push-build ray-login ray-builder-init ray-print-tag ray-clean ray-help:
	@$(MAKE) -C $(SERVE_DIR) -f Makefile.ray $(subst ray-,,$@) $(PASS_ARGS)

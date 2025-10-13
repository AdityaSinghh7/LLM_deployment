.PHONY: builder-init login build build-local push push-build push-latest print-tag help

# ---- Config ----
REGISTRY_USER := adityads7
IMAGE_NAME    := gta1-serve
IMAGE         := $(REGISTRY_USER)/$(IMAGE_NAME)
PLAT          := linux/amd64

# Generate consistent tag for build+push workflow
# Priority: 1) Command line TAG, 2) .build_tag file, 3) New timestamp
BUILD_TAG_FILE := .build_tag
ifndef TAG
  ifneq (,$(wildcard $(BUILD_TAG_FILE)))
    TAG := $(shell cat $(BUILD_TAG_FILE))
  else
    DATE_TAG := $(shell date -u +"%Y%m%d-%H%M%S")
    TAG := $(DATE_TAG)
  endif
endif
DOCKERFILE    := docker/Dockerfile.runtime

# Optional: registry cache, opt-in via PUSH_CACHE=1
ifeq ($(PUSH_CACHE),1)
CACHE_REF     := $(IMAGE):cache
CACHE_FLAGS   := --build-arg BUILDKIT_INLINE_CACHE=1 \
                 --cache-from=type=registry,ref=$(CACHE_REF) \
                 --cache-to=type=registry,ref=$(CACHE_REF),mode=max
else
CACHE_FLAGS   :=
endif

# Common build flags
BUILD_FLAGS   := --platform $(PLAT) \
                 -f $(DOCKERFILE) \
                 --provenance=false \
                 --sbom=false \
                 $(CACHE_FLAGS)

help:
	@echo "Targets:"
	@echo "  builder-init  - init buildx"
	@echo "  login         - docker login"
	@echo "  build         - build (no push), tags :latest and :\$$TAG (uses --load, saves tag to .build_tag)"
	@echo "  push          - push :latest and :\$$TAG (with retries, reads tag from .build_tag)"
	@echo "  push-build    - buildx build and push (no --load), tags :latest and :\$$TAG"
	@echo "  build-local   - alias of build"
	@echo "  push-latest   - push only :latest"
	@echo "  print-tag     - print docker.io/$(IMAGE):$(TAG) (reads from .build_tag if exists)"
	@echo "  clean         - remove .build_tag file"

builder-init:
	@echo "🔧 Initializing Docker buildx builder..."
	- docker buildx create --use --name rp-gpu-builder 2>/dev/null || docker buildx use rp-gpu-builder
	@echo "✅ Builder ready"

login:
	@echo "🔐 Logging into Docker Hub..."
	docker login
	@echo "✅ Logged in"

# Build into local docker (single-conn load), then we can push separately.
build build-local: builder-init
	@echo "🏗️  Building image for $(PLAT)"
	@echo "   Image: $(IMAGE):$(TAG)"
	@echo "   Also tagging: $(IMAGE):latest"
	docker buildx build \
		$(BUILD_FLAGS) \
		-t $(IMAGE):$(TAG) \
		-t $(IMAGE):latest \
		--load \
		.
	@echo "$(TAG)" > $(BUILD_TAG_FILE)
	@echo "✅ Built locally (not pushed). Tag saved to $(BUILD_TAG_FILE)"

# Build and push in one step using buildx --push (no local load)
push-build: builder-init
	@echo "🚀 Build & push for $(PLAT)"
	@echo "   Image: $(IMAGE):$(TAG)"
	@echo "   Also tagging: $(IMAGE):latest"
	docker buildx build \
		$(BUILD_FLAGS) \
		--output=type=registry,compression=zstd,force-compression=true,name=$(IMAGE):$(TAG),name=$(IMAGE):latest \
		.

# Robust push with small retry loop
push: login
	@echo "🚀 Pushing docker.io/$(IMAGE):$(TAG) and :latest"
	@attempt=0; max=5; delay=5; \
	until [ $$attempt -ge $$max ]; do \
	  attempt=$$((attempt+1)); \
	  echo "→ [$$attempt/$$max] docker push $(IMAGE):$(TAG)"; \
	  if docker push $(IMAGE):$(TAG); then break; fi; \
	  echo "   retrying in $$delay s…"; sleep $$delay; delay=$$((delay*2)); \
	done; \
	if [ $$attempt -ge $$max ]; then echo "❌ Push failed for :$(TAG)"; exit 1; fi; \
	attempt=0; delay=5; \
	until [ $$attempt -ge $$max ]; do \
	  attempt=$$((attempt+1)); \
	  echo "→ [$$attempt/$$max] docker push $(IMAGE):latest"; \
	  if docker push $(IMAGE):latest; then break; fi; \
	  echo "   retrying in $$delay s…"; sleep $$delay; delay=$$((delay*2)); \
	done; \
	if [ $$attempt -ge $$max ]; then echo "❌ Push failed for :latest"; exit 1; fi
	@echo "✅ Push complete."

push-latest: login
	docker push $(IMAGE):latest

print-tag:
	@echo "docker.io/$(IMAGE):$(TAG)"

clean:
	@echo "🧹 Cleaning up build artifacts..."
	-rm -f $(BUILD_TAG_FILE)
	@echo "✅ Cleaned."

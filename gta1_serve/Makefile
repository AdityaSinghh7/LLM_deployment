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

# Push behavior toggles
# - COMPRESSION: zstd or gzip (zstd reduces size, needs registry support)
# - FORCE_COMPRESSION: true/false
# - PUSH_EXPORTER: image|registry
#     image    -> use image exporter (+ --push semantics) and tags (-t)
#     registry -> use registry exporter with explicit name= entries
COMPRESSION      ?= gzip
FORCE_COMPRESSION?= false
PUSH_EXPORTER    ?= image

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

# Compose exporter flags based on PUSH_EXPORTER
ifeq ($(PUSH_EXPORTER),registry)
PUSH_OUTPUT     := --output=type=registry,compression=$(COMPRESSION),force-compression=$(FORCE_COMPRESSION),name=$(IMAGE):$(TAG),name=$(IMAGE):latest
PUSH_TAG_FLAGS  :=
else ifeq ($(PUSH_EXPORTER),image)
PUSH_OUTPUT     := --output=type=image,push=true,compression=$(COMPRESSION),force-compression=$(FORCE_COMPRESSION)
PUSH_TAG_FLAGS  := -t $(IMAGE):$(TAG) -t $(IMAGE):latest
else
$(error Unsupported PUSH_EXPORTER '$(PUSH_EXPORTER)'. Use 'image' or 'registry')
endif

help:
	@echo "Targets:"
	@echo "  builder-init  - init buildx"
	@echo "  login         - docker login"
	@echo "  build         - build (no push), tags :latest and :\$$TAG (uses --load, saves tag to .build_tag)"
	@echo "  push          - push :latest and :\$$TAG (with retries, reads tag from .build_tag)"
	@echo "  push-build    - buildx push during build (defaults: gzip, force=false). Vars: PUSH_EXPORTER=image|registry, COMPRESSION=zstd|gzip, FORCE_COMPRESSION=true|false, CLEAR_PROXY=1"
	@echo "  build-local   - alias of build"
	@echo "  push-latest   - push only :latest"
	@echo "  print-tag     - print docker.io/$(IMAGE):$(TAG) (reads from .build_tag if exists)"
	@echo "  clean         - remove .build_tag file"

# Optional builder opts (e.g., clear proxies inside buildkit)
ifeq ($(CLEAR_PROXY),1)
BUILDER_OPTS := --driver-opt env.http_proxy= --driver-opt env.https_proxy= --driver-opt env.no_proxy=*
else
BUILDER_OPTS :=
endif

builder-init:
	@echo "🔧 Initializing Docker buildx builder..."
	- docker buildx inspect rp-gpu-builder >/dev/null 2>&1 || docker buildx create --use --name rp-gpu-builder $(BUILDER_OPTS)
	- docker buildx use rp-gpu-builder
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
push-build: builder-init login
	@echo "🚀 Build & push for $(PLAT)"
	@echo "   Image: $(IMAGE):$(TAG)"
	@echo "   Also tagging: $(IMAGE):latest"
	docker buildx build \
		$(BUILD_FLAGS) \
		$(PUSH_TAG_FLAGS) \
		$(PUSH_OUTPUT) \
		.

# Robust push with retry/backoff
PUSH_RETRIES       ?= 7
PUSH_BACKOFF_INIT  ?= 5

push: login
	@echo "🚀 Pushing docker.io/$(IMAGE):$(TAG) and :latest"
	@attempt=0; max=$(PUSH_RETRIES); delay=$(PUSH_BACKOFF_INIT); \
	until [ $$attempt -ge $$max ]; do \
	  attempt=$$((attempt+1)); \
	  echo "→ [$$attempt/$$max] docker push $(IMAGE):$(TAG)"; \
	  if DOCKER_CLIENT_TIMEOUT=900 docker push $(IMAGE):$(TAG); then break; fi; \
	  echo "   retrying in $$delay s…"; sleep $$delay; delay=$$((delay*2)); \
	done; \
	if [ $$attempt -ge $$max ]; then echo "❌ Push failed for :$(TAG)"; exit 1; fi; \
	attempt=0; delay=$(PUSH_BACKOFF_INIT); \
	until [ $$attempt -ge $$max ]; do \
	  attempt=$$((attempt+1)); \
	  echo "→ [$$attempt/$$max] docker push $(IMAGE):latest"; \
	  if DOCKER_CLIENT_TIMEOUT=900 docker push $(IMAGE):latest; then break; fi; \
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

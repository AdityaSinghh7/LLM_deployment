<!-- 828e0bf0-0c55-404e-8580-bca20d81e2c6 c5773996-9ab4-4177-98aa-116072965324 -->
# vLLM-based Serverless Deployment for GTA1-32B on RunPod

## Overview

Replace the existing Ray Serve implementation with a single-process FastAPI + vLLM service optimized for RunPod serverless deployment on A100 GPUs. The solution uses persistent volume caching, smart image resizing, and Docker cross-compilation from Mac to linux/amd64.

## 1. Repository Structure

Final layout in `gta1_serve/`:

```
gta1_serve/
├── pyproject.toml          # Runtime dependencies (filled)
├── src/
│   ├── __init__.py         # Existing (keep)
│   ├── model_serving.py    # Existing Ray Serve (keep for reference, not used)
│   ├── server_mp.py        # NEW: FastAPI + vLLM service
│   └── utils_mm.py         # NEW: Image + prompt helpers
├── docker/
│   ├── Dockerfile.runtime  # NEW: Based on vllm/vllm-openai:v0.11.0-x86_64
│   └── entrypoint.sh       # NEW: Starts health + main API
├── Makefile                # NEW: Build/push with timestamp tags
└── README.md               # Optional: RunPod deployment notes
```

## 2. Service Implementation (`src/server_mp.py`)

Create a single-process FastAPI application with global vLLM instance:

**Endpoints:**

- `GET /ping` → Returns `200 "ok"` (RunPod health endpoint)
- `GET /health` → Returns model info JSON (id, dtype, device)
- `POST /call_llm` → Main inference endpoint

**Request/Response:**

- Input: JSON with `messages` array (system + user with exactly one image via data URI) and optional params (`max_new_tokens`, `temperature`, `top_p`)
- Processing: 
  - Extract exactly one image from messages
  - Convert data URI to PIL Image
  - Apply `smart_resize` using `AutoImageProcessor`
  - Rewrite prompt tokens: `<|media_begin|>...<|media_end|>` → `<|vision_start|><|image_pad|><|vision_end|>`
  - Call `LLM.generate([...], sampling_params=..., multi_modal_data={"image": [resized_image]})`
- Output: `{"response": str, "usage": {...}}`

**Initialization (module-global singleton):**

```python
LLM(
    model="Salesforce/GTA1-32B",
    tokenizer="Salesforce/GTA1-32B",
    tokenizer_mode="slow",
    trust_remote_code=True,
    dtype="bfloat16",
    limit_mm_per_prompt={"image": 1},
    tensor_parallel_size=1
)
```

Plus:

- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` for chat template
- `AutoImageProcessor.from_pretrained(..., trust_remote_code=True)` for image resizing

Environment variables:

- `MODEL_ID` (default: `Salesforce/GTA1-32B`)
- `MODEL_REVISION` (optional HF revision pin)
- `PORT` (default: `8000`)
- `PORT_HEALTH` (default: `8001`)

## 3. Utilities Module (`src/utils_mm.py`)

Extract and enhance existing helpers:

**Functions to implement:**

- `data_uri_to_pil(data_uri: str) -> PIL.Image` — Parse data URI to PIL Image (reuse from existing)
- `extract_images(messages: List[Dict]) -> List[PIL.Image]` — Extract images from user messages (reuse from existing)
- `build_prompt_with_template(tokenizer, messages) -> str` — Apply chat template + media token rewrite (reuse from existing)
- `smart_resize(image: PIL.Image, processor) -> PIL.Image` — **NEW**: Resize using `processor.patch_size`, `merge_size`, `min_pixels`, `max_pixels` per GTA1 model card

## 4. Dependencies (`pyproject.toml`)

Pin runtime-only dependencies for Python 3.10-3.12:

```toml
[project]
name = "gta1-serve"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "vllm==0.11.0",
    "fastapi",
    "uvicorn[standard]",
    "pillow",
    "blobfile",
    "transformers",
    "huggingface-hub",
    "aiohttp",
    "pydantic<3",
]
```

**Note:** Rely on vLLM base Docker image for CUDA/PyTorch stack; do not override Torch versions.

## 5. Caching & Persistence

All caches redirect to `/runpod-volume` for persistent storage across cold starts:

```bash
HF_HOME=/runpod-volume/hf
HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub
TRANSFORMERS_CACHE=/runpod-volume/hf/hub
XDG_CACHE_HOME=/runpod-volume/.cache
TRITON_CACHE_DIR=/runpod-volume/.cache/triton
VLLM_CACHE_ROOT=/runpod-volume/.cache/vllm  # optional
```

Model weights download to persistent volume on first run; subsequent starts reuse cached files.

## 6. Docker Strategy (`docker/Dockerfile.runtime`)

**Base image:** `vllm/vllm-openai:v0.11.0-x86_64`

**Build steps:**

1. Set working directory (e.g., `/app`)
2. Copy project files (`pyproject.toml`, `src/`, `docker/entrypoint.sh`)
3. Install dependencies from `pyproject.toml` (do not override base Torch/CUDA)
4. Set environment variables:

   - Cache paths from §5
   - `MODEL_ID=Salesforce/GTA1-32B`
   - `PORT=8000`, `PORT_HEALTH=8001`
   - `VLLM_USE_V1=1`
   - `TOKENIZERS_PARALLELISM=false`
   - `HF_HUB_ENABLE_HF_TRANSFER=1`

5. Make `entrypoint.sh` executable
6. Set `entrypoint.sh` as container entrypoint

**Note:** Model weights NOT baked into image; download happens on worker to persistent volume.

## 7. Entrypoint (`docker/entrypoint.sh`)

Launch script that:

1. Starts health server on `$PORT_HEALTH` (tiny FastAPI/HTTP server serving `GET /ping → 200`)
2. Starts main FastAPI app (`src/server_mp.py`) on `$PORT` with `uvicorn --workers 1 --host 0.0.0.0`
3. Optional warmup (if `WARMUP=1`): Run short Python snippet to pre-load LLM and force:

   - HF weight download
   - Triton kernel compilation
   - Single 1-token generation to materialize CUDA graphs

## 8. Makefile

Build automation with UTC timestamp tagging:

**Variables:**

```makefile
REGISTRY_USER := adityads7
IMAGE_NAME := gta1-serve
IMAGE := $(REGISTRY_USER)/$(IMAGE_NAME)
DATE_TAG := $(shell date -u +"%Y%m%d-%H%M%S")
```

**Targets:**

- `builder-init`: Create/select buildx builder for cross-compilation
- `login`: Docker Hub authentication
- `build`: Cross-compile for linux/amd64 and push with timestamp + latest tags:
  ```bash
  docker buildx build --platform linux/amd64 \
    -f docker/Dockerfile.runtime \
    -t $(IMAGE):$(DATE_TAG) -t $(IMAGE):latest \
    --provenance=false --push .
  ```

- `push`: No-op if using `--push` in build
- `print-tag`: Echo the computed image tag for RunPod config

## 9. RunPod Configuration

**Endpoint settings:**

- **Container Image:** `docker.io/<REGISTRY_USER>/gta1-serve:<DATE_TAG>`
- **GPU:** A100 80 GB
- **Ports:** Map `8000` (main API) and `8001` (health)
- **Network Volume:** Mount to `/runpod-volume`

**Environment variables:**

```bash
MODEL_ID=Salesforce/GTA1-32B
PORT=8000
PORT_HEALTH=8001
HF_HOME=/runpod-volume/hf
HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub
TRANSFORMERS_CACHE=/runpod-volume/hf/hub
XDG_CACHE_HOME=/runpod-volume/.cache
TRITON_CACHE_DIR=/runpod-volume/.cache/triton
VLLM_CACHE_ROOT=/runpod-volume/.cache/vllm
TOKENIZERS_PARALLELISM=false
VLLM_USE_V1=1
WARMUP=1  # optional
MODEL_REVISION=<commit-sha>  # optional
```

## 10. Implementation Notes

**Key differences from existing `model_serving.py`:**

- **Remove:** Ray Serve, batching decorators, multi-replica logic
- **Keep:** Prompt rewrite logic, exactly-one-image validation, data URI parsing
- **Add:** Smart image resizing with `AutoImageProcessor`, FastAPI-only endpoints
- **Change:** Use `LLM.generate([...], multi_modal_data={"image": [PIL]})` for vLLM multi-modal API

**Reusable code from existing implementation:**

- `data_uri_to_pil()` at lines 47-52
- `extract_images()` at lines 55-65
- `build_prompt_with_template()` at lines 72-82

## 11. Performance & Stability

- **Single worker:** Use `--workers 1` (one process = one GPU context)
- **Scaling:** Let RunPod scale replicas instead of forking processes
- **Engine:** vLLM V1 is default; `VLLM_USE_V1=1` set for clarity
- **Multi-modal:** Single image per prompt is the supported pattern for VLMs

## 12. Validation Checks

Before deployment, verify:

1. `GET /health` returns JSON with `"model": "Salesforce/GTA1-32B"` and `"dtype": "bfloat16"`
2. `GET /ping` on `PORT_HEALTH` returns `200` quickly (or `204` during model load)
3. First invocation warms caches: `/runpod-volume/hf` and `/runpod-volume/.cache/triton` populate as expected
4. Test inference with sample request containing one image

## Build Commands

**One-time setup:**

```bash
docker buildx create --use --name rp-gpu-builder
docker login
```

**Build and push:**

```bash
make build
# or manually:
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.runtime \
  -t docker.io/<you>/gta1-serve:$(date -u +%Y%m%d-%H%M%S) \
  -t docker.io/<you>/gta1-serve:latest \
  --provenance=false --push .
```

## References

This plan follows:

- GTA1-32B model card smart-resize pattern
- vLLM multi-modal input API (`multi_modal_data`)
- vLLM V1 engine requirements
- RunPod load balancer health check conventions
- Official vLLM Docker image for A100 compatibility

### To-dos

- [ ] Create src/utils_mm.py with data_uri_to_pil, extract_images, build_prompt_with_template, and smart_resize functions
- [ ] Create src/server_mp.py with FastAPI endpoints (/ping, /health, /call_llm) and global vLLM instance
- [ ] Fill pyproject.toml with vLLM 0.11.0 and runtime dependencies
- [ ] Create docker/Dockerfile.runtime based on vllm/vllm-openai:v0.11.0-x86_64 with cache env vars
- [ ] Create docker/entrypoint.sh to start health server and main API
- [ ] Create Makefile with builder-init, login, build, and print-tag targets for cross-compilation
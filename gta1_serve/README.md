# GTA1-32B Serverless Deployment for RunPod

vLLM-based serverless deployment for the Salesforce GTA1-32B vision-language model on RunPod A100 GPUs.

## Overview

This implementation provides a single-process FastAPI + vLLM service optimized for RunPod serverless deployment. It features:

- **Fast cold starts** via persistent volume caching
- **Smart image resizing** using AutoImageProcessor
- **Single-image inference** with vLLM multi-modal API
- **Cross-platform builds** from Mac to linux/amd64

## Architecture

```
┌─────────────────────────────────────────┐
│  RunPod Serverless Worker (A100 80GB)  │
├─────────────────────────────────────────┤
│  Health Server (port 8001)              │
│  ├─ GET /ping → 200 "ok"                │
│                                          │
│  Main API Server (port 8000)            │
│  ├─ GET /ping → 200 "ok"                │
│  ├─ GET /health → model info            │
│  └─ POST /call_llm → inference          │
│                                          │
│  vLLM Engine (bfloat16, single GPU)     │
│  └─ GTA1-32B (32B params)               │
└─────────────────────────────────────────┘
         ↓
/runpod-volume (persistent storage)
├─ hf/hub/ (model weights)
└─ .cache/triton/ (compiled kernels)
```

## Quick Start

### 1. Build and Push Docker Image

```bash
# One-time setup
make builder-init
make login

# Build and push
make build

# Get the image tag for RunPod
make print-tag
```

This cross-compiles from Mac to `linux/amd64` and pushes to Docker Hub with a timestamp tag.

### 2. Configure RunPod Endpoint

**Container Settings:**
- **Image:** `docker.io/adityads7/gta1-serve:<TIMESTAMP>`
- **GPU:** A100 80 GB
- **Container Disk:** 20 GB (minimum)
- **Network Volume:** Mount to `/runpod-volume`

**Port Configuration:**
- Main API: `8000` (TCP)
- Health Check: `8001` (TCP)

**Environment Variables:**

```bash
MODEL_ID=Salesforce/GTA1-32B
PORT=8000
PORT_HEALTH=8001

# Cache paths (persistent volume)
HF_HOME=/runpod-volume/hf
HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub
TRANSFORMERS_CACHE=/runpod-volume/hf/hub
XDG_CACHE_HOME=/runpod-volume/.cache
TRITON_CACHE_DIR=/runpod-volume/.cache/triton
VLLM_CACHE_ROOT=/runpod-volume/.cache/vllm

# vLLM settings
TOKENIZERS_PARALLELISM=false
VLLM_USE_V1=1
HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: Pre-load on startup
WARMUP=1

# Optional: Pin HF revision
# MODEL_REVISION=<commit-sha>
```

**Health Check:**
- Endpoint: `/ping`
- Port: `8001`
- Interval: 10s

### 3. Test the Endpoint

```python
import requests
import base64

# Prepare image as data URI
with open("test_image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
data_uri = f"data:image/jpeg;base64,{img_b64}"

# Build request
payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_uri},
                {"type": "text", "text": "What do you see in this image?"}
            ]
        }
    ],
    "max_new_tokens": 512,
    "temperature": 0.0,
    "top_p": 0.9
}

# Call endpoint
response = requests.post(
    "https://your-runpod-endpoint.runpod.net/call_llm",
    json=payload
)

print(response.json())
```

## API Reference

### `POST /call_llm`

Main inference endpoint.

**Request:**

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data:image/jpeg;base64,..."},
        {"type": "text", "text": "..."}
      ]
    }
  ],
  "max_new_tokens": 512,
  "temperature": 0.0,
  "top_p": 0.9
}
```

**Response:**

```json
{
  "response": "Generated text...",
  "usage": {
    "prompt_tokens": 1234,
    "generated_tokens": 56,
    "total_tokens": 1290
  },
  "error": ""
}
```

### `GET /health`

Returns model information.

**Response:**

```json
{
  "status": "ok",
  "model": "Salesforce/GTA1-32B",
  "dtype": "bfloat16",
  "device": "cuda"
}
```

### `GET /ping`

Health check (both ports 8000 and 8001).

**Response:** `200 OK` with body `"ok"`

## Development

### Local Testing (without GPU)

```bash
# Install dependencies
cd gta1_serve
pip install -e .

# Run server (will fail without GPU, but tests import logic)
cd src
python server_mp.py
```

### Project Structure

```
gta1_serve/
├── src/
│   ├── server_mp.py       # FastAPI server with vLLM
│   ├── utils_mm.py        # Image processing utilities
│   └── model_serving.py   # Legacy Ray Serve (not used)
├── docker/
│   ├── Dockerfile.runtime # Production Dockerfile
│   └── entrypoint.sh      # Container startup script
├── pyproject.toml         # Python dependencies
├── Makefile               # Build automation
└── README.md              # This file
```

## Performance Notes

### First Request (Cold Start)

The first request triggers:
1. **Model download** (~60 GB) to `/runpod-volume/hf`
2. **Triton kernel compilation** (cached to `/runpod-volume/.cache/triton`)
3. **CUDA graph materialization** (first inference)

**Time:** ~15-20 minutes on A100 with fast network

### Subsequent Requests (Warm)

- **Startup:** ~2-3 minutes (loads from volume)
- **Inference:** ~1-3 seconds per request (depends on image size and max_tokens)

### Scaling

- Use `--workers 1` (one process per GPU)
- Let RunPod handle replica scaling
- Each replica loads its own model copy

## Troubleshooting

### Model not downloading

Check:
- HuggingFace Hub is accessible
- Volume is mounted to `/runpod-volume`
- `HF_HOME` and related env vars are set

```bash
# Debug: SSH into worker and check
ls -lah /runpod-volume/hf/hub
```

### Out of memory

A100 80GB should be sufficient. If issues occur:
- Reduce `max_new_tokens`
- Check that `tensor_parallel_size=1`
- Verify only one worker is running

### Health check fails

- Ensure port `8001` is mapped in RunPod config
- Check `/ping` endpoint specifically, not `/`
- Health server starts before main API

### Slow inference

First request is always slow (cold start). For subsequent requests:
- Check image size (resize reduces tokens)
- Verify `VLLM_USE_V1=1` is set
- Monitor GPU utilization

## References

- [GTA1-32B Model Card](https://huggingface.co/Salesforce/GTA1-32B)
- [vLLM Documentation](https://docs.vllm.ai/)
- [RunPod Serverless](https://docs.runpod.io/serverless)

## License

See project LICENSE file.


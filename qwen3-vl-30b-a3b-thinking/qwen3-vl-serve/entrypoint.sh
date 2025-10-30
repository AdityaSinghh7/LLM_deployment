#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Helper: return 0 if the api_server --help shows the flag pattern
has_flag() {
  python -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -E -q -- "$1"
}

ARGS=(python -m vllm.entrypoints.openai.api_server
  --host 0.0.0.0
  --port "${PORT}"
  --model "${VLLM_MODEL}"
  --trust-remote-code
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
)

# Optional / version-dependent flags (only add if supported)
if has_flag '\-\-async-scheduling'; then
  ARGS+=(--async-scheduling)
fi

# Some vLLM builds expose a generic --limit-mm-per-prompt (no ".video")
if has_flag '\-\-limit-mm-per-prompt'; then
  # If your goal is to reject/limit video, pass a conservative value (0 disables)
  LIMIT_MM_VIDEO="${LIMIT_MM_VIDEO:-0}"
  ARGS+=(--limit-mm-per-prompt "video:${LIMIT_MM_VIDEO}")
fi

# Older proposals like --limit-mm-per-prompt.video or --mm-encoder-tp-mode may not exist;
# we intentionally skip them unless they show up in --help to avoid hard failures.

# Expert parallel (MoE) toggle
if [[ "${ENABLE_EXPERT_PARALLEL:-}" != "" ]] && has_flag '\-\-enable-expert-parallel'; then
  ARGS+=(--enable-expert-parallel)
fi

exec "${ARGS[@]}"

#!/bin/bash
set -e

echo "🚀 Starting GTA1-32B serverless deployment..."

# Print environment info
echo "📋 Configuration:"
echo "  MODEL_ID: ${MODEL_ID}"
echo "  PORT: ${PORT}"
echo "  PORT_HEALTH: ${PORT_HEALTH}"
echo "  HF_HOME: ${HF_HOME}"
echo "  VLLM_USE_V1: ${VLLM_USE_V1}"

# Optional: Warmup phase
if [ "${WARMUP}" = "1" ]; then
    echo "🔥 Running warmup..."
    python3 -c "
import os
os.chdir('/app/src')
from server_mp import initialize_model
initialize_model()
print('✅ Warmup complete - model loaded and cached')
" || echo "⚠️  Warmup failed, continuing anyway..."
fi

# Start health server in background
echo "💓 Starting health server on port ${PORT_HEALTH}..."
python3 -c "
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

health_app = FastAPI()

@health_app.get('/ping', response_class=PlainTextResponse)
async def ping():
    return 'ok'

if __name__ == '__main__':
    uvicorn.run(health_app, host='0.0.0.0', port=${PORT_HEALTH}, log_level='warning')
" &

HEALTH_PID=$!
echo "✅ Health server started (PID: ${HEALTH_PID})"

# Wait a moment for health server to start
sleep 2

# Start main FastAPI service
echo "🚀 Starting main API server on port ${PORT}..."
cd /app/src
exec python3 -m uvicorn server_mp:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --log-level info


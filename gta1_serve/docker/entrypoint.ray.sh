#!/bin/bash
set -euo pipefail

echo "🚀 Starting GTA1-32B Ray Serve deployment..."

echo "📋 Configuration (Ray):"
echo "  MODEL_ID: ${MODEL_ID:-Salesforce/GTA1-32B}"
echo "  PORT: ${PORT:-3005}"
echo "  PORT_HEALTH: ${PORT_HEALTH:-3006}"
echo "  NUM_REPLICAS: ${NUM_REPLICAS:-1}"
echo "  HF_HOME: ${HF_HOME:-/runpod-volume/hf}"

# Start health server in background on PORT_HEALTH
echo "💓 Starting health server on port ${PORT_HEALTH:-3006}..."
python3 - <<'PY'
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn, os

app = FastAPI()

@app.get('/ping', response_class=PlainTextResponse)
async def ping():
    return 'ok'

if __name__ == '__main__':
    port = int(os.environ.get('PORT_HEALTH', '3006'))
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')
PY
&

HEALTH_PID=$!
echo "✅ Health server started (PID: ${HEALTH_PID})"

sleep 1

echo "🚀 Launching provider Ray app on port ${PORT:-3005}..."
exec python3 -u /app/src/model_serving.py \
  --model_path "${MODEL_ID:-Salesforce/GTA1-32B}" \
  --host 0.0.0.0 \
  --port "${PORT:-3005}" \
  --num_replicas "${NUM_REPLICAS:-1}"


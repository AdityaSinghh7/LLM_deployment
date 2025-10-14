#!/usr/bin/env bash
set -euo pipefail

# GUI grounding test against the provider Ray Serve app on RunPod.
# Runs from repo root. Requires gta1_serve/.env with RUNPOD_URL and RUNPOD_API_KEY.

DEBUG="${DEBUG:-}"

# Load env
set -a; source gta1_serve/.env; set +a

# Validate required env
: "${RUNPOD_URL:?RUNPOD_URL is required in gta1_serve/.env}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required in gta1_serve/.env}"

# Basic sanity on URL
if [[ ! "${RUNPOD_URL}" =~ ^https?:// ]]; then
  echo "RUNPOD_URL must start with http:// or https://; got: '${RUNPOD_URL}'" >&2
  exit 1
fi

[[ -n "${DEBUG}" ]] && set -x

# Paths
IMG_PATH="gta1_serve/data/sample_02.png"
MIME="image/png"  # change to image/jpeg if needed
OUT_DIR="gta1_serve/output"
OUT_JSON="${OUT_DIR}/response.json"
OUT_ANN="${OUT_DIR}/annotated.png"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${IMG_PATH}" ]]; then
  echo "Image not found: ${IMG_PATH}" >&2
  exit 1
fi

# Build payload via Python (faster, safer than shell string substitution)
SYSTEM_PROMPT="You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
INSTRUCTION="click the icon in the bottom row, third from the left"
BODY_FILE="${OUT_DIR}/body.json"

if [[ -n "${DEBUG}" ]]; then set +x; fi
IMG_LEN=$(python3 - "$IMG_PATH" "$MIME" "$SYSTEM_PROMPT" "$INSTRUCTION" "$BODY_FILE" <<'PY'
import sys, base64, json
img_path, mime, system_prompt, instruction, out_path = sys.argv[1:]
with open(img_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
data_uri = f"data:{mime};base64,{b64}"
body = {
    "max_new_tokens": 32,
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text",  "text": instruction},
            {"type": "image", "image": data_uri}
        ]},
    ],
}
with open(out_path, 'w', encoding='utf-8') as fo:
    json.dump(body, fo)
print(len(b64))
PY
)
if [[ -n "${DEBUG}" ]]; then set -x; fi
echo "Built payload; image_b64_len=${IMG_LEN}. Starting health and POST…"

BASE="${RUNPOD_URL%/}"
# Decide mode based on URL: direct HTTP service vs RunPod runsync API
# Treat only the central API host (api.runpod.ai/...) as RUNSYNC. Subdomains like
# <id>.api.runpod.ai are DIRECT proxy endpoints.
if [[ "${BASE}" =~ ^https?://api\.runpod\.ai(/|$) ]] || [[ "${BASE}" =~ /v2/ ]] || [[ "${BASE}" =~ /(runsync|run)(/|$) ]]; then
  MODE="RUNSYNC"
else
  MODE="DIRECT"
fi
URL="${BASE}/call_llm"
if [[ "${MODE}" == "DIRECT" ]]; then
  HEALTH_URL="${BASE}/ready"
  echo "Health check (direct): ${HEALTH_URL}"
  curl -fsS --http1.1 --no-alpn -X GET "${HEALTH_URL}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "x-api-key: ${RUNPOD_API_KEY}" \
    --connect-timeout 5 --max-time 10 >/dev/null || {
    HEALTH_URL="${BASE}/health"
    echo "Retrying health (direct): ${HEALTH_URL}"
    curl -fsS --http1.1 --no-alpn -X GET "${HEALTH_URL}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "x-api-key: ${RUNPOD_API_KEY}" \
      --connect-timeout 5 --max-time 10 >/dev/null || {
      echo "Health check failed or timed out (direct). Verify RUNPOD_URL points to your HTTP Service (proxy) and API key is valid." >&2
      exit 1
    }
  }
else
  echo "Health check (runsync): ${BASE}"
  RS_HEALTH=$(curl -fsS -X POST "${BASE}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    --connect-timeout 10 --max-time 30 \
    --data '{"input":{"method":"GET","path":"/health"}}' || true)
  if [[ -z "${RS_HEALTH}" ]]; then
    echo "Health check failed or timed out (runsync). Verify endpoint ID/URL and API key." >&2
    exit 1
  fi
fi

echo "POST ${URL} (mode=${MODE})"

# Send request and save complete response object
if [[ "${MODE}" == "DIRECT" ]]; then
  curl -sS --http1.1 --no-alpn --tlsv1.2 -X POST "${URL}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "x-api-key: ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Connection: close" \
    -H "Accept-Encoding: identity" \
    --connect-timeout 10 --max-time 180 \
    --retry 2 --retry-delay 2 --retry-connrefused \
    --data-binary @"${BODY_FILE}" | tee "${OUT_JSON}" >/dev/null || {
      echo "curl POST failed; retrying via Python requests..." >&2
      python3 - "$URL" "$RUNPOD_API_KEY" "$BODY_FILE" "$OUT_JSON" <<'PY'
import json, sys
import requests
url, key, body_path, out_path = sys.argv[1:]
with open(body_path, 'r', encoding='utf-8') as f:
    body = json.load(f)
headers = {
    'Authorization': f'Bearer {key}',
    'x-api-key': key,
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Connection': 'close',
}
resp = requests.post(url, json=body, headers=headers, timeout=180)
try:
    resp.raise_for_status()
except Exception as e:
    print(f"HTTP error: {e}", file=sys.stderr)
data = resp.json() if resp.content else {}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(data, f)
print(json.dumps(data))
PY
    } 
else
  # Build runsync wrapper payload with the BODY as a JSON object
  RS_FILE="${OUT_DIR}/runsync.json"
  # BODY_FILE already created above
  python3 - "$BODY_FILE" "$RS_FILE" <<'PY'
import json, sys
bf, rf = sys.argv[1], sys.argv[2]
with open(bf, 'r', encoding='utf-8') as f:
    body = json.load(f)
wrapped = {
    "input": {
        "method": "POST",
        "path": "/call_llm",
        "headers": {"Content-Type": "application/json"},
        "body": body,
    }
}
with open(rf, 'w', encoding='utf-8') as f:
    json.dump(wrapped, f)
PY
  curl -sS -X POST "${BASE}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    --connect-timeout 10 --max-time 180 \
    --retry 2 --retry-delay 2 --retry-connrefused \
    --data @"${RS_FILE}" | tee "${OUT_JSON}" >/dev/null
fi

echo "Saved raw response to: ${OUT_JSON}"

if command -v jq >/dev/null 2>&1; then
  echo "\n--- Pretty response (jq) ---"
  jq -C . "${OUT_JSON}"
fi

# Helper: jq queries for media-token error compatible with list or object responses
has_media_error_jq='if type=="array" then any(.[]; (.error|type=="object") and (.error.message|tostring|contains("Prompt media tokens invalid after conversion"))) else (.error|type=="object") and (.error.message|tostring|contains("Prompt media tokens invalid after conversion")) end'
no_error_jq='if type=="array" then all(.[]; (.error|type!="object")) else ((.error|not) or (.error==null) or (.error=="")) end'

# If server returned media token error, retry with alternate payload shapes using curl (no extra deps)
if command -v jq >/dev/null 2>&1 && jq -e "${has_media_error_jq}" "${OUT_JSON}" >/dev/null; then
  echo "Detected media-token template error; retrying with alternate payload shapes..."
  for VARIANT in A B C D; do
    case "$VARIANT" in
      A)
        # system + user: image then text (image)
        python3 - "$IMG_PATH" "$MIME" "$SYSTEM_PROMPT" "$INSTRUCTION" "$BODY_FILE" <<'PY'
import sys, base64, json
img_path, mime, system_prompt, instruction, out_path = sys.argv[1:]
with open(img_path,'rb') as f: b64=base64.b64encode(f.read()).decode('ascii')
data_uri=f"data:{mime};base64,{b64}"
body={"max_new_tokens":32,"temperature":0.0,"messages":[
    {"role":"system","content":system_prompt},
    {"role":"user","content":[{"type":"image","image":data_uri},{"type":"text","text":instruction}]}
]}
open(out_path,'w').write(json.dumps(body))
PY
        ;;
      B)
        # system + user: text then image (image)
        python3 - "$IMG_PATH" "$MIME" "$SYSTEM_PROMPT" "$INSTRUCTION" "$BODY_FILE" <<'PY'
import sys, base64, json
img_path, mime, system_prompt, instruction, out_path = sys.argv[1:]
with open(img_path,'rb') as f: b64=base64.b64encode(f.read()).decode('ascii')
data_uri=f"data:{mime};base64,{b64}"
body={"max_new_tokens":32,"temperature":0.0,"messages":[
    {"role":"system","content":system_prompt},
    {"role":"user","content":[{"type":"text","text":instruction},{"type":"image","image":data_uri}]}
]}
open(out_path,'w').write(json.dumps(body))
PY
        ;;
      C)
        # system + user: image then text (image_url)
        python3 - "$IMG_PATH" "$MIME" "$SYSTEM_PROMPT" "$INSTRUCTION" "$BODY_FILE" <<'PY'
import sys, base64, json
img_path, mime, system_prompt, instruction, out_path = sys.argv[1:]
with open(img_path,'rb') as f: b64=base64.b64encode(f.read()).decode('ascii')
data_uri=f"data:{mime};base64,{b64}"
body={"max_new_tokens":32,"temperature":0.0,"messages":[
    {"role":"system","content":system_prompt},
    {"role":"user","content":[{"type":"image_url","image_url":{"url":data_uri}},{"type":"text","text":instruction}]}
]}
open(out_path,'w').write(json.dumps(body))
PY
        ;;
      D)
        # user only: image then text (image)
        python3 - "$IMG_PATH" "$MIME" "$SYSTEM_PROMPT" "$INSTRUCTION" "$BODY_FILE" <<'PY'
import sys, base64, json
img_path, mime, system_prompt, instruction, out_path = sys.argv[1:]
with open(img_path,'rb') as f: b64=base64.b64encode(f.read()).decode('ascii')
data_uri=f"data:{mime};base64,{b64}"
body={"max_new_tokens":32,"temperature":0.0,"messages":[
    {"role":"user","content":[{"type":"image","image":data_uri},{"type":"text","text":instruction}]}
]}
open(out_path,'w').write(json.dumps(body))
PY
        ;;
    esac
    echo "Retry variant ${VARIANT}..."
    curl -sS --http1.1 --no-alpn --tlsv1.2 -X POST "${URL}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "x-api-key: ${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      -H "Accept: application/json" \
      -H "Connection: close" \
      -H "Accept-Encoding: identity" \
      --connect-timeout 10 --max-time 180 \
      --retry 1 --retry-delay 1 --retry-connrefused \
      --data-binary @"${BODY_FILE}" -o "${OUT_JSON}" >/dev/null || true
    if jq -e "${no_error_jq}" "${OUT_JSON}" >/dev/null; then
      echo "Variant ${VARIANT} returned without error."
      break
    fi
    if ! jq -e "${has_media_error_jq}" "${OUT_JSON}" >/dev/null; then
      echo "Variant ${VARIANT} changed error; keeping response."
      break
    fi
  done

  if command -v jq >/dev/null 2>&1; then
    echo "\n--- Pretty response (jq) ---"
    jq -C . "${OUT_JSON}"
  fi
fi

# Auto-retry with alternate payload shapes (Python, handles list/object responses)
python3 - "$URL" "$RUNPOD_API_KEY" "$OUT_JSON" "$IMG_PATH" "$MIME" "$SYSTEM_PROMPT" "$INSTRUCTION" <<'PY' || true
import sys, json, base64
try:
    import requests
except Exception:
    sys.exit(0)

url, key, out_json, img_path, mime, system_prompt, instruction = sys.argv[1:]
try:
    data = json.load(open(out_json, 'r', encoding='utf-8'))
except Exception:
    sys.exit(0)

def has_media_error(obj):
    if isinstance(obj, list):
        for it in obj:
            err = it.get('error') if isinstance(it, dict) else None
            if isinstance(err, dict) and 'Prompt media tokens invalid after conversion' in str(err.get('message','')):
                return True
        return False
    if isinstance(obj, dict):
        err = obj.get('error')
        if isinstance(err, dict) and 'Prompt media tokens invalid after conversion' in str(err.get('message','')):
            return True
    return False

if not has_media_error(data):
    sys.exit(0)

with open(img_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
data_uri = f"data:{mime};base64,{b64}"

variants = [
    # A: image then text (image)
    [{"role":"system","content":system_prompt},
     {"role":"user","content":[{"type":"image","image":data_uri}, {"type":"text","text":instruction}]}],
    # B: text then image (image)
    [{"role":"system","content":system_prompt},
     {"role":"user","content":[{"type":"text","text":instruction}, {"type":"image","image":data_uri}]}],
    # C: image then text (image_url)
    [{"role":"system","content":system_prompt},
     {"role":"user","content":[{"type":"image_url","image_url":{"url":data_uri}}, {"type":"text","text":instruction}]}],
    # D: user only, image then text
    [{"role":"user","content":[{"type":"image","image":data_uri}, {"type":"text","text":instruction}]}],
]

headers = {
    'Authorization': f'Bearer {key}',
    'x-api-key': key,
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Connection': 'close',
}

for i, msgs in enumerate(variants, 1):
    body = {"max_new_tokens": 32, "temperature": 0.0, "messages": msgs}
    try:
        r = requests.post(url, json=body, headers=headers, timeout=180)
        r.raise_for_status()
        jd = r.json() if r.content else {}
    except Exception:
        continue
    if not has_media_error(jd):
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(jd, f)
        print(f"Retried with variant {i} and succeeded.")
        break
else:
    print("All payload variants still produced media token error.")
PY

if command -v jq >/dev/null 2>&1; then
  echo "\n--- Pretty response (jq) ---"
  jq -C . "${OUT_JSON}"
fi

# Print the first response text for convenience (handles list or object shape)
if command -v jq >/dev/null 2>&1; then
  echo "\n--- First response text ---"
  jq -r 'if type=="array" then (.[0].response // "") else (.response // "") end' "${OUT_JSON}" || true
fi

# Map clicks back to original image and draw circles
python3 gta1_serve/tools/draw_clicks.py \
  --image "${IMG_PATH}" \
  --response "${OUT_JSON}" \
  --output "${OUT_ANN}" || true

echo "Annotated image (if any clicks found): ${OUT_ANN}"

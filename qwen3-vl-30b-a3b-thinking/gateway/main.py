# qwen3-vl-30b-a3b-thinking/gateway/main.py
from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, Dict, List, Tuple

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="qwen3-vl gateway", version="0.1.0")

# --- Config ----------------------------------------------------------------------
BACKEND = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "300"))
# Cached probe of backend routes
_backend_has_responses: bool | None = None


# --- Utilities -------------------------------------------------------------------
def _is_image_item(item: Dict[str, Any]) -> bool:
    # Accept either OpenAI "type": "image_url" content item or Responses "input_image"
    if "type" in item:
        t = item["type"]
        if t == "image_url":
            return True
        if t == "input_image":
            return True
    # Also allow minimal {image_url: {url: ...}}
    return "image_url" in item


def _is_text_item(item: Dict[str, Any]) -> bool:
    if "type" in item:
        return item["type"] in ("text", "input_text")
    return isinstance(item.get("text"), str)


def _normalize_to_messages(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Accept either:
      - Chat Completions/Responses-like: {"messages":[...]}
      - Responses input blocks: {"input": [{"type":"input_text"|...}, ...]}
    Return standard OpenAI 'messages' list and 'extras' (model, stream, etc.).
    """
    extras = {k: v for k, v in payload.items() if k not in ("messages", "input")}
    if "messages" in payload:
        return payload["messages"], extras

    # Responses-style "input" → single user message with content blocks
    if "input" in payload:
        input_blocks = payload["input"]
        if isinstance(input_blocks, str):
            content = [{"type": "text", "text": input_blocks}]
        elif isinstance(input_blocks, list):
            content = []
            for b in input_blocks:
                t = b.get("type")
                if t in ("input_text", "text"):
                    content.append({"type": "text", "text": b.get("text", b.get("content", ""))})
                elif t in ("input_image", "image_url"):
                    url = b.get("image_url", {}).get("url") or b.get("url")
                    if not url:
                        continue
                    content.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    # Pass unknown types verbatim; backend may ignore
                    content.append(b)
        else:
            raise HTTPException(status_code=400, detail="Unsupported 'input' format")

        messages = [{"role": "user", "content": content}]
        return messages, extras

    raise HTTPException(status_code=400, detail="Missing 'messages' or 'input' in payload")


def _multi_image_transform(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If a single user message contains N images and some text, rewrite into:
      [ user(image_1), user(image_2), ..., user(image_N), user(text) ]
    Keep all non-user messages unchanged.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role != "user":
            out.append(m)
            continue

        content = m.get("content")
        if isinstance(content, str):
            # No structured content; keep as-is
            out.append(m)
            continue

        if not isinstance(content, list):
            out.append(m)
            continue

        image_items = [it for it in content if _is_image_item(it)]
        text_items = [it for it in content if _is_text_item(it)]

        # If 0 or 1 image, don't split
        if len(image_items) <= 1:
            out.append(m)
            continue

        # Split: each image gets its own user turn
        idx = 0
        for it in image_items:
            idx += 1
            # add a tiny hint caption to help coreference (optional)
            hint = {"type": "text", "text": f"(Image {idx}/{len(image_items)})"}
            img_msg = {"role": "user", "content": [hint, it]}
            out.append(img_msg)

        # Append the original text (if any) as a final user turn
        if text_items:
            merged_text = " ".join([ti.get("text", "") for ti in text_items if isinstance(ti.get("text", ""), str)]).strip()
            if merged_text:
                out.append({"role": "user", "content": [{"type": "text", "text": merged_text}]})

        # NOTE: any other non-text, non-image blocks are ignored in split mode
    return out if out else messages


async def _probe_backend_has_responses() -> bool:
    global _backend_has_responses
    if _backend_has_responses is not None:
        return _backend_has_responses
    url = f"{BACKEND}/v1/responses"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.options(url)
            _backend_has_responses = r.status_code < 400
    except Exception:
        _backend_has_responses = False
    return _backend_has_responses


async def _stream_to_client(upstream: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in upstream.aiter_raw():
        # Pass through exactly, preserving SSE "event:" / "data:" lines and chunk boundaries
        yield chunk


def _payload_for_chat(messages: List[Dict[str, Any]], extras: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal mapping for Chat Completions
    out = {
        "model": extras.get("model", "unknown"),
        "messages": messages,
        "temperature": extras.get("temperature", 0.2),
        "top_p": extras.get("top_p", 0.9),
        "max_tokens": extras.get("max_output_tokens") or extras.get("max_tokens") or 1024,
        "stream": extras.get("stream", False),
    }
    # Remove None values to avoid confusing backends
    return {k: v for k, v in out.items() if v is not None}


# --- Routes ----------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BACKEND}/v1/models")
            ok = r.status_code == 200
            has_resp = await _probe_backend_has_responses()
            return JSONResponse({"ok": ok, "backend": BACKEND, "responses_supported": has_resp})
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/responses")
async def responses(request: Request):
    """
    Accept OpenAI Responses-style payloads OR Chat messages.
    - Performs multi-image split on user turns.
    - Streams SSE unchanged from the backend.
    - If backend lacks /v1/responses, maps to /v1/chat/completions.
    """
    payload = await request.json()
    messages, extras = _normalize_to_messages(payload)
    messages = _multi_image_transform(messages)
    stream = bool(extras.get("stream", False))

    if await _probe_backend_has_responses():
        # Forward 1:1 to backend /v1/responses (no mapping needed)
        fwd = {"messages": messages, **{k: v for k, v in extras.items() if k != "input"}}
        upstream_url = f"{BACKEND}/v1/responses"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if stream:
                upstream = await client.stream("POST", upstream_url, json=fwd)
                return StreamingResponse(_stream_to_client(upstream), media_type="text/event-stream")
            r = await client.post(upstream_url, json=fwd)
            return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    # Fallback: map to chat.completions
    chat_payload = _payload_for_chat(messages, extras)
    upstream_url = f"{BACKEND}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if stream:
            upstream = await client.stream("POST", upstream_url, json=chat_payload)
            return StreamingResponse(_stream_to_client(upstream), media_type="text/event-stream")
        r = await client.post(upstream_url, json=chat_payload)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Pass-through chat.completions that still benefit from the multi-image split.
    """
    payload = await request.json()
    messages = payload.get("messages", [])
    extras = {k: v for k, v in payload.items() if k != "messages"}
    messages = _multi_image_transform(messages)
    stream = bool(extras.get("stream", False))

    upstream_url = f"{BACKEND}/v1/chat/completions"
    chat_payload = {"messages": messages, **extras}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if stream:
            upstream = await client.stream("POST", upstream_url, json=chat_payload)
            return StreamingResponse(_stream_to_client(upstream), media_type="text/event-stream")
        r = await client.post(upstream_url, json=chat_payload)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

# qwen3-vl-30b-a3b-thinking/tests/test_gateway_responses_fallback.py
import json
import pytest
import respx
import httpx
from pathlib import Path
import importlib.util

MAIN = Path(__file__).parents[1] / "gateway" / "main.py"
spec = importlib.util.spec_from_file_location("gateway_main", MAIN)
gm = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(gm)  # type: ignore

@pytest.mark.asyncio
async def test_responses_fallback_to_chat(monkeypatch):
    # Force backend base url for the module
    monkeypatch.setenv("BACKEND_BASE_URL", "http://backend:8000")
    # Reload to rebind BACKEND
    import importlib
    importlib.reload(gm)

    # Mock: /v1/responses OPTIONS -> 404 (no responses route)
    with respx.mock(assert_all_called=True) as rs:
        rs.options("http://backend:8000/v1/responses").respond(404)
        # Backend /v1/chat/completions should be called
        rs.post("http://backend:8000/v1/chat/completions").respond(200, json={"id":"test","choices":[{"message":{"role":"assistant","content":"ok"}}]})

        app = gm.app
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.post("/v1/responses", json={
                "model":"foo",
                "input":[
                    {"type":"input_text","text":"hi"},
                    {"type":"input_image","image_url":{"url":"https://x/1.png"}},
                    {"type":"input_image","image_url":{"url":"https://x/2.png"}}
                ],
                "stream": False
            })
            assert r.status_code == 200
            data = r.json()
            assert data["id"] == "test"

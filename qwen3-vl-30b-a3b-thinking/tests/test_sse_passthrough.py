# qwen3-vl-30b-a3b-thinking/tests/test_sse_passthrough.py
import pytest
import respx
import httpx
from pathlib import Path
import importlib.util
import anyio

MAIN = Path(__file__).parents[1] / "gateway" / "main.py"
spec = importlib.util.spec_from_file_location("gateway_main", MAIN)
gm = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(gm)  # type: ignore

@pytest.mark.asyncio
async def test_sse_passthrough(monkeypatch):
    monkeypatch.setenv("BACKEND_BASE_URL", "http://backend:8000")
    import importlib
    importlib.reload(gm)

    sse_body = b"event: response.refusal.delta\n" \
               b"data: {\"type\":\"message_delta\"}\n\n" \
               b"data: [DONE]\n\n"

    with respx.mock(assert_all_called=True) as rs:
        rs.options("http://backend:8000/v1/responses").respond(200)
        route = rs.post("http://backend:8000/v1/responses").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type":"text/event-stream"})
        )

        app = gm.app
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            async with ac.stream("POST", "/v1/responses", json={"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}], "stream": True}) as r:
                assert r.status_code == 200
                chunks = [chunk async for chunk in r.aiter_bytes()]
                assert b"text/event-stream" in r.headers.get("content-type","").encode() or chunks
                joined = b"".join(chunks)
                assert b"event:" in joined and b"data:" in joined

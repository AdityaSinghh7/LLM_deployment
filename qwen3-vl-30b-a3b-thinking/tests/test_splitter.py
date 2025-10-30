# qwen3-vl-30b-a3b-thinking/tests/test_splitter.py
import json
import importlib.util
from pathlib import Path

# Load gateway/main.py utilities without importing the web server:
MAIN = Path(__file__).parents[1] / "gateway" / "main.py"
spec = importlib.util.spec_from_file_location("gateway_main", MAIN)
gm = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(gm)  # type: ignore

def test_multi_image_split_basic():
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
            {"type": "image_url", "image_url": {"url": "https://x/2.png"}},
            {"type": "text", "text": "compare them"}
        ]
    }]
    out = gm._multi_image_transform(messages)
    assert len(out) == 3
    assert out[0]["role"] == "user" and out[1]["role"] == "user" and out[2]["role"] == "user"
    assert out[0]["content"][1]["image_url"]["url"].endswith("1.png")
    assert out[1]["content"][1]["image_url"]["url"].endswith("2.png")
    assert out[2]["content"][0]["text"] == "compare them"

def test_no_split_single_image():
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
            {"type": "text", "text": "desc"}
        ]
    }]
    out = gm._multi_image_transform(messages)
    assert out == messages

def test_normalize_responses_input_blocks():
    payload = {
        "model": "foo",
        "input": [
            {"type": "input_text", "text": "please analyze"},
            {"type": "input_image", "image_url": {"url": "https://x/1.png"}},
            {"type": "input_image", "image_url": {"url": "https://x/2.png"}}
        ]
    }
    messages, extras = gm._normalize_to_messages(payload)
    assert messages and messages[0]["role"] == "user"
    assert extras["model"] == "foo"
    assert any(x.get("type") == "image_url" for x in messages[0]["content"])

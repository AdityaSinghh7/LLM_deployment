import torch
import os
# -------------------------
# System / Torch defaults
# -------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid CPU oversubscription
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENGINE_IN_BACKGROUND_THREAD", "0")
import base64
import re
from typing import Dict, List, Union
from PIL import Image
from io import BytesIO
import traceback
import argparse
import asyncio
import requests
import ray
from ray import serve
from fastapi import FastAPI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import uuid


N_REPLICAS = 2

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# -------------------------
# IO helpers
# -------------------------

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64


def data_uri_to_pil(data_uri: str) -> Image.Image:
    header, b64_str = data_uri.split(",", 1)
    img_data = base64.b64decode(b64_str)
    buffer = BytesIO(img_data)
    img = Image.open(buffer)
    return img


def extract_images(messages: List[Dict]) -> List[Image.Image]:
    images = []
    for msg in messages:
        if msg.get("role") == "user":
            for content in msg.get("content", []):
                if content.get("type") in ["image", "image_url"]:
                    if content["type"] == "image":
                        images.append(data_uri_to_pil(content["image"]).convert("RGB"))
                    else:
                        images.append(data_uri_to_pil(content["image_url"]["url"]).convert("RGB"))
    return images


# -------------------------
# Prompt builder
# -------------------------

def build_prompt_with_template(tokenizer: AutoTokenizer, messages: List[Dict]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text2, n = re.subn(
        r"<\|media_begin\|>.*?<\|media_end\|>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        text,
        flags=re.S,
    )
    if n == 0:
        raise RuntimeError("Did not find <|media_begin|>...<|media_end|> block in template.")
    return text2

# -------------------------
# Deployment
# -------------------------

def build_app(model_path: str, num_replicas: int, port: int):
    api = FastAPI(title="GTA1-32B Multi-GPU Service (High-throughput)")

    @serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={"num_gpus": 1, "num_cpus": 4},
        max_ongoing_requests=16,
    )
    class GTA1Model:
        def __init__(self, model_path: str):
            gpu_ids = ray.get_gpu_ids()
            self.gpu_id = gpu_ids[0] if gpu_ids else 0
            print(f"🔍 Ray assigned GPU IDs: {gpu_ids}")        
            # Initialize vLLM within this replica (Ray sets CUDA_VISIBLE_DEVICES)
            print(f"🔄 Initializing vLLM on GPU {self.gpu_id}[ray id] from {model_path}")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            # Tokenizer override: vLLM's Qwen2.5-VL processor expects a Qwen2 tokenizer
            tok_id_env = (
                os.environ.get("TOKENIZER_ID")
                or os.environ.get("MODEL_TOKENIZER_ID")
                or os.environ.get("HF_TOKENIZER_ID")
            )
            tokenizer_id = tok_id_env or model_path
            if tokenizer_id == model_path and ("GTA1" in model_path or model_path.startswith("Salesforce/GTA1")):
                tokenizer_id = "Qwen/Qwen2.5-VL-7B-Instruct"
            tok_rev = os.environ.get("TOKENIZER_REVISION") or os.environ.get("MODEL_TOKENIZER_REVISION")
            print(f"📝 Using tokenizer for vLLM: {tokenizer_id}{'@'+tok_rev if tok_rev else ''}")

            self.llm = LLM(
                model=model_path,
                tokenizer=tokenizer_id,
                tokenizer_mode="auto",
                trust_remote_code=True,
                dtype="bfloat16",
                limit_mm_per_prompt={"image": 1},
                max_model_len=32768,
                tensor_parallel_size=1,
                tokenizer_revision=tok_rev,
            )
            self.vllm_tokenizer = self.llm.get_tokenizer()
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model_path = model_path
            self.dtype = "bfloat16"
            print(f"✅ vLLM initialized successfully (Ray GPU Id: {self.gpu_id})")

        # ------------ batching core ------------
        @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1) # increase if GPU allows
        async def _generate_batch(self, payload: Union[Dict, List[Dict]]):
            """Build prompts, enforce single image, and call vLLM.generate."""
            if isinstance(payload, dict):
                list_of_payloads = [payload]
            else:
                list_of_payloads = payload
            request_id = uuid.uuid4().hex[:8]
            # --- Build per-sample prompt/image ---
            prompts: List[str] = []
            images_per_req: List[Image.Image] = []
            error_results = []
            early_exit = False
            for p in list_of_payloads:
                try:
                    messages = p["messages"]
                    imgs = extract_images(messages)
                    if len(imgs) != 1:
                        raise RuntimeError(f"Exactly one image is required, got {len(imgs)}")
                    prompt_text = build_prompt_with_template(self.hf_tokenizer, messages)
                    # Sanity check on tokens: 1 <|image_pad|>, no <|media_placeholder|>
                    tok = self.vllm_tokenizer
                    id_imgpad = tok.encode("<|image_pad|>", add_special_tokens=False)[0]
                    id_media = tok.encode("<|media_placeholder|>", add_special_tokens=False)[0]
                    ids = tok.encode(prompt_text, add_special_tokens=False)
                    if sum(i == id_imgpad for i in ids) != 1 or any(i == id_media for i in ids):
                        raise RuntimeError("Prompt media tokens invalid after conversion")
                    prompts.append(prompt_text)
                    images_per_req.append(imgs[0])
                except Exception as e:
                    early_exit = True
                    trace = traceback.format_exc()
                    error_results.append(
                        {
                            "response": "", 
                            "error": {
                                        "message": str(e), 
                                        "trace": trace, 
                                        'type_of_payload': str(type(payload)), 
                                        'type_of_list_of_payloads': str(type(list_of_payloads)),
                                        'type_of_p': str(type(p)),
                                        'p_keys': str(p.keys()) if isinstance(p, dict) else str(p),
                                    }, 
                            "usage": {}, 
                            "gpu_id": self.gpu_id
                        }
                     )
            if early_exit:
                return error_results
            # --- vLLM generation ---
            args_base = list_of_payloads[0]
            sp = SamplingParams(
                max_tokens=args_base.get("max_new_tokens", 512),
                temperature=args_base.get("temperature", 0.0),
                top_p=args_base.get("top_p", 0.9),
            )

            requests_list = [
                {"prompt": pr, "multi_modal_data": {"image": [im]}}
                for pr, im in zip(prompts, images_per_req)
            ]

            outs = self.llm.generate(requests_list, sampling_params=sp)

            tok = self.vllm_tokenizer
            results: List[Dict] = []
            for pr, o in zip(prompts, outs):
                text = o.outputs[0].text if o.outputs else ""
                gen_tokens = len(o.outputs[0].token_ids) if (o.outputs and hasattr(o.outputs[0], 'token_ids')) else None
                prompt_tokens = len(tok.encode(pr, add_special_tokens=False))
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": gen_tokens if gen_tokens is not None else None,
                    "total_tokens": (prompt_tokens + gen_tokens) if gen_tokens is not None else None,
                }
                results.append({
                    "response": text,
                    "error": "",
                    "usage": usage,
                    "gpu_id": self.gpu_id,
                    'bs_size_in_this_request': f"{request_id}:{len(list_of_payloads)}"
                })

            return results

        # Exposed single-call entry that joins the batch
        async def call_llm(self, payload: Dict):
            try:
                res = await self._generate_batch(payload)
                return res
            except Exception as e:
                trace = traceback.format_exc()
                return {"response": "", "error": {"message": str(e), "trace": trace}, "usage": {}, "gpu_id": self.gpu_id}

        def health(self):
            return {
                "status": "ok",
                "gpu_id": self.gpu_id,
                "dtype": self.dtype,
                "model_path": self.model_path,
            }

    model = GTA1Model.bind(model_path)

    @serve.deployment(max_ongoing_requests=96)
    @serve.ingress(api)
    class GTA1App:
        def __init__(self, model_handle):
            self.model_deployment = model_handle

        @api.get("/health")
        async def health_all(self):
            # Calling the same Serve handle N times does not guarantee each call hits a different replica
            attempts = max(8, N_REPLICAS * 4)  # oversample
            calls = [self.model_deployment.health.remote() for i in range(attempts)]
            replies = await asyncio.gather(*calls)
            # dedupe by replica_id (or by tuple(gpu_id))
            seen = {}
            for r in replies:
                seen[r.get("gpu_id", f"unknown-{len(seen)}")] = r
                if len(seen) >= N_REPLICAS:
                    break
            return {"replicas": list(seen.values())}

        @api.post("/call_llm")
        async def call_llm(self, req: Dict):
            return await self.model_deployment.call_llm.remote(req)

    return GTA1App.bind(model)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Salesforce/GTA1-32B")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3005)
    parser.add_argument("--num_replicas", type=int, default=2)
    args = parser.parse_args()
    N_REPLICAS = args.num_replicas
    # Constrain Ray's perceived resources to avoid oversubscription in containers
    # and ensure a GPU resource is registered for local scheduling.
    try:
        num_cpus = int(os.environ.get("RAY_NUM_CPUS", "4"))
    except Exception:
        num_cpus = 4
    try:
        num_gpus = int(os.environ.get("RAY_NUM_GPUS", str(max(1, args.num_replicas))))
    except Exception:
        num_gpus = max(1, args.num_replicas)

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=num_cpus, num_gpus=num_gpus)

    print(f"🚀 Starting GTA1-32B service on {args.host}:{args.port}")
    serve.start(detached=True, http_options={"host": args.host, "port": args.port})

    app = build_app(args.model_path, args.num_replicas, args.port)
    serve.run(app, name="GTA1-32B", route_prefix="/")

    # Quick health sample
    try:
        r = requests.get(f"http://0.0.0.0:{args.port}/health", timeout=5)
        print(r.json())
    except Exception as e:
        print("Health probe failed:", e)

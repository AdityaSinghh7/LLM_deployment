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
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor
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

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def smart_resize(image: Image.Image, processor) -> Image.Image:
    """Provider-style smart resize using processor attributes.

    Reads patch_size, merge_size, min_pixels, max_pixels from the model's
    image processor (Qwen2VL/OpenCUA) and resizes the image accordingly.
    """
    ip = getattr(processor, 'image_processor', processor) if processor is not None else None
    size_config = getattr(ip, 'size', {}) if (ip is not None and hasattr(ip, 'size')) else {}

    patch_size = _env_int('IMAGE_PATCH_SIZE', getattr(ip, 'patch_size', size_config.get('patch_size', 14)))
    merge_size = _env_int('IMAGE_MERGE_SIZE', getattr(ip, 'merge_size', size_config.get('merge_size', 2)))
    min_pixels = _env_int('IMAGE_MIN_PIXELS', getattr(ip, 'min_pixels', size_config.get('min_pixels', 4 * 28 * 28)))
    max_pixels = _env_int('IMAGE_MAX_PIXELS', getattr(ip, 'max_pixels', size_config.get('max_pixels', 16384 * 28 * 28)))

    effective = patch_size * merge_size
    width, height = image.size
    total_pixels = width * height

    if total_pixels < min_pixels:
        scale = (min_pixels / total_pixels) ** 0.5
        target_w = int(width * scale)
        target_h = int(height * scale)
    elif total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        target_w = int(width * scale)
        target_h = int(height * scale)
    else:
        target_w, target_h = width, height

    # Round to multiples of effective patch size and ensure at least one patch
    target_w = max((target_w // effective) * effective, effective)
    target_h = max((target_h // effective) * effective, effective)

    if target_w != width or target_h != height:
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return image


def build_app(model_path: str, num_replicas: int, port: int):
    api = FastAPI(title="GTA1-32B Multi-GPU Service (High-throughput)")
    model_actor_cpus = _env_float("MODEL_ACTOR_CPUS", 3.0)
    app_actor_cpus = _env_float("APP_ACTOR_CPUS", 0.5)

    @serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={"num_gpus": 1, "num_cpus": model_actor_cpus},
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

            # Tokenizer: default to provider tokenizer from the same repo as the model
            tok_id_env = (
                os.environ.get("TOKENIZER_ID")
                or os.environ.get("MODEL_TOKENIZER_ID")
                or os.environ.get("HF_TOKENIZER_ID")
            )
            tokenizer_id = tok_id_env or model_path
            tok_rev = os.environ.get("TOKENIZER_REVISION") or os.environ.get("MODEL_TOKENIZER_REVISION")
            print(f"📝 Using tokenizer for vLLM: {tokenizer_id}{'@'+tok_rev if tok_rev else ''}")

            # Read runtime configurable KV/cache + context settings
            max_model_len = int(os.environ.get("MAX_MODEL_LEN", "24576"))
            gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
            print(
                f"⚙️ vLLM config: max_model_len={max_model_len}, "
                f"gpu_memory_utilization={gpu_memory_utilization}"
            )

            self.llm = LLM(
                model=model_path,
                tokenizer=tokenizer_id,
                tokenizer_mode=os.environ.get("TOKENIZER_MODE", "slow"),
                trust_remote_code=True,
                dtype="bfloat16",
                limit_mm_per_prompt={"image": 1},
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=1,
                tokenizer_revision=tok_rev,
            )
            self.vllm_tokenizer = self.llm.get_tokenizer()
            # Use the same tokenizer ID/revision as vLLM for chat templating,
            # to ensure special multimodal tokens match what vLLM expects.
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
                revision=tok_rev,
            )
            # Load image processor for smart resize (provider-style)
            self.image_processor = None
            try:
                print("🖼️  Loading image processor for smart resize…")
                self.image_processor = AutoImageProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
            except Exception as e_img:
                print(f"⚠️  AutoImageProcessor not available: {e_img}")
                try:
                    generic_proc = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
                    self.image_processor = getattr(generic_proc, 'image_processor', None)
                    if self.image_processor is not None:
                        print("✅ Using image_processor from AutoProcessor")
                    else:
                        print("⚠️  AutoProcessor loaded but no image_processor field; using defaults")
                except Exception as e_auto:
                    print(f"⚠️  AutoProcessor not available either: {e_auto}\nUsing default resize heuristics.")
            self.model_path = model_path
            self.dtype = "bfloat16"
            print(f"✅ vLLM initialized successfully (Ray GPU Id: {self.gpu_id})")

            # Optional: debug print token IDs to confirm alignment between vLLM and HF tokenizers
            debug_tokens = os.environ.get("DEBUG_TOKENS", "0").lower()
            if debug_tokens in ("1", "true", "yes"):                
                try:
                    vllm_imgpad = self.vllm_tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
                except Exception as e:
                    vllm_imgpad = f"error: {e}"
                hf_imgpad = None
                try:
                    # Prefer the explicit path used elsewhere
                    hf_imgpad = self.hf_tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
                except Exception:
                    try:
                        # Fallback without kwargs for custom tokenizers
                        tmp = self.hf_tokenizer.encode("<|image_pad|>")
                        hf_imgpad = tmp[0] if isinstance(tmp, (list, tuple)) else tmp
                    except Exception:
                        try:
                            hf_imgpad = self.hf_tokenizer.convert_tokens_to_ids("<|image_pad|>")
                        except Exception as e2:
                            hf_imgpad = f"error: {e2}"
                print(f"🔎 Token alignment: <|image_pad|> vLLM={vllm_imgpad} HF={hf_imgpad}")

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
                    # Provider-style smart resize
                    resized = smart_resize(imgs[0], self.image_processor)
                    images_per_req.append(resized)
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

        async def health(self):
            return {
                "status": "ok",
                "gpu_id": self.gpu_id,
                "dtype": self.dtype,
                "model_path": self.model_path,
            }

    model = GTA1Model.bind(model_path)

    @serve.deployment(max_ongoing_requests=96, ray_actor_options={"num_cpus": app_actor_cpus})
    @serve.ingress(api)
    class GTA1App:
        def __init__(self, model_handle):
            self.model_deployment = model_handle

        @api.get("/ready")
        async def ready(self):
            return {"status": "ok"}

        @api.get("/ping")
        async def ping(self):
            return {"status": "ok"}

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

    # Prevent the driver (PID 1) from exiting so the container stays alive
    # and the detached Ray Serve app continues running.
    try:
        import signal, time
        print("🟢 Serve is running; holding process open. Press Ctrl+C to exit.")
        try:
            # On Unix, this blocks until a signal is received
            signal.pause()
        except (AttributeError, KeyboardInterrupt):
            # Fallback for platforms without signal.pause() or manual interrupt
            while True:
                time.sleep(3600)
    except Exception:
        # As a last resort, loop sleep
        import time
        while True:
            time.sleep(3600)

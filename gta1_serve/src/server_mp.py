"""
FastAPI server for GTA1-32B model using vLLM.
Single-process deployment optimized for RunPod serverless on A100 GPUs.
"""

import os
import traceback
from typing import Dict, Optional

# Set environment defaults before imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor
from vllm import LLM, SamplingParams

from utils_mm import extract_images, build_prompt_with_template, smart_resize


# -------------------------
# Configuration
# -------------------------

MODEL_ID = os.environ.get("MODEL_ID", "Salesforce/GTA1-32B")
MODEL_REVISION = os.environ.get("MODEL_REVISION", None)
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", os.environ.get("HF_TOKENIZER_ID", None)) or os.environ.get("MODEL_TOKENIZER_ID", None) or os.environ.get("MODEL_TOKENIZER", None) or None
TOKENIZER_ID = TOKENIZER_ID or os.environ.get("MODEL_ID", "Salesforce/GTA1-32B")
TOKENIZER_REVISION = os.environ.get("TOKENIZER_REVISION", None)
PORT = int(os.environ.get("PORT", "8000"))
PORT_HEALTH = int(os.environ.get("PORT_HEALTH", "8001"))


# -------------------------
# Global model instance
# -------------------------

llm_instance: Optional[LLM] = None
hf_tokenizer = None
image_processor = None
model_dtype = "bfloat16"
tokenizer_mode = os.environ.get("TOKENIZER_MODE", "slow")
try:
    # Default to 8192 if not provided
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
except Exception:
    max_model_len = 8192


def initialize_model():
    """Initialize global LLM instance and tokenizers."""
    global llm_instance, hf_tokenizer, image_processor
    
    if llm_instance is not None:
        return  # Already initialized
    
    print(f"🔄 Initializing vLLM for {MODEL_ID}...")
    
    # Load tokenizer for chat template
    print("📝 Loading tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION or MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Load image processor for smart resize
    print("🖼️  Loading image processor...")
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
    except Exception as e_img:
        print(f"⚠️  AutoImageProcessor not available for {MODEL_ID}: {e_img}")
        # Try a more generic AutoProcessor, which may bundle an image processor
        try:
            generic_proc = AutoProcessor.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
            )
            image_processor = getattr(generic_proc, "image_processor", None)
            if image_processor is not None:
                print("✅ Using image_processor from AutoProcessor")
            else:
                print("⚠️  AutoProcessor loaded but no image_processor field; using defaults")
        except Exception as e_auto:
            print(f"⚠️  AutoProcessor not available either: {e_auto}\nUsing default resize heuristics.")
            image_processor = None  # smart_resize will fallback to defaults
    
    # Initialize vLLM
    print("🚀 Loading vLLM engine...")
    llm_kwargs = dict(
        model=MODEL_ID,
        tokenizer=TOKENIZER_ID,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=True,
        dtype=model_dtype,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION or MODEL_REVISION,
    )
    llm_kwargs["max_model_len"] = max_model_len
    llm_instance = LLM(**llm_kwargs)
    
    print(f"✅ vLLM initialized successfully for {MODEL_ID}")


# -------------------------
# Request/Response models
# -------------------------

class CallLLMRequest(BaseModel):
    """Request model for /call_llm endpoint."""
    messages: list = Field(..., description="List of message dictionaries with role and content")
    max_new_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.0, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling top-p")


class CallLLMResponse(BaseModel):
    """Response model for /call_llm endpoint."""
    response: str = Field(..., description="Generated text")
    usage: Dict = Field(..., description="Token usage statistics")
    error: str = Field("", description="Error message if any")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    model: str
    dtype: str
    device: str


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(
    title="GTA1-32B vLLM Service",
    description="Single-process FastAPI + vLLM deployment for RunPod",
    version="0.1.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    initialize_model()


@app.get("/ping", response_class=PlainTextResponse)
async def ping():
    """
    Health check endpoint for RunPod load balancer.
    Returns 200 OK if service is running.
    """
    return "ok"


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Detailed health check with model information.
    """
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return HealthResponse(
        status="ok",
        model=MODEL_ID,
        dtype=model_dtype,
        device="cuda"
    )


@app.post("/call_llm", response_model=CallLLMResponse)
async def call_llm(request: CallLLMRequest):
    """
    Main inference endpoint.
    
    Accepts messages with exactly one image via data URI and generates response.
    """
    try:
        # Ensure model is initialized
        if llm_instance is None:
            initialize_model()
        
        # Extract images from messages
        images = extract_images(request.messages)
        
        # Enforce exactly one image
        if len(images) != 1:
            raise ValueError(f"Exactly one image is required, got {len(images)}")
        
        # Smart resize the image
        resized_image = smart_resize(images[0], image_processor)
        
        # Build prompt with template and token rewrite
        prompt_text = build_prompt_with_template(hf_tokenizer, request.messages)
        
        # Validate prompt tokens (sanity check)
        vllm_tokenizer = llm_instance.get_tokenizer()
        id_imgpad = vllm_tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
        id_media = vllm_tokenizer.encode("<|media_placeholder|>", add_special_tokens=False)[0]
        ids = vllm_tokenizer.encode(prompt_text, add_special_tokens=False)
        
        if sum(i == id_imgpad for i in ids) != 1 or any(i == id_media for i in ids):
            raise RuntimeError("Prompt media tokens invalid after conversion")
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        # Generate with vLLM
        outputs = llm_instance.generate(
            [{"prompt": prompt_text, "multi_modal_data": {"image": [resized_image]}}],
            sampling_params=sampling_params
        )
        
        # Extract response
        output = outputs[0]
        generated_text = output.outputs[0].text if output.outputs else ""
        
        # Calculate token usage
        gen_tokens = len(output.outputs[0].token_ids) if (output.outputs and hasattr(output.outputs[0], 'token_ids')) else None
        prompt_tokens = len(ids)
        
        usage = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens if gen_tokens is not None else 0,
            "total_tokens": (prompt_tokens + gen_tokens) if gen_tokens is not None else prompt_tokens,
        }
        
        return CallLLMResponse(
            response=generated_text,
            usage=usage,
            error=""
        )
        
    except Exception as e:
        trace = traceback.format_exc()
        print(f"❌ Error in /call_llm: {str(e)}\n{trace}")
        
        return CallLLMResponse(
            response="",
            usage={},
            error=f"{str(e)}\n{trace}"
        )


# -------------------------
# Main entry point
# -------------------------

if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting GTA1-32B service on port {PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        workers=1,
        log_level="info"
    )

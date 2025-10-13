"""
Multi-modal utilities for GTA1-32B model serving.
Handles image processing, data URI parsing, and prompt template building.
"""

import base64
import re
from typing import Dict, List
from PIL import Image
from io import BytesIO


def data_uri_to_pil(data_uri: str) -> Image.Image:
    """
    Convert a data URI to a PIL Image.
    
    Args:
        data_uri: Data URI string (e.g., "data:image/png;base64,...")
        
    Returns:
        PIL Image object
    """
    header, b64_str = data_uri.split(",", 1)
    img_data = base64.b64decode(b64_str)
    buffer = BytesIO(img_data)
    img = Image.open(buffer)
    return img


def extract_images(messages: List[Dict]) -> List[Image.Image]:
    """
    Extract images from messages array.
    Only processes user role messages with image or image_url content types.
    
    Args:
        messages: List of message dictionaries with role and content
        
    Returns:
        List of PIL Image objects
    """
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


def build_prompt_with_template(tokenizer, messages: List[Dict]) -> str:
    """
    Build prompt using tokenizer's chat template and rewrite media tokens.
    
    Converts <|media_begin|>...<|media_end|> blocks to 
    <|vision_start|><|image_pad|><|vision_end|> for GTA1-32B compatibility.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of message dictionaries
        
    Returns:
        Formatted prompt string with rewritten media tokens
        
    Raises:
        RuntimeError: If no media block found in template
    """
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


def smart_resize(image: Image.Image, processor) -> Image.Image:
    """
    Smart resize for GTA1-32B using AutoImageProcessor parameters.
    
    Resizes image according to model's patch_size, merge_size, min_pixels, 
    and max_pixels constraints as specified in the GTA1 model card.
    
    Args:
        image: PIL Image to resize
        processor: AutoImageProcessor with size configuration
        
    Returns:
        Resized PIL Image
    """
    # Get size config from processor
    size_config = processor.size if hasattr(processor, 'size') else {}
    
    # Extract parameters (with defaults matching GTA1-32B)
    patch_size = size_config.get('patch_size', 14)
    merge_size = size_config.get('merge_size', 2)
    min_pixels = size_config.get('min_pixels', 4 * 28 * 28)
    max_pixels = size_config.get('max_pixels', 16384 * 28 * 28)
    
    # Calculate effective patch size after merging
    effective_patch_size = patch_size * merge_size
    
    # Get original dimensions
    width, height = image.size
    aspect_ratio = width / height
    
    # Calculate target dimensions within pixel constraints
    total_pixels = width * height
    
    if total_pixels < min_pixels:
        # Scale up to meet minimum
        scale_factor = (min_pixels / total_pixels) ** 0.5
        target_width = int(width * scale_factor)
        target_height = int(height * scale_factor)
    elif total_pixels > max_pixels:
        # Scale down to meet maximum
        scale_factor = (max_pixels / total_pixels) ** 0.5
        target_width = int(width * scale_factor)
        target_height = int(height * scale_factor)
    else:
        target_width = width
        target_height = height
    
    # Round to nearest multiple of effective patch size
    target_width = (target_width // effective_patch_size) * effective_patch_size
    target_height = (target_height // effective_patch_size) * effective_patch_size
    
    # Ensure at least one patch
    target_width = max(target_width, effective_patch_size)
    target_height = max(target_height, effective_patch_size)
    
    # Resize if dimensions changed
    if target_width != width or target_height != height:
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    
    return image


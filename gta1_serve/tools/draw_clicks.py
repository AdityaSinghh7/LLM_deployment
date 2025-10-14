#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple, Union

try:
    from PIL import Image, ImageDraw
except Exception as e:
    print("Pillow (PIL) is required. Install with: pip install pillow")
    raise


CLICK_KWARGS_REGEX = re.compile(r"pyautogui\.click\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\)")
CLICK_POSARGS_REGEX = re.compile(r"pyautogui\.click\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def extract_clicks_from_text(text: str) -> List[Tuple[int, int]]:
    clicks: List[Tuple[int, int]] = []
    for x, y in CLICK_KWARGS_REGEX.findall(text or ""):
        clicks.append((int(x), int(y)))
    for x, y in CLICK_POSARGS_REGEX.findall(text or ""):
        clicks.append((int(x), int(y)))
    return clicks


def extract_clicks_from_results(result_items: Union[Dict, List[Dict]]) -> List[Tuple[int, int]]:
    clicks: List[Tuple[int, int]] = []
    if isinstance(result_items, dict):
        result_items = [result_items]
    for item in result_items:
        if isinstance(item, dict) and item.get("error"):
            continue
        text = item.get("response", "") if isinstance(item, dict) else ""
        clicks.extend(extract_clicks_from_text(text))
    return clicks


def smart_resize(height: int,
                 width: int,
                 factor: int = 28,
                 min_pixels: int = 1000,
                 max_pixels: int = 10**12) -> Tuple[int, int]:
    """Approximate provider smart resize used server-side.

    - Keep aspect ratio.
    - Ensure dims are multiples of `factor`.
    - Nudge area to be >= min_pixels (rare for screenshots) and <= max_pixels.
    """
    area = height * width
    if area <= 0:
        return height, width
    scale = 1.0
    if area < min_pixels:
        scale = math.sqrt(min_pixels / float(area))
    # Note: default max_pixels here is huge, so this rarely triggers
    if area * scale * scale > max_pixels:
        scale = math.sqrt(max_pixels / float(area))
    new_h = max(factor, int(round((height * scale) / factor)) * factor)
    new_w = max(factor, int(round((width * scale) / factor)) * factor)
    return new_h, new_w


def compute_resized_dims_for_server_mapping(image_path: str) -> Tuple[int, int, int, int]:
    with Image.open(image_path) as im:
        width, height = im.size
    resized_h, resized_w = smart_resize(height, width, factor=28, min_pixels=1000, max_pixels=10**12)
    return width, height, int(resized_w), int(resized_h)


def map_clicks_to_original(clicks_resized: List[Tuple[int, int]],
                           original_w: int,
                           original_h: int,
                           resized_w: int,
                           resized_h: int) -> List[Tuple[int, int]]:
    if resized_w <= 0 or resized_h <= 0:
        return []
    scale_x = original_w / float(resized_w)
    scale_y = original_h / float(resized_h)
    mapped: List[Tuple[int, int]] = []
    for x, y in clicks_resized:
        mapped_x = int(round(x * scale_x))
        mapped_y = int(round(y * scale_y))
        mapped.append((mapped_x, mapped_y))
    return mapped


def draw_circles_on_image(image_path: str,
                          points: List[Tuple[int, int]],
                          output_path: str,
                          radius: int = 8,
                          outline: Tuple[int, int, int] = (255, 0, 0),
                          fill: Tuple[int, int, int] = (0, 255, 0),
                          width: int = 3) -> None:
    if not points:
        print("No clicks found in response; skipping annotation.")
        return
    with Image.open(image_path).convert("RGB") as img:
        drawer = ImageDraw.Draw(img)
        for (x, y) in points:
            left = x - radius
            top = y - radius
            right = x + radius
            bottom = y + radius
            drawer.ellipse([(left, top), (right, bottom)], outline=outline, fill=fill, width=width)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
    print(f"Annotated image saved to: {output_path} (points drawn: {len(points)})")


def main():
    ap = argparse.ArgumentParser(description="Parse grounding response, map clicks back to original image, and draw circles.")
    ap.add_argument("--image", required=True, help="Path to original image")
    ap.add_argument("--response", required=True, help="Path to saved response JSON from /call_llm")
    ap.add_argument("--output", default="gta1_serve/output/annotated.png", help="Output annotated image path")
    args = ap.parse_args()

    with open(args.response, "r", encoding="utf-8") as f:
        data = json.load(f)

    clicks_resized = extract_clicks_from_results(data)
    if not clicks_resized:
        print("No pyautogui.click(...) coordinates found in response.")
        return

    ow, oh, rw, rh = compute_resized_dims_for_server_mapping(args.image)
    mapped = map_clicks_to_original(clicks_resized, ow, oh, rw, rh)
    draw_circles_on_image(args.image, mapped, args.output)


if __name__ == "__main__":
    main()


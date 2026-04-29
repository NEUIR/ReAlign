#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the number of vision tokens for every image under the corpus
images directory, using Qwen's token-counting rules.
"""

import math
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR  = PROJECT_ROOT / "dataset"

# Root directory to scan recursively for images
ROOT_DIR = DATASET_DIR / "OpenDocVQA-Corpus" / "images"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".ico", ".avif",
}


def token_calculate(image_path: str) -> tuple[int, int, int]:
    """
    Compute the aligned (h_bar, w_bar) and token count for an image
    following Qwen's vision token rules.

    Returns:
        (h_bar, w_bar, token_count)
        token_count includes +2 for <|vision_bos|> and <|vision_eos|>.
    """
    with Image.open(image_path) as img:
        width, height = img.width, img.height

    # Align to the nearest multiple of 28
    h_bar = round(height / 28) * 28
    w_bar = round(width  / 28) * 28

    min_pixels = 28 * 28 * 4       # 4 tokens minimum
    max_pixels = 1280 * 28 * 28    # 1 280 tokens maximum

    # Rescale while maintaining approximate aspect ratio, then re-align
    if h_bar * w_bar > max_pixels:
        beta  = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / 28) * 28
        w_bar = math.floor(width  / beta / 28) * 28
    elif h_bar * w_bar < min_pixels:
        beta  = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / 28) * 28
        w_bar = math.ceil(width  * beta / 28) * 28

    tokens = (h_bar * w_bar) // (28 * 28)  # number of patches
    return h_bar, w_bar, int(tokens + 2)    # +2 for vision BOS/EOS tokens


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise SystemExit(f"Directory not found: {root.resolve()}")

    total_imgs   = 0
    total_tokens = 0

    for p in root.rglob("*"):
        if not is_image_file(p):
            continue
        try:
            h_bar, w_bar, tokens = token_calculate(str(p))
            total_imgs   += 1
            total_tokens += tokens
            with Image.open(p) as im:
                orig_w, orig_h = im.width, im.height
            print(
                f"[OK] {p.as_posix()}  "
                f"orig=({orig_w}x{orig_h})  "
                f"aligned=({h_bar}x{w_bar})  "
                f"tokens={tokens}"
            )
        except Exception as e:
            print(f"[FAIL] {p.as_posix()}  error: {e}")

    print("\n===== Summary =====")
    print(f"Images counted : {total_imgs}")
    print(f"Total tokens   : {total_tokens}")


if __name__ == "__main__":
    main()

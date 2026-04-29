# -*- coding: utf-8 -*-
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image

# Allowed image extensions (extend as needed)
_IMG_EXTS = r"jpg|jpeg|png|webp|bmp|tif|tiff|gif|svg|ico|avif|heic|heif"
# Fallback pattern: extract tokens matching "a/b/c.ext"
_IMG_TOKEN_RE = re.compile(rf"([A-Za-z0-9_\-./]+?\.(?:{_IMG_EXTS}))", re.I)


def parse_list_field(val) -> List[str]:
    """Normalise a DataFrame cell into a flat list of strings ['path1', 'path2', ...]."""
    # 1) Already an iterable container
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]

    # 2) NumPy array (including nested columns from pyarrow / fastparquet)
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return [str(x).strip() for x in val.tolist() if str(x).strip()]
    except Exception:
        pass

    # 3) Null / empty
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []

    # 4) Try JSON / Python literal first
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(s)
            if isinstance(obj, (list, tuple, set)):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            continue

    # 5) Handle NumPy repr format: "['a' 'b']" — extract quoted tokens
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", s)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]

    # 6) Last resort: extract tokens by image extension
    tokens = _IMG_TOKEN_RE.findall(s)
    return [t.strip() for t in tokens if t.strip()]


def add_syc_to_path(orig: str, idx: int | None = None) -> str:
    """
    Append '_syc' to every path component and stem.
    If idx is given, the file stem becomes '<stem>_syc_<idx>'.

    Examples:
        a/b/c.png  -> a_syc/b_syc/c_syc.png      (idx=None)
        a/b/c.png  -> a_syc/b_syc/c_syc_0.png    (idx=0)
    """
    p     = Path(orig)
    parts = list(p.parts)
    new_parts = []
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            stem, suf = os.path.splitext(part)
            tag = f"{stem}_syc" if idx is None else f"{stem}_syc_{idx}"
            new_parts.append(tag + suf)
        else:
            new_parts.append(part + "_syc")
    return Path(*new_parts).as_posix()


def smart_resize(image_path: str) -> Tuple[int, int]:
    """
    Compute the (h_bar, w_bar) aligned dimensions following Qwen's resize rules.

    Returns:
        (h_bar, w_bar) — note the order is (height, width).
    """
    with Image.open(image_path) as im:
        height, width = im.height, im.width

    h_bar = round(height / 28) * 28
    w_bar = round(width  / 28) * 28

    min_pixels = 28 * 28 * 4        # 4 tokens minimum
    max_pixels = 1280 * 28 * 28     # 1 280 tokens maximum

    if h_bar * w_bar > max_pixels:
        beta  = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / 28) * 28
        w_bar = math.floor(width  / beta  / 28) * 28
    elif h_bar * w_bar < min_pixels:
        beta  = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / 28) * 28
        w_bar = math.ceil(width  * beta  / 28) * 28

    return h_bar, w_bar


def _clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def crop_xyxy_resized(orig_img_path: str, xyxy: List[float], out_path: str):
    """
    Resize the source image according to Qwen's rules, then crop using
    coordinates that are relative to the *resized* image.

    Pillow conventions used:
        resize(size) expects (width, height)  -> pass (w_bar, h_bar)
        crop(box)    expects (left, top, right, bottom)
    """
    h_bar, w_bar = smart_resize(orig_img_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(orig_img_path) as im0:
        im = im0.resize((w_bar, h_bar), resample=Image.BICUBIC)
        x1, y1, x2, y2 = _clamp_xyxy(*xyxy, w_bar, h_bar)
        crop = im.crop((x1, y1, x2, y2))
        crop.save(out_p)

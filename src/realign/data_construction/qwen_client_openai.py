# -*- coding: utf-8 -*-
"""
OpenAI-compatible client for multimodal grounding and text judge calls.
Works with any OpenAI-compatible API endpoint.

Configure BASE_URL and set the OPENAI_API_KEY environment variable:
    export OPENAI_API_KEY="your-api-key"
"""

import os
import time
import base64
from pathlib import Path
from typing import List
from mimetypes import guess_type
from openai import OpenAI

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR  = PROJECT_ROOT / "dataset"

# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://api.studio.nebius.com/v1/"  # replace with your provider's endpoint

client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

GROUNDER_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"   # multimodal vision model
JUDGE_MODEL_NAME    = "Qwen/Qwen2.5-72B-Instruct"       # text-only judge model

TEMPERATURE         = 0.7
JUDGE_TEMP          = 0.0
MAX_RETRIES         = 7
RETRY_BACKOFF       = 2.0
MAX_IMAGES_PER_REQ  = 10   # recommended upper bound for image_url blocks per request


def _encode_image_to_data_url(path: str) -> str:
    """Read a local image file and return a base64 data URL for use in image_url.url."""
    with open(path, "rb") as f:
        b = f.read()
    mime, _ = guess_type(path)
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_grounding(user_text: str, image_paths: List[str]) -> str:
    """
    Multimodal grounding call (OpenAI-compatible).

    Args:
        user_text:    Text prompt (e.g. rendered PROMPT_USER).
        image_paths:  List of local image file paths (up to MAX_IMAGES_PER_REQ).

    Returns:
        Model response text (typically a JSON string).
    """
    images = (image_paths or [])[:MAX_IMAGES_PER_REQ]
    content_blocks = []

    for p in images:
        abs_path = os.path.abspath(p)
        if not os.path.exists(abs_path):
            continue
        data_url = _encode_image_to_data_url(abs_path)
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })

    content_blocks.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": content_blocks},
    ]

    last_err = None
    for i in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROUNDER_MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if i < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** i)
            else:
                raise last_err


def call_judge(user_text: str) -> str:
    """
    Text-only judge call (OpenAI-compatible).

    Args:
        user_text:  Fully rendered prompt (e.g. PROMPT_LLM_JUDGE.format(...)).

    Returns:
        Model response text (typically a JSON string).
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_text},
    ]

    last_err = None
    for i in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=messages,
                temperature=JUDGE_TEMP,
                max_tokens=512,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if i < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** i)
            else:
                raise last_err

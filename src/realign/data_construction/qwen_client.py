# -*- coding: utf-8 -*-
"""
DashScope client for multimodal grounding and text judge calls.
Uses Alibaba Cloud's DashScope API (qwen2.5-vl-72b-instruct).

Set the DASHSCOPE_API_KEY environment variable before running:
    export DASHSCOPE_API_KEY="your-api-key"
"""

import os
import time
from pathlib import Path
from typing import List

from dashscope import MultiModalConversation, Generation
import dashscope

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR  = PROJECT_ROOT / "dataset"

# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
dashscope.base_http_api_url = BASE_URL

DASHSCOPE_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
MODEL_NAME         = "qwen2.5-vl-72b-instruct"
TEMPERATURE        = 0.7
MAX_RETRIES        = 3
RETRY_BACKOFF      = 2.0

JUDGE_MODEL_NAME   = "qwen2.5-72b-instruct"
JUDGE_TEMPERATURE  = 0


def call_grounding(user_text: str, image_paths: List[str]) -> str:
    """
    Multimodal grounding call via DashScope.

    Args:
        user_text:    Text prompt (e.g. rendered PROMPT_USER).
        image_paths:  List of absolute local image file paths.

    Returns:
        Model response text (typically a JSON string).
    """
    user_content = []
    for p in image_paths:
        print(p)
        abs_path = os.path.abspath(p)
        user_content.append({"image": f"file://{abs_path}"})

    user_content.append({"text": user_text})

    messages = [
        {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
        {"role": "user",   "content": user_content},
    ]

    last_err = None
    for i in range(MAX_RETRIES + 1):
        try:
            resp = MultiModalConversation.call(
                api_key=DASHSCOPE_API_KEY,
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
            )
            return resp["output"]["choices"][0]["message"].content[0]["text"].strip()
        except Exception as e:
            last_err = e
            if i < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** i)
            else:
                raise last_err


def call_judge(user_text: str) -> str:
    """
    Text-only judge call via DashScope.

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
            resp = Generation.call(
                model=JUDGE_MODEL_NAME,
                messages=messages,
                result_format="message",
                temperature=JUDGE_TEMPERATURE,
                api_key=DASHSCOPE_API_KEY,
            )
            try:
                return resp.output.choices[0].message.content.strip()
            except Exception:
                return resp["output"]["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            if i < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** i)
            else:
                raise last_err

# -*- coding: utf-8 -*-
"""
Build the synthetic grounding dataset from OpenDocVQA query parquets.

For each query, the script:
  1. Resolves the relevant document images.
  2. Calls the grounding model to obtain bounding boxes.
  3. Crops sub-regions (based on Qwen-resized coordinates).
  4. Appends results to an output CSV incrementally.
"""

import os
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from prompt import PROMPT_USER
from qwen_client import call_grounding
from utils import parse_list_field, add_syc_to_path, crop_xyxy_resized

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR  = PROJECT_ROOT / "dataset"

INPUT_PARQUET = DATASET_DIR / "OpenDocVQA" / "data"          # directory of query parquet(s)
IMAGE_ROOT    = DATASET_DIR / "OpenDocVQA-Corpus" / "images" # extracted corpus images
OUTPUT_CSV    = PROJECT_ROOT / "synthetic_data" / "OpenDocVQA-Query-1.csv"

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
FLUSH_EVERY = 10
N           = 10000
SEED        = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Strip "Instruct: ... Query:" prefix, keep only the plain question
_QRY_RE = re.compile(r'(?is)\bquery\s*:\s*(.*)\Z')


def extract_pure_question(q: str) -> str:
    s = str(q).replace("\r\n", "\n").strip()
    m = _QRY_RE.search(s)
    if m:
        return m.group(1).strip()
    s2 = re.sub(r'(?is)^\s*instruct:.*?\bquery\s*:\s*', '', s, count=1)
    return s2.strip()


def render_prompt(query_text: str) -> str:
    return PROMPT_USER.replace("{query}", query_text)


def extract_json(text: str) -> Dict[str, Any]:
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip().lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and ("boxes" in obj or "regions" in obj):
            return obj
    except Exception:
        pass
    try:
        start = s.index("{")
        end   = s.rindex("}") + 1
        obj   = json.loads(s[start:end])
        if isinstance(obj, dict) and ("boxes" in obj or "regions" in obj):
            return obj
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}; text preview: {s[:200]}...")


def to_abs(root: str, rel_path: str) -> str:
    """Return absolute path; pass through if already absolute."""
    return rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)


def load_done_ids(out_csv: Path) -> set:
    done = set()
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("query_id"):
                    done.add(row["query_id"])
    return done


def append_rows(out_csv: Path, rows: List[dict]):
    fieldnames = [
        "query_id", "query",
        "relevant_doc_ids", "relevant_doc_ids_bbox",
        "describe", "dataset_names",
        "model_boxes",
        "answers",
    ]
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_parquet(INPUT_PARQUET)

    required = {"query_id", "query", "relevant_doc_ids", "dataset_names"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Parquet missing columns: {missing}")

    df = df.sample(n=min(N, len(df)), random_state=SEED)

    out_csv  = Path(OUTPUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_csv)

    image_root_str = str(IMAGE_ROOT)
    buf, saved, skipped, failed = [], 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building crops", unit="row"):
        qid: str       = str(row["query_id"]).strip()
        question: str  = extract_pure_question(row["query"])
        imgs_rel: List[str] = parse_list_field(row["relevant_doc_ids"])
        dsets: List[str]    = parse_list_field(row["dataset_names"])
        answers = row["answers"]

        if qid in done_ids:
            skipped += 1
            continue
        if not imgs_rel:
            failed += 1
            print(f"[WARN] {qid}: no images, skipping")
            continue
        if len(imgs_rel) > 1:
            print(f"[INFO] {qid}: multi-image sample ({len(imgs_rel)}), skipping")
            continue

        imgs_abs: List[str] = [to_abs(image_root_str, p) for p in imgs_rel]

        user_text = render_prompt(question)

        try:
            raw = call_grounding(user_text, imgs_abs)
        except Exception as e:
            failed += 1
            print(f"[ERROR] {qid}: grounding call failed: {e}")
            continue

        try:
            obj = extract_json(raw)
        except Exception as e:
            failed += 1
            print(f"[ERROR] {qid}: JSON parse failed: {e}")
            continue

        bbox_paths_rel: List[str]        = []
        desc_list: List[str]             = []
        model_boxes: List[List[float]]   = []

        if "boxes" in obj:
            base_img_rel = imgs_rel[0]
            base_img_abs = imgs_abs[0]
            for b in obj.get("boxes", []):
                xyxy = b.get("area")
                desc = str(b.get("describe", "")).strip()
                if not xyxy or len(xyxy) != 4:
                    continue

                try:
                    model_boxes.append([float(v) for v in xyxy[:4]])
                except Exception:
                    pass

                out_rel = add_syc_to_path(base_img_rel, idx=len(bbox_paths_rel))
                out_abs = to_abs(image_root_str, out_rel)

                try:
                    crop_xyxy_resized(base_img_abs, xyxy, out_abs)
                except Exception as e:
                    print(f"[WARN] {qid}: crop failed for {base_img_rel} {xyxy}: {e}")
                    continue

                bbox_paths_rel.append(out_rel)
                desc_list.append(desc or "")

        elif "regions" in obj:
            print(f"[INFO] {qid}: legacy 'regions' format not handled, skipping")
            continue

        out_row = {
            "query_id":             qid,
            "query":                question,
            "relevant_doc_ids":     json.dumps(imgs_rel,    ensure_ascii=False),
            "model_boxes":          json.dumps(model_boxes, ensure_ascii=False),
            "relevant_doc_ids_bbox": json.dumps(bbox_paths_rel, ensure_ascii=False),
            "describe":             json.dumps(desc_list,   ensure_ascii=False),
            "dataset_names":        json.dumps([f"{d}_syc" for d in dsets], ensure_ascii=False),
            "answers":              answers,
        }
        buf.append(out_row)
        saved += 1

        if len(buf) >= FLUSH_EVERY:
            append_rows(out_csv, buf)
            buf.clear()

    if buf:
        append_rows(out_csv, buf)

    print(f"[DONE] saved={saved}, skipped={skipped}, failed={failed}, out={out_csv.resolve()}")


if __name__ == "__main__":
    main()

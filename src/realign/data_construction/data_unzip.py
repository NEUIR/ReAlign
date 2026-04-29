#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract images embedded in OpenDocVQA-Corpus parquet shards into a local
images directory (dataset/OpenDocVQA-Corpus/images/).

Run from any working directory:
    python src/realign/data_construction/data_unzip.py
"""

import pandas as pd
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pyarrow as pa
except ImportError:
    pa = None

# ---------------------------------------------------------------------------
# Path configuration — resolved relative to this file so it works regardless
# of the current working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR  = PROJECT_ROOT / "dataset"

CORPUS_PARQUET_DIR = DATASET_DIR / "OpenDocVQA-Corpus" / "data"
OUTPUT_ROOT        = DATASET_DIR / "OpenDocVQA-Corpus" / "images"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".ico", ".heic", ".heif", ".avif",
}


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------

def extract_image_bytes(val) -> bytes:
    """
    Extract raw image bytes from a DataFrame cell that may be stored as:
    - native bytes / bytearray / memoryview
    - dict containing a 'bytes' key (common in HuggingFace parquet exports)
    - numpy.ndarray with dtype uint8
    - pyarrow Buffer / BinaryScalar
    """
    if val is None:
        raise ValueError("image value is None")

    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val)

    if isinstance(val, dict):
        for k in ("bytes", "data", "value", "image"):
            b = val.get(k)
            if isinstance(b, (bytes, bytearray, memoryview)):
                return bytes(b)

    if np is not None and isinstance(val, np.ndarray):
        if val.dtype == np.uint8:
            return val.tobytes()

    if pa is not None:
        if hasattr(val, "as_py"):
            val_py = val.as_py()
            if isinstance(val_py, (bytes, bytearray, memoryview)):
                return bytes(val_py)
            if isinstance(val_py, dict):
                b = val_py.get("bytes")
                if isinstance(b, (bytes, bytearray, memoryview)):
                    return bytes(b)
        if isinstance(val, pa.Buffer):
            return val.to_pybytes()

    # Last-resort: handle repr-style strings like b'\x89PNG...'
    if isinstance(val, str) and val.startswith(("b'", 'b"')):
        try:
            return eval(val)  # safe only for trusted data sources
        except Exception:
            pass

    raise TypeError(
        f"Unsupported image cell type: {type(val)}; "
        f"value preview: {str(val)[:60]}..."
    )


def process_parquet(parquet_path: Path, output_root: Path) -> tuple[int, int]:
    """Extract all images from one parquet shard to output_root/<doc_id>."""
    print(f"Processing: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    required_cols = {"doc_id", "image"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Parquet missing required columns: {missing}")

    saved, failed = 0, 0
    for i, row in df.iterrows():
        doc_id    = str(row["doc_id"]).strip()
        img_cell  = row["image"]

        try:
            img_bytes = extract_image_bytes(img_cell)
        except Exception as e:
            failed += 1
            print(f"  [FAIL] idx={i} doc_id={doc_id}: {e}")
            continue

        out_path = output_root / doc_id
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            saved += 1
            if saved % 100 == 0:
                print(f"  [OK] saved {saved} images (last: {out_path.name})")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] write idx={i} -> {out_path}: {e}")

    print(f"  -> saved={saved}, failed={failed}\n")
    return saved, failed


def count_images(root: Path) -> int:
    """Recursively count image files under root."""
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# Optional: inspect query / corpus parquet schema
# ---------------------------------------------------------------------------

def inspect_query_parquet():
    """Print schema and a few rows from the OpenDocVQA query parquet(s)."""
    import pandas as pd
    query_dir = DATASET_DIR / "OpenDocVQA" / "data"
    df = pd.read_parquet(query_dir)
    print(f"Query rows : {len(df)}")
    print(df.info())
    print(df.head(3))


def inspect_corpus_parquet(shard: int = 0):
    """Print schema and a few rows from one corpus parquet shard."""
    parquet_files = sorted(CORPUS_PARQUET_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in: {CORPUS_PARQUET_DIR}")
    df = pd.read_parquet(parquet_files[shard])
    print(f"Corpus shard [{shard}] rows : {len(df)}")
    print(df.info())
    print(df.head(3))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parquet_files = sorted(CORPUS_PARQUET_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in: {CORPUS_PARQUET_DIR}\n"
            "Make sure you have run:\n"
            "  huggingface-cli download --repo-type dataset yanghaoir/ReAlign-Set "
            "--local-dir ./dataset"
        )

    print(f"Found {len(parquet_files)} parquet shard(s) in {CORPUS_PARQUET_DIR}")
    print(f"Output directory : {OUTPUT_ROOT}\n")

    total_saved, total_failed = 0, 0
    for pf in parquet_files:
        s, f = process_parquet(pf, OUTPUT_ROOT)
        total_saved  += s
        total_failed += f

    total_images = count_images(OUTPUT_ROOT)
    print("===== Summary =====")
    print(f"Total saved  : {total_saved}")
    print(f"Total failed : {total_failed}")
    print(f"Total images : {total_images}")
    print(f"Output root  : {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()

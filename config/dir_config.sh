# =============================================================================
# Data and Model Directories
# =============================================================================

# --- Phi-3 Vision (MODEL_PATH for Phi-3 training and encode scripts)
# Download: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
export REALIGN_PHI3_MODEL_DIR="/path/to/Phi-3-vision-128k-instruct"

# --- VDocRetriever Phi-3 LoRA pretrained weights (LORA_PATH for Phi-3 training only)
# Download: https://huggingface.co/NTT-hil-insight/VDocRetriever-Phi3-vision-pretrained
export REALIGN_PHI3_LORA_PRETRAINED_DIR="/path/to/VDocRetriever-Phi3-vision-pretrained"

# --- Qwen2.5-VL-7B (MODEL_PATH for Qwen training and QWEN in encode_qwen scripts)
# Download: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
export REALIGN_QWEN_MODEL_DIR="/path/to/Qwen2.5-VL-7B-Instruct"

# --- Qwen training: optional LoRA base path (leave empty to skip loading pretrained LoRA)
export REALIGN_QWEN_LORA_PATH=""

# --- OpenDocVQA query set root (QUERY_PATH in encode_query* and search_eval scripts)
# Download: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA
export REALIGN_OPEN_DOC_VQA_DIR="/path/to/OpenDocVQA"

# --- OpenDocVQA-Corpus root (CORPUS_PATH in encode_shard* scripts; each dataset is a subdirectory)
# Download: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus
export REALIGN_OPEN_DOC_VQA_CORPUS_ROOT="/path/to/OpenDocVQA-Corpus"

# --- Training: --dataset_path / --corpus_path (shared by Phi-3 and Qwen)
# dataset can be a path relative to the repo root; corpus is typically the data/ subdir of OpenDocVQA-Corpus
# Download: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus/tree/main/data
export REALIGN_TRAIN_DATASET_PATH="/path/to/ReAlign-Trainset"
export REALIGN_TRAIN_CORPUS_PATH="/path/to/OpenDocVQA-Corpus/data"

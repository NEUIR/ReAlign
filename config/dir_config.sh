# =============================================================================
# Data and Model Directories
# =============================================================================

# Resolve the absolute path of the project root (one level above this config/ directory)
_REALIGN_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REALIGN_ROOT="$(cd "${_REALIGN_CONFIG_DIR}/.." && pwd)"

# --- Phi-3 Vision (MODEL_PATH for Phi-3 training and encode scripts)
# Download: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
export REALIGN_PHI3_MODEL_DIR="${_REALIGN_ROOT}/dataset/Phi3"

# --- VDocRetriever Phi-3 LoRA pretrained weights (LORA_PATH for Phi-3 training only)
# Download: https://huggingface.co/NTT-hil-insight/VDocRetriever-Phi3-vision-pretrained
export REALIGN_PHI3_LORA_PRETRAINED_DIR="${_REALIGN_ROOT}/dataset/VDocRetriever-Phi3-vision-pretrained"

# --- Qwen2.5-VL-7B (MODEL_PATH for Qwen training and QWEN in encode_qwen scripts)
# Download: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
export REALIGN_QWEN_MODEL_DIR="Qwen2.5-VL-7B-Instruct"

# --- Qwen training: optional LoRA base path (leave empty to skip loading pretrained LoRA)
export REALIGN_QWEN_LORA_PATH=""

# --- OpenDocVQA query set root (QUERY_PATH in encode_query* and search_eval scripts)
# Download: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA
export REALIGN_OPEN_DOC_VQA_DIR="${_REALIGN_ROOT}/dataset/OpenDocVQA"

# --- OpenDocVQA-Corpus root (CORPUS_PATH in encode_shard* scripts; each dataset is a subdirectory)
# Download: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus
export REALIGN_OPEN_DOC_VQA_CORPUS_ROOT="${_REALIGN_ROOT}/dataset/OpenDocVQA-Corpus"

# --- Training: --dataset_path / --corpus_path (shared by Phi-3 and Qwen)
# Download: https://huggingface.co/datasets/yanghaoir/ReAlign-Trainset
export REALIGN_TRAIN_DATASET_PATH="${_REALIGN_ROOT}/dataset/train_data"
# Download: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus/tree/main/data
export REALIGN_TRAIN_CORPUS_PATH="${_REALIGN_ROOT}/dataset/OpenDocVQA-Corpus/data"

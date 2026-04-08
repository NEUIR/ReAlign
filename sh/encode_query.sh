#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL> <DATASET> [BATCH_SIZE]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATASET=$3
BATCH_SIZE=${4:-2}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

eval "$(conda shell.bash hook)"
conda activate reproduce

source "${REPO_DIR}/config/dir_config.sh"
QUERY_PATH="${REALIGN_OPEN_DOC_VQA_DIR}"
PHI3="${REALIGN_PHI3_MODEL_DIR}"

EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
LORA="${REPO_DIR}/outputs/${MODEL}"

mkdir -p "${EMBEDDING_OUTPUT_DIR}"

echo "[Query] GPU=${CUDA_VISIBLE_DEVICES} Dataset=${DATASET} BatchSize=${BATCH_SIZE}"

python -m realign.realignretriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${PHI3} \
  --lora_name_or_path ${LORA} \
  --lora \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --query_max_len 256 \
  --dataset_name ${QUERY_PATH} \
  --dataset_config ${DATASET} \
  --dataset_split test \
  --encode_output_path "${EMBEDDING_OUTPUT_DIR}/query-${DATASET}.pkl"

echo "[Query] ${DATASET} encoding done."

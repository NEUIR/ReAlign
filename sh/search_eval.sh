#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL> <DATASET>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATASET=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

eval "$(conda shell.bash hook)"
conda activate reproduce

source "${REPO_DIR}/config/dir_config.sh"
QUERY_PATH="${REALIGN_OPEN_DOC_VQA_DIR}"

EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
RESULT_DIR="${REPO_DIR}/result/${MODEL}"

mkdir -p "${RESULT_DIR}"

if [ "$DATASET" == "docvqa" ]; then
    DEPTH=700
else
    DEPTH=1000
fi

echo "[Eval] Dataset=${DATASET} Depth=${DEPTH}"

python -m realign.realignretriever.driver.search \
    --query_reps "${EMBEDDING_OUTPUT_DIR}/query-${DATASET}.pkl" \
    --document_reps "${EMBEDDING_OUTPUT_DIR}/corpus.${DATASET}.*.pkl" \
    --depth ${DEPTH} \
    --batch_size 64 \
    --save_text \
    --save_ranking_to "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.txt"

python -m realign.utils.format.convert_result_to_trec \
    --input "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.txt" \
    --output "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.trec" \
    --remove_query

python -m realign.utils.format.convert_qas_to_trec_qrels \
    --dataset_name ${QUERY_PATH} \
    --dataset_config ${DATASET} \
    --output "${EMBEDDING_OUTPUT_DIR}/qrels.${DATASET}.txt"

echo "=================================================="
echo "  Evaluation results: ${DATASET} (depth=${DEPTH})"
echo "=================================================="

python -m pyserini.eval.trec_eval -c \
    -mrecip_rank \
    -mmap \
    -mrecall.1,5,10 \
    -mndcg_cut.1,5,10 \
    "${EMBEDDING_OUTPUT_DIR}/qrels.${DATASET}.txt" \
    "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.trec" \
    | tee -a "${RESULT_DIR}/eval.txt"

echo ""
echo "[Eval] ${DATASET} evaluation done."
echo "==================================================" >> "${RESULT_DIR}/eval.txt"
echo "Dataset: ${DATASET}  Depth: ${DEPTH}  Model: ${MODEL}" >> "${RESULT_DIR}/eval.txt"
echo "==================================================" >> "${RESULT_DIR}/eval.txt"

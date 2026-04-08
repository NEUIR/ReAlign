#!/bin/bash
set -e

BG_PIDS=()

cleanup() {
    echo ""
    echo "  Signal received, cleaning up background processes..."
    for PID in "${BG_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            pkill -TERM -P "$PID" 2>/dev/null
            kill -TERM "$PID" 2>/dev/null
        fi
    done
    sleep 1
    for PID in "${BG_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            pkill -KILL -P "$PID" 2>/dev/null
            kill -KILL "$PID" 2>/dev/null
        fi
    done
    echo "  All background processes cleaned up."
    exit 130
}

trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 2 ]; then
    echo "Error: insufficient arguments."
    echo "Usage: bash $0 <MODEL> <GPU_IDS> [DATASETS]"
    echo ""
    echo "Examples:"
    echo "  bash $0 model1 0,1,2,3"
    echo "  bash $0 model1 1,2,3,5,7 docvqa,slidevqa"
    echo "  bash $0 model1 1,2,3,5,7 [docvqa,slidevqa]"
    exit 1
fi

MODEL=$1
GPU_IDS_STR=$2

DEFAULT_DATASETS="docvqa,infovqa,chartqa,slidevqa,plotqa,arxivqa"
DATASETS_RAW=${3:-${DEFAULT_DATASETS}}
DATASETS_STR=$(echo "${DATASETS_RAW}" | tr -d '[]')

IFS=',' read -ra GPU_IDS <<< "${GPU_IDS_STR}"
NUM_GPUS=${#GPU_IDS[@]}

IFS=',' read -ra DATASETS <<< "${DATASETS_STR}"

BATCH_SIZE=8

echo "=============================================="
echo "  ReAlign Multi-GPU Evaluation"
echo "=============================================="
echo "  Model:    ${MODEL}"
echo "  GPU:      ${GPU_IDS_STR} (${NUM_GPUS} total)"
echo "  Datasets: ${DATASETS_STR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Conda env: reproduce"
echo "=============================================="
echo ""

EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
RESULT_DIR="${REPO_DIR}/result/${MODEL}"
LOG_DIR="${REPO_DIR}/log/eval-${MODEL}"
mkdir -p "${EMBEDDING_OUTPUT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

TOTAL_DATASETS=${#DATASETS[@]}
CURRENT=0
FAILED_DATASETS=()

for DATASET in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${CURRENT}/${TOTAL_DATASETS}] Dataset: ${DATASET}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "  Removing old ${DATASET} embedding files..."
    rm -f "${EMBEDDING_OUTPUT_DIR}/corpus.${DATASET}."*.pkl
    rm -f "${EMBEDDING_OUTPUT_DIR}/query-${DATASET}.pkl"

    echo "  [Step 1/3] Corpus encoding (${NUM_GPUS} shards in parallel)..."
    PIDS=()
    SHARD_FAILED=0

    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_ID=${GPU_IDS[$i]}
        LOG_FILE="${LOG_DIR}/${DATASET}.corpus.shard${i}.log"
        > "${LOG_FILE}"

        bash "${SCRIPT_DIR}/encode_shard.sh" \
            "${GPU_ID}" "${MODEL}" "${DATASET}" "${NUM_GPUS}" "${i}" "${BATCH_SIZE}" \
            > "${LOG_FILE}" 2>&1 &

        PIDS+=($!)
        BG_PIDS+=($!)
        echo "    Shard ${i}/${NUM_GPUS} -> GPU ${GPU_ID} (PID: ${PIDS[-1]})"
    done

    MONITOR_LOG="${LOG_DIR}/${DATASET}.corpus.shard0.log"
    while true; do
        ALL_DONE=1
        for PID in "${PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then
                ALL_DONE=0
                break
            fi
        done

        if [ -f "${MONITOR_LOG}" ]; then
            PROGRESS=$(tr '\r' '\n' < "${MONITOR_LOG}" | grep -oP '\s*\d+%\|[^|]+\|\s*\d+/\d+\s*\[.*?\]' | tail -1)
            if [ -n "$PROGRESS" ]; then
                printf "\r    [Shard 0 progress]%s" "$PROGRESS"
            fi
        fi

        [ ${ALL_DONE} -eq 1 ] && break
        sleep 3
    done
    if [ -f "${MONITOR_LOG}" ]; then
        PROGRESS=$(tr '\r' '\n' < "${MONITOR_LOG}" | grep -oP '\s*\d+%\|[^|]+\|\s*\d+/\d+\s*\[.*?\]' | tail -1)
        if [ -n "$PROGRESS" ]; then
            printf "\r    [Shard 0 progress]%s\n" "$PROGRESS"
        else
            printf "\n"
        fi
    else
        printf "\n"
    fi

    for ((i=0; i<${#PIDS[@]}; i++)); do
        if ! wait ${PIDS[$i]}; then
            echo "    ✗ Shard ${i} encoding failed! Log: ${LOG_DIR}/${DATASET}.corpus.shard${i}.log"
            SHARD_FAILED=1
        fi
    done

    if [ ${SHARD_FAILED} -ne 0 ]; then
        echo "  ✗ Corpus encoding failed, skipping ${DATASET}"
        FAILED_DATASETS+=("${DATASET}")
        continue
    fi
    echo "  ✓ All corpus shards encoded."

    echo "  [Step 2/3] Query encoding (GPU ${GPU_IDS[0]})..."
    QUERY_LOG="${LOG_DIR}/${DATASET}.query.log"

    if ! bash "${SCRIPT_DIR}/encode_query.sh" \
        "${GPU_IDS[0]}" "${MODEL}" "${DATASET}" "${BATCH_SIZE}" \
        > "${QUERY_LOG}" 2>&1; then
        echo "  ✗ Query encoding failed! Log: ${QUERY_LOG}"
        FAILED_DATASETS+=("${DATASET}")
        continue
    fi
    echo "  ✓ Query encoding done."

    echo "  [Step 3/3] Retrieval & evaluation (GPU ${GPU_IDS[0]})..."
    EVAL_LOG="${LOG_DIR}/${DATASET}.eval.log"

    if ! bash "${SCRIPT_DIR}/search_eval.sh" \
        "${GPU_IDS[0]}" "${MODEL}" "${DATASET}" \
        2>&1 | tee "${EVAL_LOG}"; then
        echo "  ✗ Evaluation failed! Log: ${EVAL_LOG}"
        FAILED_DATASETS+=("${DATASET}")
        continue
    fi
    echo "  ✓ ${DATASET} evaluation done."
    echo ""
done

echo ""
echo "=============================================="
echo "  Evaluation Summary"
echo "=============================================="
echo "  Model:    ${MODEL}"
echo "  GPU:      ${GPU_IDS_STR}"
echo "  Total datasets: ${TOTAL_DATASETS}"
echo "  Succeeded: $((TOTAL_DATASETS - ${#FAILED_DATASETS[@]}))"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "  Failed: ${#FAILED_DATASETS[@]} (${FAILED_DATASETS[*]})"
    echo "  Log dir: ${LOG_DIR}"
    echo "=============================================="
    exit 1
else
    echo "  Failed: 0"
    echo ""
    echo "  Results: ${RESULT_DIR}/eval.txt"
    echo "  Log dir: ${LOG_DIR}"
    echo "=============================================="
fi

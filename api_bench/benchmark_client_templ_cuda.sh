#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

if [ -z "$BENCHMARK_LLM" ]; then
    echo "[ERROR] please set env BENCHMARK_LLM to the absolute path of benchmark_serving.py script"
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    echo "[ERROR] please set env DATASET_PATH to the dataset path"
    exit 1
fi

if [ -z "$VLLM_SERVER_HOST" ];then
    echo "[ERROR] please set env VLLM_SERVER_HOST to the vllm server host"
    exit 1
fi

if [ -z "$VLLM_SERVER_PORT" ];then
    echo "[ERROR] please set env VLLM_SERVER_PORT to the vllm server port"
    exit 1
fi

MODEL_SIZE=$1

if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ -z "$TP_SIZE" ]; then
    TP_SIZE=1
fi

PROMPTS=$3

if [ -z "$PROMPTS" ]; then
    PROMPTS=1000
fi

CLIENTS=$4

if [ -z "$CLIENTS" ]; then
    CLIENTS=1
fi

STOP_TIME=$5

if [ -z "$STOP_TIME" ]; then
    STOP_TIME=300
fi

MODEL_DIR="$PERF_BASE_PATH/../../hf_models/llama-${MODEL_SIZE}b-hf"

# python python/benchmark_serving.py --host 10.198.31.25 --port 8000 --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --num-threads 64 --disable-tqdm --thread-stop-time 30

CMD="python ${BENCHMARK_LLM} \
--host $VLLM_SERVER_HOST \
--port $VLLM_SERVER_PORT \
--model $MODEL_DIR \
--dataset-name sharegpt \
--dataset-path $DATASET_PATH \
--num-prompts $PROMPTS \
--num-threads $CLIENTS \
--thread-stop-time $STOP_TIME"

echo "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"

eval "$CMD"
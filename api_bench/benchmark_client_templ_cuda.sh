#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

if [ -z "$BENCHMARK_LLM" ]; then
    BENCHMARK_LLM="$PERF_BASE_PATH/python/benchmark_serving.py"
fi

if [ -z "$DATASET_PATH" ]; then
    DATASET_PATH="$PERF_BASE_PATH/ShareGPT_V3_unfiltered_cleaned_split.json"
fi

if [ -z "$VLLM_SERVER_URL" ];then
    VLLM_SERVER_URL="http://10.198.31.25:8000"
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

if [ -z "$MODEL_DIR" ];then
    MODEL_DIR="$PERF_BASE_PATH/../../hf_models/llama-${MODEL_SIZE}b-hf"
fi

# python python/benchmark_serving.py --host 10.198.31.25 --port 8000 --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --num-threads 64 --disable-tqdm --thread-stop-time 30

CMD="python ${BENCHMARK_LLM} \
--base-url $VLLM_SERVER_URL \
--model $MODEL_DIR \
--dataset-name sharegpt \
--dataset-path $DATASET_PATH \
--num-prompts $PROMPTS \
--num-threads $CLIENTS \
--thread-stop-time $STOP_TIME"

echo "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"

eval "$CMD"
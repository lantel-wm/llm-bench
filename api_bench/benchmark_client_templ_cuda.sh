#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

if [ -z "$BENCHMARK_LLM" ]; then
    BENCHMARK_LLM="$PERF_BASE_PATH/python/benchmark_serving.py"
fi

if [ -z "$DATASET_PATH" ]; then
    # DATASET_PATH="$PERF_BASE_PATH/ShareGPT_V3_unfiltered_cleaned_split.json"
    DATASET_PATH="$PERF_BASE_PATH/datasets/samples_1024.json"
fi

if [ -z "$VLLM_SERVER_URL" ];then
    VLLM_SERVER_URL="10.198.31.25:8000"
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

TURNS=$4

if [ -z "$TURNS" ]; then
    TURNS=1
fi

CLIENTS=$5

if [ -z "$CLIENTS" ]; then
    CLIENTS=1
fi

RAMP_UP_TIME=$6

if [ -z "$RAMP_UP_TIME" ]; then
    RAMP_UP_TIME=1
fi

STOP_TIME=$7

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
--num-turns $TURNS \
--num-threads $CLIENTS \
--ramp-up-time $RAMP_UP_TIME \
--thread-stop-time $STOP_TIME"

echo "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"
INFO "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"

eval "$CMD"
#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

export MODEL_DIR="/mnt/llm/llm_perf/hf_models/llama-${MODEL_SIZE}b-hf"
export BENCHMARK_LLM="$PERF_BASE_PATH/python/benchmark_serving_num_clients.py"
export DATASET_PATH="$PERF_BASE_PATH/datasets/samples_1024.json"
export PPL_SERVER_URL="127.0.0.1:23333"
export VLLM_SERVER_URL="http://127.0.0.1:8000"
export SERVER_URL="$VLLM_SERVER_URL" # vllm
# export SERVER_URL="$PPL_SERVER_URL" # ppl
export OPMX_MODEL_PATH="/mnt/llm/llm_perf/opmx_models"
export HF_MODEL_PATH="/mnt/llm/llm_perf/hf_models"
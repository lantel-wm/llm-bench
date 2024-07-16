#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

if [ -z "$VLLM_SERVER_URL" ];then
    VLLM_SERVER_URL="http://10.198.31.25:8000"
    VLLM_SERVER_HOST=$(echo $VLLM_SERVER_URL | sed -E 's|http://([^:/]+).*|\1|')
    VLLM_SERVER_PORT=$(echo $VLLM_SERVER_URL | sed -E 's|.*:([0-9]+)|\1|')
fi


MODEL_SIZE=$1

if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ -z "$TP_SIZE" ]; then
    TP_SIZE=1
fi

CLIENTS=$3

if [ -z "$CLIENTS" ]; then
    CLIENTS=1
fi

MODEL_DIR="$PERF_BASE_PATH/../../hf_models/llama-${MODEL_SIZE}b-hf"

# python -m vllm.entrypoints.openai.api_server --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --swap-space 16 --disable-log-requests --enforce-eager --host 10.198.31.25  --port 8000

CMD="nohup python -m vllm.entrypoints.openai.api_server \
--model $MODEL_DIR \
--tensor-parallel-size $TP_SIZE \
--swap-space 16 \
--disable-log-requests \
--enforce-eager \
--host $VLLM_SERVER_HOST \
--port $VLLM_SERVER_PORT \
--uvicorn-log-level warning \
>> log/server.log 2>&1 &"

# --disable-log-stats \

echo "SERVER STARTING: MODEL${MODEL_SIZE}B TP${TP_SIZE} HOST${HOST} PORT${PORT} -> $CMD"
INFO "SERVER STARTING: MODEL${MODEL_SIZE}B TP${TP_SIZE} HOST${HOST} PORT${PORT} -> $CMD"

eval "$CMD"

SERVER_PID=$!

if [ -z "$SERVER_PID" ]; then
    echo "[ERROR] SERVER START FAILED"
    ERROR "SERVER START FAILED"
else
    echo "SERVER PID: $SERVER_PID"
    INFO "SERVER PID: $SERVER_PID"
fi

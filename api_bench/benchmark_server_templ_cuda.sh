#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
LOG_DIR="$PERF_BASE_PATH/log/benchmark_all_cuda.log"

if [ -z "$VLLM_SERVER_URL" ];then
    VLLM_SERVER_URL="http://10.198.31.25:8000"
fi

function check_server_status() {
    local url="${VLLM_SERVER_URL}/v1/models"
    response=$(curl -s "${url}")
    
    if [[ $response == *"\"object\":\"list\""* ]]; then
        return 0
    else
        return 1
    fi
}

MODEL_SIZE=$1
TP_SIZE=$2
SERVER_PID=$(bash "$PERF_BASE_PATH/start_server.sh" "$MODEL_SIZE" "$TP_SIZE")

if [ ! -n "$SERVER_PID" ]; then
    echo "[ERROR] SERVER START FAILED"
else
    SERVER_PID=$(echo "$SERVER_PID" | grep -oP "SERVER PID: \K[0-9]+")

    attempt=30
    while [ $attempt -gt 0 ]; do
        # echo "Attempt $attempt"
        sleep 5
        attempt=$((attempt-1))
        check_server_status
        if [ $? -eq 0 ]; then
            break
        fi
    done

    if [ $attempt -eq 0 ]; then
        echo "[ERROR] SERVER START TIMEOUT"
        if [ -f "$LOG_DIR" ]; then
            echo "[ERROR] SERVER START TIMEOUT" >> "$LOG_DIR"
        fi
    else 
        echo "SERVER STARTED $SERVER_PID"
    fi
fi
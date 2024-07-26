#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

MODEL_SIZE=$1
TP_SIZE=$2
CLIENTS=$3
BACKEND=$4
SERVER_PID=$(bash "$PERF_BASE_PATH/start_${BACKEND}_server.sh" "$MODEL_SIZE" "$TP_SIZE" "$CLIENTS")

if [ -z "$SERVER_URL" ];then
    if [ "$BACKEND" == "vllm" ]; then
        SERVER_URL="http://127.0.0.1:8000"
    elif [ "$BACKEND" == "ppl" ]; then
        SERVER_URL="127.0.0.1:23333"
    fi
fi

function check_server_status() {
    if [ "$BACKEND" == "vllm" ]; then
        CMD="python $PERF_BASE_PATH/python/check_server_status.py --server-url $SERVER_URL --backend vllm --model /mnt/llm/llm_perf/hf_models/llama-${MODEL_SIZE}b-hf"
        status=$(eval "$CMD")
    elif [ "$BACKEND" == "ppl" ]; then
        CMD="python $PERF_BASE_PATH/python/check_server_status.py --server-url $SERVER_URL --backend ppl"
        status=$(eval "$CMD")
    fi

    if [ -z "$status" ]; then
        echo "[ERROR] SERVER STATUS CHECK FAILED"
    fi

    if [ "$status" == "OK" ]; then
        return 0
    else
        return 1
    fi
}



if [ ! -n "$SERVER_PID" ]; then
    echo "[ERROR] SERVER START FAILED"
else
    SERVER_PID=$(echo "$SERVER_PID" | grep -oP "SERVER PID: \K[0-9]+")

    attempt=90
    errno=0
    while [ $attempt -gt 0 ]; do
        # echo "Attempt $attempt"
        sleep 10
        attempt=$((attempt-1))
        check_server_status

        if [ $? -eq 0 ]; then
            break
        fi

        if ! ps -p "$SERVER_PID" > /dev/null; then
            echo "SERVER PID $SERVER_PID is not running"
            ERROR "SERVER PID $SERVER_PID is not running"
            errno=1
            unset SERVER_PID
            break
        fi
    done

    if [ $attempt -eq 0 ]; then
        echo "[ERROR] SERVER START TIMEOUT"
        ERROR "SERVER START TIMEOUT"
    elif [ $errno -eq 1 ]; then
        echo "[ERROR] SERVER START FAILED"
        ERROR "SERVER START FAILED"
    else
        echo "SERVER STARTED $SERVER_PID"
    fi
fi
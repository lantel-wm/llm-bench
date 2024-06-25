#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

if [ ! -n "$VLLM_SERVER_HOST" ];then
    echo "[ERROR] please set env VLLM_SERVER_HOST to the vllm server host"
    exit 1
fi

if [ ! -n "$VLLM_SERVER_PORT" ];then
    echo "[ERROR] please set env VLLM_SERVER_PORT to the vllm server port"
    exit 1
fi


function check_server_status() {
    local url="http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1/models"
    response=$(curl -s "${url}")
    
    if [[ $response == *"\"object\":\"list\""* ]]; then
        # echo "Server started successfully with PID ${SERVER_PID}"
        return 0
    else
        # echo "Server is not ready"
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

    attempt=20
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
    else 
        echo "SERVER STARTED $SERVER_PID"
    fi
fi
#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
BACKEND=$1

if [ -z "$BACKEND" ]; then
    BACKEND="vllm"
fi

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    TURNS=$4
    CLIENTS=$5
    RAMP_UP_TIME=$6
    STOP_TIME=$7
    MODE=$8

    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} TURNS$TURNS CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME ${MODE^^}]"
    INFO "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} TURNS$TURNS CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME ${MODE^^}]"
    RES=$(bash "$PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh" "${MODEL_SIZE}" "${GPUS}" "${PROMPTS}" "${TURNS}" "${CLIENTS}" "${RAMP_UP_TIME}" "${STOP_TIME}" "${BACKEND}" | grep "CSV format output")
    RES=${RES##*:}

    if [ -z "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        INFO "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$CLIENTS,$MODE,$RES" >> "$PERF_BASE_PATH/result/benchmark_${BACKEND}_all_cuda_result.csv"
    fi
}

function launch_server_and_test() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    RAMP_UP_TIME=$4
    STOP_TIME=$5
    MODE=$6

    SERVER_PID=$(bash "$PERF_BASE_PATH"/benchmark_server_templ_cuda.sh "$MODEL_SIZE" "$GPUS" "$CLIENTS" "$BACKEND" | grep -o "[0-9]\+")
    SERVER_PID=${SERVER_PID##*:}

    if [ -z "$SERVER_PID" ]; then
        echo "[ERROR] SERVER START FAILED"
        exit 1
    else
        echo "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
        INFO "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
    fi

    for CLIENTS in "${_NUM_CLIENTS_LIST[@]}"; do
        for TURNS in "${_NUM_TURNS_LIST[@]}"; do
            unittest "$MODEL_SIZE" "$GPUS" "$PROMPTS" "$TURNS" "$CLIENTS" "$RAMP_UP_TIME" "$STOP_TIME" "$MODE"
        done
    done

    kill -9 "$SERVER_PID"
    INFO "SERVER STOPPED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
}


source "$PERF_BASE_PATH/logging.sh"
create_log "$BACKEND"

_MODE_LIST=(fp16)
_7B_TP_LIST=(1)
_13B_TP_LIST=(2)
_65B_TP_LIST=(8)
_70B_TP_LIST=(8)
# _70B_TP_LIST=(4)

# _NUM_CLIENTS_LIST=(16 32 64 128 256 512)
# _NUM_CLIENTS_LIST=(1 5 10 20 30 40 50 60 70 80 100)
_NUM_CLIENTS_LIST=(10 20 30 40 50 100 200 300)
# _NUM_CLIENTS_LIST=(60)
# _NUM_TURNS_LIST=(1 2 4 8 16)
_NUM_TURNS_LIST=(1)

for MODE in "${_MODE_LIST[@]}"; do

for GPUS in "${_7B_TP_LIST[@]}"; do
    # model_size tp num_requests ramp_up_time stop_time mode
    launch_server_and_test 7 "$GPUS" 1024 1 300 "$MODE"
done

# for GPUS in "${_13B_TP_LIST[@]}"; do
#     launch_server_and_test 13 "$GPUS" 10000 512 1 300 "$MODE"
# done

# for GPUS in "${_65B_TP_LIST[@]}"; do
#     launch_server_and_test 65 "$GPUS" 10000 512 1 300 "$MODE"
# done

# for GPUS in "${_70B_TP_LIST[@]}"; do
#     launch_server_and_test 70 "$GPUS" 10000 512 1 300 "$MODE"
# done

done
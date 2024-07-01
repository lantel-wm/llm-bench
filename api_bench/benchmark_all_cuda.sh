#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    CLIENTS=$4
    RAMP_UP_TIME=$5
    STOP_TIME=$6
    MODE=$7

    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME ${MODE^^}]"
    INFO "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME ${MODE^^}]"
    RES=$(bash "$PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh" "${MODEL_SIZE}" "${GPUS}" "${PROMPTS}" "${CLIENTS}" "${RAMP_UP_TIME}" "${STOP_TIME}" | grep "CSV format output")
    RES=${RES##*:}

    if [ -z "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        INFO "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$CLIENTS,$MODE,$RES" >> "$PERF_BASE_PATH/result/benchmark_all_cuda_result.csv"
    fi
}

function launch_server_and_test() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    CLIENTS=$4
    RAMP_UP_TIME=$5
    STOP_TIME=$6
    MODE=$7

    SERVER_PID=$(bash "$PERF_BASE_PATH"/benchmark_server_templ_cuda.sh "$MODEL_SIZE" "$GPUS" "$CLIENTS" | grep -o "[0-9]\+")
    SERVER_PID=${SERVER_PID##*:}

    if [ -z "$SERVER_PID" ]; then
        echo "[ERROR] SERVER START FAILED"
        exit 1
    else
        echo "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
        INFO "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
    fi

    for CLIENTS in "${_NUM_CLIENTS_LIST[@]}"; do
        unittest "$MODEL_SIZE" "$GPUS" "$PROMPTS" "$CLIENTS" "$RAMP_UP_TIME" "$STOP_TIME" "$MODE"
    done

    kill -9 "$SERVER_PID"
    INFO "SERVER STOPPED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
}



source "$PERF_BASE_PATH/logging.sh"
create_log


_MODE_LIST=(fp16)
_7B_TP_LIST=(1)
_13B_TP_LIST=(2)
_65B_TP_LIST=(8)
_70B_TP_LIST=(8)
# _70B_TP_LIST=(4)

_NUM_CLIENTS_LIST=(512)

for MODE in "${_MODE_LIST[@]}"; do

for GPUS in "${_7B_TP_LIST[@]}"; do
    # model_size tp num_prompts num_clients ramp_up_time stop_time mode
    launch_server_and_test 7 "$GPUS" 10000 512 1 300 "$MODE"
done

for GPUS in "${_13B_TP_LIST[@]}"; do
    launch_server_and_test 13 "$GPUS" 10000 512 1 300 "$MODE"
done

for GPUS in "${_65B_TP_LIST[@]}"; do
    launch_server_and_test 65 "$GPUS" 10000 512 1 300 "$MODE"
done

# for GPUS in "${_70B_TP_LIST[@]}"; do
#     launch_server_and_test 70 "$GPUS" 10000 512 1 300 "$MODE"
# done

done
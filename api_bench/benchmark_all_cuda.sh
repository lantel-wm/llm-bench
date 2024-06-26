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
    RES=$(bash "$PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh" "${MODEL_SIZE}" "${GPUS}" "${PROMPTS}" "${CLIENTS}" "${RAMP_UP_TIME}" "${STOP_TIME}" | grep "CSV format output")
    RES=${RES##*:}

    if [ -z "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$CLIENTS,$MODE,$RES" >> "$PERF_BASE_PATH/benchmark_all_cuda_result.csv"        
    fi
}

function launch_server_and_test() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    RAMP_UP_TIME=$4
    STOP_TIME=$5
    MODE=$6

    SERVER_PID=$(bash "$PERF_BASE_PATH"/benchmark_server_templ_cuda.sh "$MODEL_SIZE" "$GPUS" | grep -o "[0-9]\+")
    SERVER_PID=${SERVER_PID##*:}

    if [ -z "$SERVER_PID" ]; then
        echo "[ERROR] SERVER START FAILED"
        exit 1
    else
        echo "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
    fi

    for CLIENTS in "${_NUM_CLIENTS_LIST[@]}"; do
        unittest "$MODEL_SIZE" "$GPUS" "$PROMPTS" "$CLIENTS" "$RAMP_UP_TIME" "$STOP_TIME" "$MODE"
    done

    kill -9 "$SERVER_PID"
}

# io_tps prefill_tps decode_tps
echo "model_size(B),tp,num_clients,mode,success_rate,qps,avg_inlen,avg_outlen,o_tps,io_tps,\
min_ttft(ms),max_ttft(ms),mean_ttft(ms),median_ttft(ms),p90_ttft(ms),p99_ttft(ms),\
min_tpot(ms),max_tpot(ms),mean_tpot(ms),median_tpot(ms),p90_tpot(ms),p99_tpot(ms),\
min_e2e(ms),max_e2e(ms),mean_e2e(ms),median_e2e(ms),p90_e2e(ms),p99_e2e(ms)" > "$PERF_BASE_PATH/benchmark_all_cuda_result.csv"

_MODE_LIST=(fp16)
_7B_TP_LIST=(1)
_13B_TP_LIST=(2)
_65B_TP_LIST=(8)
_70B_TP_LIST=(4 8)

# _NUM_CLIENTS_LIST=(1 2 4 8 16 32 64 128 256 512)
_NUM_CLIENTS_LIST=(64)

for MODE in "${_MODE_LIST[@]}"; do

for GPUS in "${_7B_TP_LIST[@]}"; do
    # model_size tp num_prompts ramp_up_time stop_time mode
    launch_server_and_test 7 "$GPUS" 10000 1 300 "$MODE"
done

# for GPUS in "${_13B_TP_LIST[@]}"; do
#     launch_server_and_test 13 "$GPUS" 1000 1 300 "$MODE"
# done

# for GPUS in "${_65B_TP_LIST[@]}"; do
#     launch_server_and_test 65 "$GPUS" 1000 10 "$MODE"
# done

# for GPUS in "${_70B_TP_LIST[@]}"; do
#     launch_server_and_test 70 "$GPUS" 1000 10 "$MODE"
# done

done
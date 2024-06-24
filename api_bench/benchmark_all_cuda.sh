SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    CLIENTS=$4
    STOP_TIME=$5
    MODE=$6
    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} CLIENTS$CLIENTS STOP_TIME$STOP_TIME ${MODE^^}]"
    RES=`bash $PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh ${MODEL_SIZE} ${GPUS} ${PROMPTS} ${CLIENTS} ${STOP_TIME} | grep "CSV format output"`
    RES=${RES##*:}

    if [ ! -n "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$CLIENTS,$MODE,$RES" >> $PERF_BASE_PATH/benchmark_all_cuda_result.csv        
    fi
}

echo "model_size(B),tp,num_clients,mode,success_rate,rps,i_tps,o_tps,\
min_ttft(ms),max_ttft(ms),mean_ttft(ms),median_ttft(ms),p90_ttft(ms),p99_ttft(ms),\
min_tpot(ms),max_tpot(ms),mean_tpot(ms),median_tpot(ms),p90_tpot(ms),p99_tpot(ms),\
min_tpr(ms),max_tpr(ms),mean_tpr(ms),median_tpr(ms),p90_tpr(ms),p99_tpr(ms)" > $PERF_BASE_PATH/benchmark_all_cuda_result.csv

_MODE_LIST=(fp16)
_7B_TP_LIST=(1)
_13B_TP_LIST=(2)
_65B_TP_LIST=(8)
_70B_TP_LIST=(8)

_NUM_CLIENTS_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)

for MODE in ${_MODE_LIST[@]}; do

    for GPUS in ${_7B_TP_LIST[@]}; do

        SERVER_PID=`bash $PERF_BASE_PATH/benchmark_server_templ_cuda.sh 7 $GPUS | grep -o "[0-9]\+"`
        SERVER_PID=${SERVER_PID##*:}

        if [ ! -n "$SERVER_PID" ]; then
            echo "[ERROR] SERVER START FAILED"
            exit 1
        else
            echo "SERVER STARTED[$SERVER_PID]: MODEL7B TP${GPUS}"
        fi

        for NUM_CLIENTS in ${_NUM_CLIENTS_LIST[@]}; do
            unittest 7 1 1000 $NUM_CLIENTS 10 $MODE
        done

        kill -9 $SERVER_PID

    done
done
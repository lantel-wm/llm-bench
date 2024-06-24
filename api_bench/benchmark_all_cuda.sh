SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    CLIENTS=$3
    MODE=$4
    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} CLIENTS$CLIENTS ${MODE^^}]"
    LATENCY=`bash $PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh ${MODEL_SIZE} ${GPUS} ${BATCH} ${INLEN} ${OUTLEN} | grep -oP "Avg latency: \K[0-9]+\.[0-9]+"`

    if [ ! -n "$LATENCY" ]; then
        echo "[FAILED]"
    else
        LATENCY=$(echo "scale=2; $LATENCY * 1000" | bc)
        O_TPS=$(echo "scale=4; $BATCH * $OUTLEN / ($LATENCY / 1000)" | bc)
        IO_TPS=$(echo "scale=4; $BATCH * ($OUTLEN + $INLEN) / ($LATENCY / 1000)" | bc)
        echo "[OK] $LATENCY, $O_TPS, $IO_TPS"
        echo "$MODEL_SIZE,$GPUS,$BATCH,$INLEN,$OUTLEN,$MODE,$LATENCY,$O_TPS,$IO_TPS" >> $PERF_BASE_PATH/benchmark_all_cuda_result.csv        
    fi
}
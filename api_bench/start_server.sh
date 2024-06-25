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


MODEL_SIZE=$1

if [ ! -n "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ ! -n "$TP_SIZE" ]; then
    TP_SIZE=1
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
> server.log 2>&1 &"

# --disable-log-stats \

echo "SERVER STARTED: MODEL${MODEL_SIZE}B TP${TP_SIZE} HOST${HOST} PORT${PORT} -> $CMD"

eval "$CMD"

SERVER_PID=$!

echo "SERVER PID: $SERVER_PID"

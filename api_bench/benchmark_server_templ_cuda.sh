SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

MODEL_SIZE=$1

if [ ! -n "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ ! -n "$TP_SIZE" ]; then
    TP_SIZE=1
fi

HOST=$3

if [ ! -n "$HOST" ]; then
    HOST="localhost"
fi

PORT=$4

if [ ! -n "$PORT" ]; then
    PORT=8000
fi

MODEL_DIR="$PERF_BASE_PATH/../../hf_models/llama-${MODEL_SIZE}b-hf"

# python -m vllm.entrypoints.openai.api_server --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --swap-space 16 --disable-log-requests --enforce-eager --host 10.198.31.25  --port 8000

CMD="python -m vllm.entrypoints.openai.api_server \
--model $MODEL_DIR \
--tensor-parallel-size $TP_SIZE \
--swap-space 16 \
--disable-log-requests \
--enforce-eager \
--host $HOST \
--port $PORT \
--uvicorn-log-level warning \
--disable-log-stats"

echo "SERVER START: MODEL${MODEL_SIZE}B TP${TP_SIZE} HOST${HOST} PORT${PORT} -> $CMD"

eval "$CMD"
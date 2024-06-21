SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

export BENCHMARK_EXTENDED_OPTIONS=

bash $PERF_BASE_PATH/benchmark_templ_cuda.sh $*

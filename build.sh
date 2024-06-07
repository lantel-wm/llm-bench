#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 帮助信息函数
show_help() {
  echo "Usage: $0 <cuda_version>"
  echo ""
  echo "Install the appropriate version of vllm based on the specified CUDA version."
  echo ""
  echo "Arguments:"
  echo "  -h                Show this help message and exit."
  echo "  <cuda_version>    The CUDA version to use for the installation (either 11 or 12)."
  echo ""
  echo "Examples:"
  echo "  $0 11             Install vllm for CUDA 11."
  echo "  $0 12             Install vllm for CUDA 12."
  echo ""
  echo "Script absolute path: $SCRIPT_DIR"
}

# 检查输入参数
if [ $# -ne 1 ]; then
  show_help
  exit 1
fi

# 获取输入参数
PARAM=$1

# 检查是否需要显示帮助信息
if [ "$PARAM" == "-h" ]; then
  show_help
  exit 0
fi

# 警告信息并询问用户确认
echo -e "[WARNING] This script should be run in a newly created conda environment."
read -p "Are you sure to continue? (Y/n): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "[INFO] Installation aborted."
  exit 1
fi

# 安装不同版本的vllm
case $PARAM in
  "11")
    BENCHMARK_LLM="$SCRIPT_DIR/python/benchmark_latency_old_version.py"
    echo "[INFO] Installing vllm==0.4.2 for CUDA 11..."
    # pip install -r requirements_vllm042.txt
    pip install https://github.com/vllm-project/vllm/releases/download/v0.4.2/vllm-0.4.2+cu118-cp39-cp39-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118 
    pip install flash_attn
    ;;
  "12")
    BENCHMARK_LLM="$SCRIPT_DIR/python/benchmark_latency.py"
    echo "[INFO] Installing vllm==0.4.3 for CUDA 12..."
    # pip install -r requirements_vllm043.txt
    pip install vllm==0.4.3
    pip install vllm_flash_attn

    ;;
  *)
    echo "[ERROR] Unsupported CUDA version: $PARAM. Only 11 and 12 are supported."
    exit 1
    ;;
esac

echo "[INFO] vllm installed for CUDA $PARAM"

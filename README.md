# llm-benchmark

vLLM，PPL等LLM推理框架的静态和动态性能测试。

## 前置准备

1.conda环境：

新建一个conda环境，使用python3.9版本，不要用已有的conda环境进行测试。

```shell
conda create -n llm_bench python=3.9 -y
conda activate llm_bench
```

2.安装依赖：

根据本机CUDA版本进行安装，支持CUDA11和CUDA12，推荐使用CUDA12。

```shell
## CUDA 12.x
bash ./build.sh 12

## CUDA 11.x
bash ./build.sh 11
```

3.权重文件

vllm测试需要hugging face格式的权重，ppl测试需要pmx格式的权重。

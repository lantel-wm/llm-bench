# vllm-benchmark

提供vllm static benchmark 和 api benchmark.

## 前置准备

1.conda环境：

新建一个conda环境，使用python3.9版本，不要用已有的conda环境进行测试。

```shell
conda create -n vllm_static_bench python=3.9 -y
conda activate vllm_static_bench
```

2.安装依赖：

根据本机CUDA版本进行安装，支持CUDA11和CUDA12，推荐使用CUDA12。

```shell
## CUDA 12.x
bash ./build.sh 12

## CUDA 11.x
bash ./build.sh 11
```

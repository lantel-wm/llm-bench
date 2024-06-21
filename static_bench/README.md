# static-benchmark

## 测试启动

运行脚本之前，先设置`BENCHMARK_LLM`环境变量：

```shell
# CUDA 12.x, vllm==0.4.3
export BENCHMARK_LLM="$(dirname "$(realpath ".")")/python/benchmark_latency.py"

# CUDA 11.x, vllm<=0.4.2
export BENCHMARK_LLM="$(dirname "$(realpath ".")")/python/benchmark_latency_old_version.py"
```

启动`benchmark_all_cuda.sh`即可开始测试：

```shell
bash benchmark_all_cuda.sh
```
启动`benmark_one_cuda_fp16.sh`可以进行单个case的测试并查看详情log:

```shell
bash benmark_one_cuda_fp16.sh
```

## 调整测试算例

`benchmark_all_cuda.sh`中可以调整测试的batch size，model size，tp size，in len，out len，mode等参数。

## 结果保存

测试结果保存在benchmark_all_cuda.sh，如果确认没问题，可以重命名成benchmark_result_{显卡型号}_{yyyymmdd}.csv，并移动到result文件夹，
例如benchmark_result_a800_20240501.csv


# api-benchmark

## 前置准备

下载 sharegpt 数据集：

```shell
cd api_bench
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

设置环境变量：
```shell
export VLLM_SERVER_URL=http://10.198.31.25:8000
export BENCHMARK_LLM=${REPO_PATH}/api_bench/python/benchmark_serving.py
export DATASET_PATH=${REPO_PATH}/api_bench/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 测试启动

```shell
bash benchmark_all_cuda.sh
```

## 使用Python测试单个case

启动 server：
```shell
python -m vllm.entrypoints.openai.api_server --model PATH_TO_MODEL --swap-space 16 --disable-log-requests --enforce-eager --host YOUR_IP_ADDRESS  --port 8000
```

启动 client，开始benchmark：
```shell
python python/benchmark_serving.py --host YOUR_IP_ADDRESS --port 8000 --model PATH_TO_MODEL --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompt 200 --thread-num 48 --disable-tqdm --thread-stop-time 300
```

## 参数设置

- `--thread-num`：并发数，为1则执行动态并发，大于1则控制并发数固定。
- `--num-prompt`：每个并发的测试的prompt数量。总prompt数量需要乘以并发数。
- `--thread-stop-time`：每个并发的测试的持续时间，单位秒。

## TODO

<!-- # 留三位小数
# 数据集的平均input output长度
# min_prompt_len, max_prompt_len, max_total_len, min_prompt_len
# input 长度，total len 的柱状图
# sample sharegpt 指定任意轮对话 -->

- [x] 所有结果保留三位小数
- [x] 去除不必要的参数和代码
- [ ] 输出数据集的平均input_len output_len
- [ ] 添加 min_prompt_len, max_prompt_len, min_prompt_len, max_total_len参数，控制request的sample过程
- [ ] 绘制input_len，total_len的柱状图
- [ ] 添加chat_turns参数，实现sample sharegpt 指定任意轮对话的功能
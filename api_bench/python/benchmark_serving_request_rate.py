"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_requests> # By default <num_requests> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import logging
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func_async import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@dataclass
class BenchmarkMetrics:
    completed: int
    successful_rate: float
    total_input: int
    total_output: int
    mean_input_tokens: float
    mean_output_tokens: float
    max_input_tokens: int
    max_output_tokens: int
    request_throughput: float
    in_out_throughput: float
    output_throughput: float
    
    min_ttft_ms: float
    max_ttft_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    
    min_tpot_ms: float
    max_tpot_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    
    min_e2e_ms: float
    max_e2e_ms: float
    mean_e2e_ms: float
    median_e2e_ms: float
    std_e2e_ms: float
    p90_e2e_ms: float
    p99_e2e_ms: float
    
    min_itl_ms: float
    max_itl_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    system_prompt_path: Optional[str] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        
        if system_prompt_path is not None:
            with open(system_prompt_path) as f:
                prompt = f.read() + '\n' + prompt
        
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
        
        if i == len(dataset) - 1:
            # If we have reached the end of the dataset, then we need to shuffle
            # the dataset again.
            random.shuffle(filtered_dataset)
            i = 0
            
    return filtered_dataset


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request_id, request in enumerate(input_requests):
        yield request_id, request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input_tokens = 0
    max_input_tokens = 0
    max_output_tokens = 0
    completed = 0
    itls: List[float] = []
    e2es: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2es: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            actual_output_lens.append(output_len)
            total_input_tokens += input_requests[i][1]
            max_input_tokens = max(max_input_tokens, input_requests[i][1])
            max_output_tokens = max(max_output_tokens, output_len)
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2es.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)
            
    total_output_tokens = sum(actual_output_lens)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        successful_rate=completed / len(outputs),
        total_input=total_input_tokens,
        total_output=total_output_tokens,
        mean_input_tokens=total_input_tokens / completed,
        mean_output_tokens=total_output_tokens / completed,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        request_throughput=completed / dur_s,
        in_out_throughput=(total_input_tokens + total_output_tokens) / dur_s,
        output_throughput=total_output_tokens / dur_s,
        
        min_ttft_ms=np.min(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        max_ttft_ms=np.max(ttfts or 0) * 1000,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        
        min_tpot_ms=np.min(tpots or 0) * 1000,
        max_tpot_ms=np.max(tpots or 0) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        
        min_e2e_ms=np.min(e2es or 0) * 1000,
        max_e2e_ms=np.max(e2es or 0) * 1000,
        mean_e2e_ms=np.mean(e2es or 0) * 1000,
        median_e2e_ms=np.median(e2es or 0) * 1000,
        std_e2e_ms=np.std(e2es or 0) * 1000,
        p90_e2e_ms=np.percentile(e2es or 0, 90) * 1000,
        p99_e2e_ms=np.percentile(e2es or 0, 99) * 1000,
        
        min_itl_ms=np.min(itls or 0) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p90_itl_ms=np.percentile(itls or 0, 90) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


def dump_metrics_and_results(metrics: BenchmarkMetrics):
    print("CSV header output:\
success_rate,qps,avg_inlen,avg_outlen,max_inlen,max_outlen,o_tps,io_tps,\
min_ttft,max_ttft,mean_ttft,median_ttft,std_ttft,p90_ttft,p99_ttft,\
min_tpot,max_tpot,mean_tpot,median_tpot,std_tpot,p90_tpot,p99_tpot,\
min_e2e,max_e2e,mean_e2e,median_e2e,std_e2e,p90_e2e,p99_e2e,\
min_itl,max_itl,mean_itl,median_itl,std_itl,p90_itl,p99_itl")
          
    csv_line = ""
    csv_line += f"{metrics.successful_rate:.3f},"
    csv_line += f"{metrics.request_throughput:.3f},"
    csv_line += f"{metrics.mean_input_tokens:.3f},"
    csv_line += f"{metrics.mean_output_tokens:.3f},"
    csv_line += f"{metrics.max_input_tokens},"
    csv_line += f"{metrics.max_output_tokens},"
    csv_line += f"{metrics.output_throughput:.3f},"
    csv_line += f"{metrics.in_out_throughput:.3f},"
    
    csv_line += f"{metrics.min_ttft_ms:.3f},"
    csv_line += f"{metrics.max_ttft_ms:.3f},"
    csv_line += f"{metrics.mean_ttft_ms:.3f},"
    csv_line += f"{metrics.median_ttft_ms:.3f},"
    csv_line += f"{metrics.std_ttft_ms:.3f},"
    csv_line += f"{metrics.p90_ttft_ms:.3f},"
    csv_line += f"{metrics.p99_ttft_ms:.3f},"
    
    csv_line += f"{metrics.min_tpot_ms:.3f},"
    csv_line += f"{metrics.max_tpot_ms:.3f},"
    csv_line += f"{metrics.mean_tpot_ms:.3f},"
    csv_line += f"{metrics.median_tpot_ms:.3f},"
    csv_line += f"{metrics.std_tpot_ms:.3f},"
    csv_line += f"{metrics.p90_tpot_ms:.3f},"
    csv_line += f"{metrics.p99_tpot_ms:.3f},"
    
    csv_line += f"{metrics.min_e2e_ms:.3f},"
    csv_line += f"{metrics.max_e2e_ms:.3f},"
    csv_line += f"{metrics.mean_e2e_ms:.3f},"
    csv_line += f"{metrics.median_e2e_ms:.3f},"
    csv_line += f"{metrics.std_e2e_ms:.3f},"
    csv_line += f"{metrics.p90_e2e_ms:.3f},"
    csv_line += f"{metrics.p99_e2e_ms:.3f},"
    
    csv_line += f"{metrics.min_itl_ms:.3f},"
    csv_line += f"{metrics.max_itl_ms:.3f},"
    csv_line += f"{metrics.mean_itl_ms:.3f},"
    csv_line += f"{metrics.median_itl_ms:.3f},"
    csv_line += f"{metrics.std_itl_ms:.3f},"
    csv_line += f"{metrics.p90_itl_ms:.3f},"
    csv_line += f"{metrics.p99_itl_ms:.3f}"
    
    print(f"CSV format output:{csv_line}")


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request_id, request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            request_id=request_id,
        )
        logging.debug(f"request_id: {request_id}, prompt_len: {prompt_len}, output_len: {output_len}")
        tasks.append(
            asyncio.create_task(request_func(request_func_input=request_func_input)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
    )
    
    dump_metrics_and_results(metrics)


def main(args: argparse.Namespace):
    # unset http proxy
    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["https_proxy"] = ""
    # print(args)
    random.seed(0)
    np.random.seed(0)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    input_requests = sample_sharegpt_requests(
        dataset_path=args.dataset_path,
        num_requests=args.num_requests,
        tokenizer=tokenizer,
        fixed_output_len=args.sharegpt_output_len,
    )
        
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of requests to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Path to the system prompt file. None for no system prompt.",
    )

    args = parser.parse_args()
    main(args)
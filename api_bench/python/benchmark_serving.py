"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model> --swap-space 16 \
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
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
"""
import argparse
import asyncio
import threading
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, REQUEST_FUNCS,
                                  RequestFuncInput, RequestFuncOutput)
from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class BenchmarkMetrics:
    completed: int
    successful_rate: float
    total_input: int
    total_output: int
    request_throughput: float
    in_out_throughput: float
    output_throughput: float
    
    min_ttft_ms: float
    max_ttft_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    
    min_tpot_ms: float
    max_tpot_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    
    min_e2e_ms: float
    max_e2e_ms: float
    mean_e2e_ms: float
    median_e2e_ms: float
    p90_e2e_ms: float
    p99_e2e_ms: float


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    # print("[I] Sampling requests...")
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

    return filtered_dataset


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


async def get_request_async(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)
        

def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
):
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        time.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input_tokens = 0
    completed = 0
    tpots = []
    ttfts = []
    e2es = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer(outputs[i].generated_text).input_ids)
            actual_output_lens.append(output_len)
            total_input_tokens += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
                
            ttfts.append(outputs[i].ttft)
            e2es.append(outputs[i].latency)
            completed += 1
            
        else:
            actual_output_lens.append(0)

    total_output_tokens = sum(actual_output_lens)
    
    metrics = BenchmarkMetrics(
        completed=completed,
        successful_rate=completed / len(outputs),
        total_input=total_input_tokens,
        total_output=total_output_tokens,
        request_throughput=completed / dur_s,
        in_out_throughput=(total_input_tokens + total_output_tokens) / dur_s,
        output_throughput=total_output_tokens / dur_s,
        
        min_ttft_ms=np.min(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        max_ttft_ms=np.max(ttfts or 0) * 1000,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        
        min_tpot_ms=np.min(tpots) * 1000,
        max_tpot_ms=np.max(tpots) * 1000,
        mean_tpot_ms=np.mean(tpots) * 1000,
        median_tpot_ms=np.median(tpots) * 1000,
        p90_tpot_ms=np.percentile(tpots, 90) * 1000,
        p99_tpot_ms=np.percentile(tpots, 99) * 1000,
        
        min_e2e_ms=np.min(e2es) * 1000,
        max_e2e_ms=np.max(e2es) * 1000,
        mean_e2e_ms=np.mean(e2es) * 1000,
        median_e2e_ms=np.median(e2es) * 1000,
        p90_e2e_ms=np.percentile(e2es, 90) * 1000,
        p99_e2e_ms=np.percentile(e2es, 99) * 1000,
    )

    return metrics, actual_output_lens


def dump_metrics_and_results(
    metrics: BenchmarkMetrics, 
    actual_output_lens: List[int],
    outputs: List[RequestFuncOutput], 
    benchmark_duration: float
) -> dict:
    # success_rate, qps, o_tps, io_tps, min_ttft, max_ttft, mean_ttft, median_ttft, p90_ttft, p99_ttft, min_tpot, max_tpot, mean_tpot, median_tpot, p90_tpot, p99_tpot, min_tpr, max_tpr, mean_tpr, median_tpr, p90_tpr, p99_tpr
    csv_line = ""
    csv_line += f"{metrics.successful_rate:.3f},"
    csv_line += f"{metrics.request_throughput:.3f},"
    csv_line += f"{metrics.output_throughput:.3f},"
    csv_line += f"{metrics.in_out_throughput:.3f},"
    csv_line += f"{metrics.min_ttft_ms:.3f},"
    csv_line += f"{metrics.max_ttft_ms:.3f},"
    csv_line += f"{metrics.mean_ttft_ms:.3f},"
    csv_line += f"{metrics.median_ttft_ms:.3f},"
    csv_line += f"{metrics.p90_ttft_ms:.3f},"
    csv_line += f"{metrics.p99_ttft_ms:.3f},"
    csv_line += f"{metrics.min_tpot_ms:.3f},"
    csv_line += f"{metrics.max_tpot_ms:.3f},"
    csv_line += f"{metrics.mean_tpot_ms:.3f},"
    csv_line += f"{metrics.median_tpot_ms:.3f},"
    csv_line += f"{metrics.p90_tpot_ms:.3f},"
    csv_line += f"{metrics.p99_tpot_ms:.3f},"
    csv_line += f"{metrics.min_e2e_ms:.3f},"
    csv_line += f"{metrics.max_e2e_ms:.3f},"
    csv_line += f"{metrics.mean_e2e_ms:.3f},"
    csv_line += f"{metrics.median_e2e_ms:.3f},"
    csv_line += f"{metrics.p90_e2e_ms:.3f},"
    csv_line += f"{metrics.p99_e2e_ms:.3f}"
    print(f"CSV format output:{csv_line}")

    return {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "in_out_throughput": metrics.in_out_throughput,
        "output_throughput": metrics.output_throughput,
        "min_ttft_ms": metrics.min_ttft_ms,
        "max_ttft_ms": metrics.max_ttft_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "min_tpot_ms": metrics.min_tpot_ms,
        "max_tpot_ms": metrics.max_tpot_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "min_e2e_ms": metrics.min_e2e_ms,
        "max_e2e_ms": metrics.max_e2e_ms,
        "mean_e2e_ms": metrics.mean_e2e_ms,
        "median_e2e_ms": metrics.median_e2e_ms,
        "p90_e2e_ms": metrics.p90_e2e_ms,
        "p99_e2e_ms": metrics.p99_e2e_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

async def benchmark_async(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request_async(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start_time
    
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    return dump_metrics_and_results(metrics, actual_output_lens, outputs, benchmark_duration)

def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    thread_id: int = -1,
):
    if backend in REQUEST_FUNCS:
        request_func = REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")


    benchmark_start_time = time.perf_counter()
    outputs = []
    for request in get_request(input_requests, request_rate):
        if args.thread_stop_time > 0 and time.perf_counter() - benchmark_start_time >= args.thread_stop_time:
            break
        
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
                
        outputs.append(request_func(request_func_input=request_func_input))
        

    if thread_id == -1:
        benchmark_duration = time.perf_counter() - benchmark_start_time
        
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
        )

        return dump_metrics_and_results(metrics, actual_output_lens, outputs, benchmark_duration)
    
    else:
        return outputs


class benchThread(threading.Thread):
    def __init__(self, thread_id, ramp_up_time, backend, api_url, model_id, tokenizer, input_requests,
                 best_of, use_beam_search, request_rate):
        super(benchThread, self).__init__()
        self.thread_id = thread_id
        self.ramp_up_time = ramp_up_time
        self.backend = backend
        self.api_url = api_url
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.input_requests = input_requests
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.request_rate = request_rate
        
    def run(self):
        time.sleep(self.ramp_up_time)
        self.outputs = benchmark(
                backend=self.backend,
                api_url=self.api_url,
                model_id=self.model_id,
                tokenizer=self.tokenizer,
                input_requests=self.input_requests,
                best_of=self.best_of,
                use_beam_search=self.use_beam_search,
                request_rate=self.request_rate,
                thread_id=self.thread_id,
            )
        
    def get_result(self):
        self.join()
        return self.outputs


def main(args: argparse.Namespace):
    print(args)
    assert args.num_threads > 0, "Number of threads must be greater than 0."
    
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    # sample requests
    input_request = [sample_sharegpt_requests(
        dataset_path=args.dataset_path,
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        fixed_output_len=args.sharegpt_output_len,
    ) for _ in range(args.num_threads)]
    input_requests = sum(input_request, [])                

    # start benchmark
    benchmark_start_time = time.perf_counter()
    threads = []
    for i in range(args.num_threads):
        thread = benchThread(i, i * args.ramp_up_time / args.num_threads, backend, api_url, model_id, tokenizer, input_requests[i * args.num_prompts:(i + 1) * args.num_prompts],
                                args.best_of, args.use_beam_search, args.request_rate)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    benchmark_duration = time.perf_counter() - benchmark_start_time

    # gather benchmark result
    all_outputs = []
    for thread in threads:
        outputs = thread.get_result()
        print(f"len(outputs): {len(outputs)}, thread_id: {thread.thread_id}")
        all_outputs += outputs
        
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=all_outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    benchmark_result = dump_metrics_and_results(metrics, actual_output_lens, all_outputs, benchmark_duration)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
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
        required=True,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet"],
        help="Name of the dataset to benchmark on.",
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
        "Name or path of the tokenizer, if not using the default tokenizer.",
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
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
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
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use for the benchmark.",
    )
    parser.add_argument(
        "--ramp-up-time",
        type=float,
        default=0,
        help="Ramp up time in seconds for each thread.",
    )
    parser.add_argument(
        "--thread-stop-time",
        type=float,
        default=0,
        help="Stop time in seconds for each thread.",
    )

    args = parser.parse_args()
    main(args)
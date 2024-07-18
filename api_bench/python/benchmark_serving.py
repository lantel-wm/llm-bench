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
        --dataset-path <path to dataset> \
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
"""
import argparse
import threading
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
from backend_request_func import (REQUEST_FUNCS, RequestFuncInput, RequestFuncOutput)
from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class BenchmarkMetrics:
    completed: int
    successful_rate: float
    total_input: int
    total_output: int
    mean_input_tokens: float
    mean_output_tokens: float
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
    num_turns: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    # print("[I] Sampling requests...")
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    num_turns *= 2 # Each turn has a prompt and a completion.
    # Filter out the conversations with less than num_turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= num_turns]
    # Only keep the first num_turns of each conversation.
    dataset = [[data["conversations"][turn]["value"] for turn in range(num_turns)] for data in dataset]


    # Shuffle the dataset.
    random.seed(0)
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break
        
        prompt = ""
        for j in range(num_turns - 1):
            prompt += dataset[i][j] + "\n"
        completion = dataset[i][-1]
        
        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        
        filtered_dataset.append((prompt, prompt_len, output_len))
        
        if i == len(dataset) - 1:
            i = 0

    return filtered_dataset


def get_request(
    input_requests: List[Tuple[str, int, int]],
):
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request


def calculate_metrics(
    input_requests_list: List[List[Tuple[str, int, int]]],
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
            thread_id = outputs[i].thread_id
            request_id = outputs[i].request_id
            input_request = input_requests_list[thread_id][request_id]
            # print(f"thread_id: {thread_id}, request_id: {request_id}")
            total_input_tokens += input_request[1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
                
            ttfts.append(outputs[i].ttft)
            e2es.append(outputs[i].latency)
            completed += 1
            
        else:
            # print(f"outputs[{i}].error: {outputs[i].error}")
            actual_output_lens.append(0)

    total_output_tokens = sum(actual_output_lens)
    
    metrics = BenchmarkMetrics(
        completed=completed,
        successful_rate=completed / len(outputs),
        total_input=total_input_tokens,
        total_output=total_output_tokens,
        mean_input_tokens=total_input_tokens / completed,
        mean_output_tokens=total_output_tokens / completed,
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
):
    # success_rate, qps, avg_inlen, avg_outlen, o_tps, io_tps, min_ttft, max_ttft, mean_ttft, median_ttft, p90_ttft, p99_ttft, min_tpot, max_tpot, mean_tpot, median_tpot, p90_tpot, p99_tpot, min_tpr, max_tpr, mean_tpr, median_tpr, p90_tpr, p99_tpr
    csv_line = ""
    csv_line += f"{metrics.successful_rate:.3f},"
    csv_line += f"{metrics.request_throughput:.3f},"
    csv_line += f"{metrics.mean_input_tokens:.3f},"
    csv_line += f"{metrics.mean_output_tokens:.3f},"
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
    

def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    thread_id: int = -1,
    num_requests: int = -1,
):
    if backend in REQUEST_FUNCS:
        request_func = REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")


    benchmark_start_time = time.perf_counter()
    outputs = []
    for request_id, request in enumerate(get_request(input_requests)):
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
            thread_id=thread_id,
            request_id=request_id,
            num_requests=num_requests,
        )
        outputs.append(request_func(request_func_input=request_func_input))
        
    return outputs


class benchThread(threading.Thread):
    def __init__(self, thread_id, ramp_up_time, backend, api_url, model_id, tokenizer, input_requests,
                 best_of, use_beam_search, num_requests):
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
        self.num_requests = num_requests
        
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
                thread_id=self.thread_id,
                num_requests=self.num_requests,
            )
        
    def get_result(self):
        self.join()
        return self.outputs

def roll(lst: list, n: int):
    n = n % len(lst)
    return lst[n:] + lst[:n]
    

def main(args: argparse.Namespace):
    # print(args)
    logging.debug(args)
    assert args.num_requests > 0, "Number of threads must be greater than 0."
    
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if backend in ["vllm", "openai"]:
        api_url = f"{args.base_url}{args.endpoint}"
        if not api_url.startswith("http"):
            api_url = f"http://{api_url}"
        logging.debug(f"using vllm backend with api url: {api_url}")
    elif backend in ["ppl"]:
        api_url = args.base_url
        logging.debug(f"using ppl backend with api url: {api_url}")
    
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    # sample requests
    input_requests = sample_sharegpt_requests(
        dataset_path=args.dataset_path,
        num_requests=args.num_requests,
        num_turns=args.num_turns,
        tokenizer=tokenizer,
        fixed_output_len=args.sharegpt_output_len,
    ) 
    
    # start benchmark
    benchmark_start_time = time.perf_counter()
    threads = []
    input_requests_list = []
    for thread_id in range(args.num_requests):
        input_requests_i = input_requests[thread_id:] + input_requests[:thread_id]
        if thread_id % 2 == 1:
            input_requests_i = input_requests_i[::-1]
        input_requests_list.append(input_requests_i)
        thread = benchThread(thread_id, thread_id * args.ramp_up_time / args.num_requests, backend, api_url, model_id, tokenizer, input_requests_i,
                                args.best_of, args.use_beam_search, args.num_requests)
        thread.start()
        threads.append(thread)
        logging.debug(f"started thread {thread_id} with ramp up time {thread_id * args.ramp_up_time / args.num_requests}")

    for thread in threads:
        thread.join()
        
    benchmark_duration = time.perf_counter() - benchmark_start_time

    # gather benchmark result
    all_outputs = []
    for thread in threads:
        outputs = thread.get_result()
        all_outputs += outputs
        
    metrics, actual_output_lens = calculate_metrics(
        input_requests_list=input_requests_list,
        outputs=all_outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    dump_metrics_and_results(metrics, actual_output_lens, all_outputs, benchmark_duration)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(REQUEST_FUNCS.keys()),
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
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use for the benchmark.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=1,
        help="Number of chat turns to use for the benchmark. A prompt and a completion are considered as one turn.",
    )
    parser.add_argument(
        "--ramp-up-time",
        type=float,
        default=1,
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

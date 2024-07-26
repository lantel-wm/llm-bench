import json
import os
import sys
import time
import traceback
import logging
from grpc import aio
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
import huggingface_hub.constants
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from ppl_server_utils import llm_pb2, llm_pb2_grpc

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    request_id: int = 0


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""
    thread_id: Optional[int] = None
    request_id: int = 0
    

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        output_len = 0
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                output.itl.append(timestamp -
                                                  most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                                output_len += 1

                    output.generated_text = generated_text
                    output.output_len = output_len
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output

async def async_request_ppl_completions(request_func_input: RequestFuncInput) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    channel = aio.insecure_channel(api_url)
    stub = llm_pb2_grpc.LLMServiceStub(channel)
    
    thread_id = request_func_input.thread_id
    request_id = request_func_input.request_id
    num_requests = request_func_input.num_requests
    
    request = llm_pb2.Request(
        id=thread_id * num_requests + request_id,
        prompt=request_func_input.prompt,
        temperature=0.0,
        stopping_parameters=llm_pb2.StoppingCriteriaParameters(
            max_new_tokens=request_func_input.output_len,
            ignore_eos_token=True
        )
    )
    batched_request = llm_pb2.BatchedRequest(req=[request])
    
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id, 
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len
    )
    
    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    
    try:
        async for response in stub.Generation(batched_request):
            for rsp in response.rsp:
                if rsp.status == llm_pb2.Status.FAILED:
                    logging.warning(f"Request {request.id} failed")
                    output.success = False
                    output.error = "Response Status: FAILED"
                    break
                
                else:
                    if rsp.generated:
                        timestamp = time.perf_counter()
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)
                        
                        most_recent_timestamp = timestamp
                        generated_text += rsp.generated
                        output_len += 1
                        
                    if rsp.status == llm_pb2.Status.FINISHED:
                        logging.info(f"Request {request.id} finished")
                        latency = time.perf_counter() - st
                        output.success = True
                        break
        
        output.generated_text = generated_text
        output.output_len = output_len
        output.latency = time.perf_counter() - st
                       
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
    
    await channel.close()
    return output



# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


# def get_model(pretrained_model_name_or_path: str) -> str:
#     if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
#         from modelscope import snapshot_download

#         model_path = snapshot_download(
#             model_id=pretrained_model_name_or_path,
#             local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
#             ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

#         return model_path
#     return pretrained_model_name_or_path


# def get_tokenizer(
#     pretrained_model_name_or_path: str, trust_remote_code: bool
# ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
#     if pretrained_model_name_or_path is not None and not os.path.exists(
#             pretrained_model_name_or_path):
#         pretrained_model_name_or_path = get_model(
#             pretrained_model_name_or_path)
#     return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
#                                          trust_remote_code=trust_remote_code)


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
}
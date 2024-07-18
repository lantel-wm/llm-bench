import json
import os
import sys
import time
import grpc
import traceback
import requests
from dataclasses import dataclass, field
from typing import List, Optional

from ppl_server_utils import llm_pb2, llm_pb2_grpc

HTTP_TIMEOUT = 6 * 60 * 60

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    thread_id: Optional[int] = None
    request_id: int = 0
    num_requests: int = 1


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    thread_id: Optional[int] = None
    request_id: int = 0

# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/completions"
    ), "OpenAI Completions API URL must end with 'v1/completions'."

    assert not request_func_input.use_beam_search
    payload = {
        "model": request_func_input.model,
        "prompt": request_func_input.prompt,
        "temperature": 0.0,
        "best_of": request_func_input.best_of,
        "max_tokens": request_func_input.output_len,
        "min_tokens": request_func_input.output_len,
        "stream": True,
        "ignore_eos": True,
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }

    output = RequestFuncOutput(thread_id=request_func_input.thread_id, request_id=request_func_input.request_id)
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(url=api_url, json=payload, 
                           headers=headers, stream=True,
                           timeout=HTTP_TIMEOUT) as response:
            if response.status_code == 200:
                for chunk in response.iter_lines():
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                    
                    if chunk == "[DONE]":
                        latency = time.perf_counter() - st
                    else:
                        data = json.loads(chunk)
                        
                        if data["choices"][0]["text"]:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                                
                            # Decoding phase
                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # do not want to include as inter-token-latency
                            elif data.get("usage", None) is None:
                                output.itl.append(timestamp -
                                                    most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += data["choices"][0]["text"]

                output.generated_text = generated_text
                output.success = True
                output.latency = latency
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.text: {response.text}"

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output

def request_ppl_completions(request_func_input: RequestFuncInput) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    channel = grpc.insecure_channel(api_url)
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
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    
    # try:
    response_stream = stub.Generation(batched_request)
    for response in response_stream:
        for rsp in response.rsp:
            if rsp.status == llm_pb2.Status.FINISHED:
                latency = time.perf_counter() - st
                output.success = True
                break
            elif rsp.status == llm_pb2.Status.FAILED:
                output.success = False
                output.error = "Response Status: FAILED"
                break
            elif rsp.generated:
                timestamp = time.perf_counter()
                if ttft == 0.0:
                    ttft = time.perf_counter() - st
                    output.ttft = ttft
                else:
                    output.itl.append(timestamp - most_recent_timestamp)
                
                most_recent_timestamp = timestamp
                generated_text += rsp.generated
    
    output.generated_text = generated_text
    output.latency = latency
                       
    # except Exception:
    #     output.success = False
    #     exc_info = sys.exc_info()
    #     output.error = "".join(traceback.format_exception(*exc_info))
        
    return output
                    

REQUEST_FUNCS = {
    "vllm": request_openai_completions,
    "openai": request_openai_completions,
    "ppl": request_ppl_completions,
}

if __name__ == '__main__':
    request_func_input = RequestFuncInput(
        prompt="The future of AI is here.\nThe future of AI is here. The technology is already being used in a variety of industries, from healthcare to manufacturing.\nThe future of AI is here. The technology is already being used in a variety of industries, from healthcare to manufacturing. But what does",
        api_url="127.0.0.1:23333",
        prompt_len=1,
        output_len=64,
        model="llama",
        thread_id=0,
        request_id=0,
        num_requests=1024,
    )
    
    output = request_ppl_completions(request_func_input)
    
    print(output)

import os
import grpc
import logging
import argparse
import requests

from ppl_server_utils import llm_pb2, llm_pb2_grpc

def check_server_status(server_url: str, backend: str, model: str = "") -> str:
    if backend in "ppl":
        channel = grpc.insecure_channel(server_url)
        stub = llm_pb2_grpc.LLMServiceStub(channel)
        request = llm_pb2.Request(
            id=2**64 - 1, # uint64_max
            prompt="Check server status",
            temperature=0.0,
            stopping_parameters=llm_pb2.StoppingCriteriaParameters(
                max_new_tokens=1,
                ignore_eos_token=True
            )
        )
        batched_request = llm_pb2.BatchedRequest(req=[request])
        try:
            response_stream = stub.Generation(batched_request)
            for response in response_stream:
                for rsp in response.rsp:
                    if rsp.status == llm_pb2.Status.FAILED:
                        return "NOT READY"
                    else:
                        return "OK"
        except Exception:
            return "NOT READY"
        
    elif backend in "vllm":
        assert model != "", "Model name is required for vllm backend"
        if not server_url.startswith("http://"):
            server_url = "http://" + server_url
        if not server_url.endswith("/v1/completions"):
            server_url = server_url + "/v1/completions"
        payload = {
            "model": model,
            "prompt": "Check server status",
            "temperature": 0.0,
            "max_tokens": 1,
            "min_tokens": 1,
            "stream": True,
            "ignore_eos": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        try:
            with requests.post(url=server_url, json=payload, 
                            headers=headers, stream=True,
                            timeout=6 * 60 * 60) as response:
                if response.status_code == 200:
                    for chunk in response.iter_lines():
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        chunk = chunk.decode("utf-8")
                else:
                    return "NOT READY"

        except Exception:
            return "NOT READY"
        
        return "OK"

    
    
if __name__ == "__main__":
    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["https_proxy"] = ""
    
    parser = argparse.ArgumentParser(description='Check server status')
    parser.add_argument('--server-url', type=str, required=True, help='Server URL')
    parser.add_argument('--backend', type=str, required=True, help='Backend type')
    parser.add_argument('--model', type=str, required=False, help='Model name')
    args = parser.parse_args()
    
    print(check_server_status(args.server_url, args.backend, args.model))
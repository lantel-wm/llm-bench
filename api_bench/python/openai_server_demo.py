from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://10.198.31.25:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# print("Models:", models)

completion = client.completions.create(model="/mnt/llm2/llm_perf/hf_models/llama-7b-hf",
                                      prompt="San Francisco is a")
print("Completion result:", completion)
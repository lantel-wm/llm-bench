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

prompt = """
Route optimization and planning is a crucial aspect of transportation management in the supply chain. It involves the process of selecting the most efficient routes for transportation of goods from one place to another. This is done with the aim of reducing transportation costs, increasing delivery times, improving delivery accuracy, and reducing environmental impact. The process of route optimization and planning requires a deep understanding of the transportation network, including the transportation modes, capacities, and time frames involved in each delivery.

Route optimization and planning is an iterative process that involves considering multiple factors and making trade-offs to find the best solution. Some of the key factors considered in route optimization and planning include:

1. Transportation modes: The selection of the most appropriate transportation mode is a crucial aspect of route optimization and planning. Different transportation modes have different costs, capacities, and delivery times, and the best mode will depend on the type of goods being transported and the location of the delivery.
2. Delivery schedules: The delivery schedules of goods must be carefully considered when optimizing routes. This includes considering the lead times involved in each delivery and ensuring that goods arrive at the destination on time.
3. Delivery costs: Route optimization and planning should aim to minimize the overall delivery costs, including the cost of transportation, packaging, and any other costs involved in the delivery process.
4. Environmental impact: Route optimization and planning should aim to minimize the environmental impact of transportation. This can be achieved by selecting the most efficient transportation modes and routes, and by reducing the carbon footprint of transportation.

Route optimization and planning can be performed manually, but it is often more efficient to use a specialized software tool. These tools can automate many of the calculations involved in route optimization and planning and can help to ensure that the best solution is found.

In conclusion, route optimization and planning is a critical aspect of transportation management in the supply chain. It involves the process of selecting the most efficient routes for transportation of goods and is aimed at reducing transportation costs, increasing delivery times, improving delivery accuracy, and reducing environmental impact. Effective route optimization and planning requires a deep understanding of the transportation network and the key factors involved in each delivery. The use of specialized software tools can help to automate the process and ensure that the best solution is found.
"""

completion = client.completions.create(model="/mnt/llm2/llm_perf/hf_models/llama-7b-hf",
                                      prompt=prompt)
print("Completion result:", completion.choices[0].text)

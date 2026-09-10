from dotenv import load_dotenv

load_dotenv()

# text generation using the huggingface models with API
"""
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash'
)

model = ChatHuggingFace(llm = llm)

response = model.invoke("Who are you ?")

print(response.text)
"""

# text geberation with huggingface opensource models using 
from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)

model = ChatHuggingFace(llm = llm)

response = model.invoke("Who are you ?")

print(response.content)
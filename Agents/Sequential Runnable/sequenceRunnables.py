from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StuOutputParser

# prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# llm model
model = ChatMistralAI(model = "mistral-small-2506")

# 3. Output Parser
parser = StuOutputParser()

chains = prompt | model | parser

result = chains.invoke("Machine Learning")

print(result)

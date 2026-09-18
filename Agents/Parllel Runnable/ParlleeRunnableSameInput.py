from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StuOutputParser

# prompt template
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# detailed_prompt
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# llm model
model = ChatMistralAI(model = "mistral-small-2506")

# 3. Output Parser
parser = StuOutputParser()

chains = {
    "short" : short_prompt | model | parser,
    "detailed" : detailed_prompt | model | parser
}


result = chains.invoke("Machine Learning")



print(result['short'])
print(result['detailed'])
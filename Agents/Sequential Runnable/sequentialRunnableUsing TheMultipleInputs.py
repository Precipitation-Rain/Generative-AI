from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in short"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Take this explanation and make it detailed:\n{text}"
)

chain = (
    RunnableLambda(lambda x: {"topic": x})
    | short_prompt
    | model
    | parser
    | RunnableLambda(lambda x: {"text": x})
    | detailed_prompt
    | model
    | parser
)

result = chain.invoke("Machine Learning")

print(result)
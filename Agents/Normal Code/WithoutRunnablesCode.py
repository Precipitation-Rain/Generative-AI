from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()


# -------- SHORT --------
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in short"
)

short_chain = short_prompt | model | parser

short_result = short_chain.invoke({
    "topic": "Machine Learning"
})


# -------- DETAILED --------
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

detailed_chain = detailed_prompt | model | parser

detailed_result = detailed_chain.invoke({
    "topic": "Deep Learning"
})


print(short_result)
print(detailed_result)
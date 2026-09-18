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
    "Explain {topic} in detail"
)

summary_prompt = ChatPromptTemplate.from_template(
    "Give a summary of {topic}"
)


chain = (
    # Get first input
    RunnableLambda(lambda x: {
        "topic": x["short"],
        "detailed_topic": x["detailed"],
        "summary_topic": x["summary"]
    })

    # First prompt
    | short_prompt
    | model
    | parser

    # Now the other inputs are gone!
)
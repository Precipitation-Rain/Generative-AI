from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable,RunnableLambda , RunnableParallel

# components
model = ChatGroq(model = 'openai/gpt-oss-20b')
parser = StrOutputParser()

# two diff prompts
short_prompts = ChatPromptTemplate.from_template(
    "Explain {topic} in short explanation"
)

detailed_prompts = ChatPromptTemplate.from_template(
    "Explain {topic} in detailed explanation"
)


# input
chains = RunnableParallel(
{    'short' : RunnableLambda(lambda x : x['short']) |  short_prompts | model | parser,
    'detailed' : RunnableLambda(lambda x : x['detailed']) | detailed_prompts | model | parser}
)

result = chains.invoke(
    {
        "short" : {"topic" : "Machine Learning"},
        "detailed" : {"topic" : "Deep Learning"}
    }
)

# this is used when same inout goes into both chains
# result = chains.invoke(
#     {
#         "topic" : "Machine Learning "
#     }
# )


print(result['short'])
print(result['detailed'])



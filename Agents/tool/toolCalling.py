from rich import print
from langchain.tools import tool 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
load_dotenv()



@tool
def str_length(text : str) -> int:
    """This returns the length of the input string"""
    return len(text)

llm = ChatGroq(model = "openai/gpt-oss-20b")

# Bind the tolls with llm
llm_with_tool = llm.bind_tools([str_length])

result = llm.invoke("This returns the length of the input string : 'hello'")
print(result)

print('\n\n\n\n\n')

result2 = llm_with_tool.invoke("This returns the length of the input string : 'hello'")
print(result2)


from rich import print
from langchain.tools import tool 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain.messages import HumanMessage

# build an custom tool[Tool Building]
@tool
def get_length(text : str) -> int:
    """The tool is designed to return the length of the string"""
    return len(text)

# llm Object 
llm = ChatGroq(model = "openai/gpt-oss-20b")


# Bind the tool with llm[Tool Binding]
llm_with_tool = llm.bind_tools([get_length])

# define function name with actual tool name
tools = {
    "get_length" : get_length
}

message = []
temp = "Rajvardhan"
query = HumanMessage(f"Return the no of characters in the string. The string is {temp}")
message.append(query)

result = llm_with_tool.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

result = llm_with_tool.invoke(message)
print(result.content)








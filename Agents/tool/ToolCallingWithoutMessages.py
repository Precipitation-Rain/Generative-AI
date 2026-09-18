from rich import print
from langchain.tools import tool 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# build an custom tool[Tool Building]
@tool
def get_length(text : str) -> int:
    """The tool is designed to return the length of the string"""
    return len(text)

llm = ChatGroq(model = "openai/gpt-oss-20b")


# Bind the tool with llm[Tool Binding]
llm_with_tool = llm.bind_tools([get_length])
temp = "I am Rajvardhan"
result = llm_with_tool.invoke(f"Use the get_length tool to calculate the length of this exact string: the string is '{temp}'")
# print(result.tool_calls)

# Tool Calling 
if result.tool_calls:
    tool_calls = result.tool_calls[0]
    # print(tool_calls)

tool_name = tool_calls['name']
tool_args = tool_calls['args']

# print(tool_name)
# print(tool_args)

tool_result = get_length.invoke(tool_args)
# print(tool_result)

# returning result back to the llm
final_response = llm_with_tool.invoke(f"The length of text is {tool_result.content}")
print(final_response.content)




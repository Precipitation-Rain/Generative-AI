from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result = 5)
llm = ChatGroq(model = "openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
    """
You are an helpful ai assiatant.
Summarize the following news into clear bullet points
{news}
"""
)

chains = prompt | llm 
search_results = search_tool.run("Lastest updates in Russia ukrain war on 16 th septempber till 4 pm ")

final_result = chains.invoke({"news" : search_results})

print(type(final_result))
print()
print()

print(final_result.content)


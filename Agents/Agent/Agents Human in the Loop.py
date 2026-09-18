# Loading all the modules
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage , ToolMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient
import os , requests
from rich import print

# Create a weather tool
@tool
def get_weather(city : str) -> str:
    """Get current weather of the city"""
    API_KEY=  os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print("Something went wrong")

    description = data['weather'][0]['description']
    temp = data['main']['temp']

    return f"The weather in {city} : {description} , {temp}°C"

# print(get_weather.invoke("Pune"))

# Get news tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city : str) -> str:
    """Get the letest news about city"""

    query = f"Get the top 5 latest news about the {city}"

    response = tavily_client.search(
        query= query,
        search_depth='basic',
        max_results=5,
        topic="news"
    )

    data = response.get("results")

    if not data:
        print("No news found")

    news_list = []
    for news in data:
        title = news.get("title")
        url = news.get("url")
        content = news.get("content")

        news_list.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    # IMPORTANT: outside the for loop
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

# print(get_news.invoke("Pune"))

# LLm bindings
llm = ChatGroq(model = "openai/gpt-oss-20b")

llm_with_tool = llm.bind_tools([get_news , get_weather])

tools = {
    "get_news" : get_news,
    "get_weather" : get_weather
}

# tool execution and calling
messages = []



print("City intelligance system")
print("Enter 0 to exit")

while True:
    # user input
    user_input = input("You : ")

    if user_input == "0":
        break

    messages.append(HumanMessage(content=user_input))

    # system output --> AI Messages
    while True:
        result = llm_with_tool.invoke(messages)

        messages.append(result)

        # if llm choose to call some tools

        if result.tool_calls:

            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                # Human in the Loop
                confirm = input(f"Agent wants to call {tool_name},plz enter (yes / no)")

                if confirm.lower() == 'no':
                    break

                tool_result = tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id = tool_call['id']
                ))

                continue

        else:
            print(result.content)
            break



            


    





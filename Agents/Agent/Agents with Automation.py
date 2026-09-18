# Loading all the modules
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage , ToolMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient
import os , requests
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# Create a weather tool
@tool
def get_weather(city : str) -> str:
    """Get current weather of the city"""
    API_KEY=  os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return f"Could not get weather for {city}. Error: {data.get('message', 'Unknown error')}"

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
@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() == "no":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

# LLm bindings
llm = ChatGroq(model = "openai/gpt-oss-20b")

agent = create_agent(
    model=llm,
    tools=[get_news , get_weather],
    system_prompt="You are an helpfull and unbaised and trythful city assistant",
    middleware=[human_approval]
)

print("City Agent | Enter exit to quit")

while True:
    user_input = input("You : ")

    if user_input.lower() == 'exit':
        break

    result = agent.invoke({
        "messages":[{"role" : "user" , "content" : user_input}]
    })

    print("Bot : ",result.get('messages')[-1].content)
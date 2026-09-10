from langchain_mistralai import ChatMistralAI
from langchain.messages import AIMessage , SystemMessage , HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model = 'ministral-3b-latest')
print("________________________________________WELCOME_______________________________________")
print("press 1 for the funnt mode")
print("Press 2 for the angry mode")
print("Press 3 for the sad mode")

mode = input("Mode : ")

sysmsg = ""
if mode == '1':
    sysmsg = "You are an funny ai , give reply in funny mode"

elif mode == '2':
    sysmsg = "You are an sad AI , give replay in angry mode"

elif mode == '3':
    sysmsg = "You are an sad ai , give replay in very sad mode"

print("-----------------Welcome , I am TRON your personal AI Companion(Enter 0 to exit)---------------------")
msg = [
    SystemMessage(content=sysmsg)
]

while True:
    prompt = input("You: ")
    msg.append(HumanMessage(content=prompt))
    if prompt == '0':
        break
    response = model.invoke(msg)
    msg.append(AIMessage(content=response.content))
    print("TRON: ",response.content)

print(msg)   
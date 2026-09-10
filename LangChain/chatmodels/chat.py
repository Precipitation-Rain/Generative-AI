from dotenv import load_dotenv

load_dotenv()

# using init_chat_model method
# from langchain.chat_models import init_chat_model
# model = init_chat_model("google_genai:gemini-3.6-flash")

# response = model.invoke("Greatest Coder of all Time?")

# response = response.content
# answer = response[0]['text']
# print(answer)


# # using the Model class method
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatGoogleGenerativeAI(model = 'gemini-3.6-flash')
# response = model.invoke("Tell some Greatest drives in F1 History")
# print(response.content[0]['text'])


# using groq api
# from langchain.chat_models import init_chat_model

# model = init_chat_model("qwen/qwen3.6-27b" , model_provider='groq' , max_tokens = 500)
# response = model.invoke("Who is Max verstappen")
# print(response.content)

# Using Model Class of the GROQ

# from langchain_groq import ChatGroq

# model = ChatGroq(model='qwen/qwen3.6-27b' , max_tokens=700)
# response = model.invoke("is axai chin is park of china or india?")
# print(response.content)


## USING THE MISTRAL

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = 'ministral-3b-latest')
response = model.invoke("Which contry developed you and how you are beter than other ai models")
print(response.content)

## PARAMETERS
# temprature :
# - values lies between 0 to 1
# value = 0 =====> reasoning and mathematics realted tasks , value = 1 ======> creativity related tasks

# max_tokens : 
# limits you output tokens



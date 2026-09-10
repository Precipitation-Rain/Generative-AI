from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = 'ministral-3b-latest')




from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages(
    [("system" , "You are an AI and have to summarize the text in very simple language and in brief") ,
     ("human" , "{docs}")]
)


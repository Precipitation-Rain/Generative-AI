from dotenv import load_dotenv
load_dotenv()


from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = 'ministral-3b-latest')


from langchain_community.document_loaders import TextLoader
data = TextLoader(r"D:\Study\Generative AI\RAG\documentLoaders\notes.txt")
docs = data.load()

from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages(
    [("system" , "You are an AI and have to summarize the text in very simple language and in brief") ,
     ("human" , "{docs}")]
)

prompt = template.format_messages(docs = docs[0].page_content)

response = model.invoke(prompt)
print(response.content)
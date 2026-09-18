from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# load the api keys
load_dotenv()

# embedding model 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# load that created vector_store
vectorstore = Chroma(
    persist_directory='chroma-DB',
    embedding_function=embedding_model
)

# create rtriever object
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 2,
        "fetch_k" : 5,
        "lambda_mult":0.5
    }
)

# llm object
llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [("system",
    """You are a helpful AI assistant.
    Use ONLY the provided context to answer the question.

    If the answer is not present in the context,
    say: "I could not find the answer in the document."
    """
    ),
    ("human",
    """Context: {context}
    Question : {question} """
        )]
)

print("==================RAG SYSTEM CREATED=====================")
while True:
    query = input("You : ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context":context,
            "question":query
        }
    )

    response = llm.invoke(final_prompt)
    print(f"\n Bot : {response.content}")




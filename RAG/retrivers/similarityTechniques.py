from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")

]

embedding_model = HuggingFaceEmbeddings()

vectoreStore = Chroma.from_documents(
    embedding=embedding_model,
    documents=docs,
    persist_directory='temp-db'
)

similarity_retriever = vectoreStore.as_retriever(
    search_type = "similarity",
    search_kwargs = {'k' : 3}
)

# similarity
print("===========SIMILARITY SEARCH RESULTS=============")
similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectoreStore.as_retriever(
    search_type = "mmr",
    search_kwargs = {'k' : 3}
)
 

# mmr
print("===========MMMR RESULTS=============")
similarity_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)
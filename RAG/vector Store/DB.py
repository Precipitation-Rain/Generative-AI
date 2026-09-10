from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

vectorStore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-DB"
)

result = vectorStore.similarity_search("What is used for data analysis" , k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

    



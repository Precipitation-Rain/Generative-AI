#Load the api keys
from dotenv import load_dotenv
load_dotenv()

#load the documents
from langchain_community.document_loaders import PyPDFLoader
data = PyPDFLoader(r"D:\Study\Generative AI\RAG\documentLoaders\deeplearning.pdf")
docs = data.load()

#do chunking of the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

#create embeddings and store into vector database
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory = "chroma-DB"
)






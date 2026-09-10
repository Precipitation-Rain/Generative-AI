from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"D:\Study\Generative AI\RAG\documentLoaders\GRU.pdf")
docs = data.load()

split = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=0
)

chunks = split.split_documents(docs)
print(len(chunks))

print(chunks[0].page_content)


from langchain_community.document_loaders import TextLoader

data = TextLoader(r"D:\Study\Generative AI\RAG\documentLoaders\notes.txt")

docs = data.load()

print(docs[0].metadata)


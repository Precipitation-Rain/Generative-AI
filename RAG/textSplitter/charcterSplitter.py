from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

data = TextLoader(r"D:\Study\Generative AI\RAG\documentLoaders\notes.txt")
docs = data.load()

split = CharacterTextSplitter(
    separator="",
    chunk_size = 100,
    chunk_overlap=0
)

chunks = split.split_documents(docs)
print(len(chunks))

for chunk in chunks:
    print(chunk.page_content)
    print()
    print()



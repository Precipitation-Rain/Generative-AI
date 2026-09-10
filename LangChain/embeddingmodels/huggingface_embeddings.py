from langchain_huggingface import ChatHuggingFace , HuggingFaceEmbeddings
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model

texts = [
    "Hello i am Rajvardhan",
    "Whay are you doing",
    "Who are you"   
]

# Provider Specific
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector = embedding.embed_documents(texts)
print(vector)

# Universal:

"""model = init_embeddings("sentence-transformers/all-MiniLM-L6-v2" , provider='huggingface' )
vector = model.embed_documents(texts)"""


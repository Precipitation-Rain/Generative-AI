from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)

texts = [
    "Hello i am Rajvardhan",
    "Whay are you doing",
    "Who are you"
]

# vector = embedding.embed_query("Hello I am a boy")
vector = embedding.embed_documents(texts , output_dimensionality=20)

print(vector)
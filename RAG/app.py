import os
import shutil
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

DB_PATH = "chroma-DB"


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions from its content.")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    st.divider()

    st.info(
        "Upload a PDF first, then ask questions "
        "using the chat box."
    )


# --------------------------------------------------
# CREATE RAG DATABASE
# --------------------------------------------------

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    pdf_path = os.path.join(
        "temp",
        uploaded_file.name
    )

    os.makedirs("temp", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # --------------------------------------------------
    # DELETE OLD DATABASE
    # --------------------------------------------------

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


    # --------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------

    with st.spinner("📖 Reading PDF..."):

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()


    # --------------------------------------------------
    # CHUNKING
    # --------------------------------------------------

    with st.spinner("✂️ Splitting document..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)


    # --------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------

    with st.spinner("🔢 Creating embeddings..."):

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )


    # --------------------------------------------------
    # CREATE VECTOR DATABASE
    # --------------------------------------------------

    with st.spinner("🗄️ Creating vector database..."):

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )


    # --------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 5,
            "lambda_mult": 0.5
        }
    )


    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )


    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:

"I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )


    # --------------------------------------------------
    # SUCCESS MESSAGE
    # --------------------------------------------------

    st.success(
        f"✅ {uploaded_file.name} is ready!"
    )

    st.write(
        f"**Pages:** {len(docs)}  |  "
        f"**Chunks:** {len(chunks)}"
    )


    # --------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------

    question = st.chat_input(
        "Ask something about your PDF..."
    )


    # --------------------------------------------------
    # RAG PIPELINE
    # --------------------------------------------------

    if question:

        # Show user question
        with st.chat_message("user"):
            st.write(question)


        # Retrieve relevant chunks
        with st.spinner("🔎 Searching document..."):

            retrieved_docs = retriever.invoke(question)


        # Convert chunks into context
        context = "\n\n".join(
            [
                doc.page_content
                for doc in retrieved_docs
            ]
        )


        # Create final prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("🤖 Generating answer..."):

                response = llm.invoke(
                    final_prompt
                )

                st.write(response.content)


else:

    st.info(
        "👈 Upload a PDF from the sidebar to start."
    )


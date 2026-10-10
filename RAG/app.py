"""
PDF RAG Assistant (Streamlit)

Upload one or more PDFs, then ask questions. Answers come only from the
uploaded documents, and every answer shows the pages it was drawn from.

Design notes (why this version works when deployed):
- The vector store is IN MEMORY (no persist_directory). Nothing is written to
  disk, so there is no database to store, upload or clean up.
- The index is built ONCE per set of uploaded files and kept in
  st.session_state. Asking a question no longer re-reads and re-embeds the PDF.
- The embedding model and LLM client are loaded once with st.cache_resource.
- API key comes from the environment / .env locally, or from Streamlit
  secrets when deployed. It is never hard-coded.

Run locally:  streamlit run app.py
"""

import hashlib
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Small embedding model (~90 MB) so the app fits free-tier memory limits.
# Swap for "sentence-transformers/all-mpnet-base-v2" if you host somewhere
# with more RAM and want slightly better retrieval.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
FETCH_K = 12
MAX_TOTAL_MB = 30  # guard against uploads that would exhaust free-tier RAM

NOT_FOUND_MESSAGE = "I could not find the answer in the document."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n"
            "Use ONLY the provided context to answer the question.\n"
            f'If the answer is not present in the context, say: "{NOT_FOUND_MESSAGE}"',
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def ensure_api_key() -> bool:
    """Make sure GROQ_API_KEY is set (env/.env locally, secrets when deployed)."""
    if os.getenv("GROQ_API_KEY"):
        return True
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets file configured
        key = None
    if key:
        os.environ["GROQ_API_KEY"] = key
        return True
    return False


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0)


# --------------------------------------------------------------------------
# Core RAG logic (no Streamlit calls below this line except in main())
# --------------------------------------------------------------------------
def load_and_split(files):
    """files: list of (filename, bytes). Returns (chunks, total_pages)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks, total_pages = [], 0

    for name, data in files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            pages = PyPDFLoader(path).load()
        finally:
            os.remove(path)

        for page in pages:
            page.metadata["source"] = name  # show the real filename, not a temp path
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1  # 1-based
        total_pages += len(pages)
        chunks.extend(splitter.split_documents(pages))

    chunks = [c for c in chunks if c.page_content.strip()]
    return chunks, total_pages


def build_index(files, embeddings):
    """Build an in-memory Chroma index. Nothing is written to disk."""
    chunks, total_pages = load_and_split(files)
    if not chunks:
        raise ValueError(
            "No extractable text found. The PDF may be a scanned image "
            "(this app does not do OCR)."
        )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"rag_{uuid.uuid4().hex}",  # unique per session, no collisions
    )
    return {"vectorstore": vectorstore, "pages": total_pages, "chunks": len(chunks)}


def files_signature(files) -> str:
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode())
        digest.update(data)
    return digest.hexdigest()


def ensure_index(files, state, embeddings):
    """Build the index only when the uploaded files actually changed."""
    signature = files_signature(files)
    current = state.get("index")
    if current is not None and current["signature"] == signature:
        return current

    if current is not None:  # free the old in-memory collection
        try:
            current["vectorstore"].delete_collection()
        except Exception:
            pass

    index = build_index(files, embeddings)
    index["signature"] = signature
    state["index"] = index
    state["messages"] = []  # new documents -> fresh conversation
    return index


def answer_question(question, vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
    response = llm.invoke(PROMPT.invoke({"context": context, "question": question}))
    return response.content, docs


def format_sources(docs):
    return [
        {
            "label": f"{d.metadata.get('source', '?')} · page {d.metadata.get('page', '?')}",
            "snippet": d.page_content.strip().replace("\n", " ")[:300],
        }
        for d in docs
    ]


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['label']}**")
            st.caption(s["snippet"] + "…")


def main():
    st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")
    st.title("📚 PDF RAG Assistant")
    st.write("Upload PDFs, then ask questions. Answers are grounded in your documents only.")

    if not ensure_api_key():
        st.error(
            "GROQ_API_KEY is not set. Add it to a local `.env` file, or to "
            "Streamlit secrets when deployed."
        )
        st.stop()

    st.session_state.setdefault("index", None)
    st.session_state.setdefault("messages", [])

    with st.sidebar:
        st.header("📄 Documents")
        uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        st.caption("Documents are processed in memory and are not stored on the server.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if uploaded:
        files = [(f.name, f.getvalue()) for f in uploaded]
        total_mb = sum(len(data) for _, data in files) / (1024 * 1024)
        if total_mb > MAX_TOTAL_MB:
            st.error(f"Uploads total {total_mb:.1f} MB; the limit is {MAX_TOTAL_MB} MB.")
            st.stop()
        try:
            with st.spinner("Reading, chunking and embedding documents..."):
                index = ensure_index(files, st.session_state, load_embeddings())
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not process the documents: {e}")
            st.stop()

        with st.sidebar:
            st.success(f"Ready · {len(files)} file(s)")
            st.write(f"**Pages:** {index['pages']}  |  **Chunks:** {index['chunks']}")
    else:
        st.info("👈 Upload one or more PDFs from the sidebar to start.")

    # Conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            render_sources(msg.get("sources"))

    index = st.session_state.get("index") if uploaded else None
    question = st.chat_input("Ask something about your PDFs...", disabled=index is None)

    if question and index is not None:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and generating answer..."):
                    reply, docs = answer_question(question, index["vectorstore"], load_llm())
                sources = [] if NOT_FOUND_MESSAGE in reply else format_sources(docs)
                st.write(reply)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "sources": sources}
                )
            except Exception as e:
                st.error(f"The model call failed: {e}")


if __name__ == "__main__":
    main()
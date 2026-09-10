import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ---------------- MODEL ----------------

model = ChatMistralAI(
    model="ministral-3b-latest"
)

# ---------------- SESSION MEMORY ----------------

if "msg" not in st.session_state:
    st.session_state.msg = [
        SystemMessage(
            content="You are an funny ai, give answer in funny way but don't bore the next person(user)"
        )
    ]

# ---------------- UI ----------------

st.title("🤖 TRON")
st.write("Your Personal AI Companion")

st.divider()

# Display previous conversation
for message in st.session_state.msg:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


# ---------------- USER INPUT ----------------

prompt = st.chat_input("Talk to TRON...")

if prompt:

    # Exit functionality remains the same
    if prompt == "0":
        st.stop()

    # Add user's message
    st.session_state.msg.append(
        HumanMessage(content=prompt)
    )

    # Show user's message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from model
    response = model.invoke(st.session_state.msg)

    # Add AI response to conversation
    st.session_state.msg.append(
        AIMessage(content=response.content)
    )

    # Show AI response
    with st.chat_message("assistant"):
        st.write(response.content)
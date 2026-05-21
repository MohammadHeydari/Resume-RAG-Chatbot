import streamlit as st
from rag_graph import ask

st.set_page_config(page_title="RAG ChatBot")

st.title("My RAG ChatBot (Ollama + LangGraph)")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:

    # user
    st.chat_message("user").markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # get response from RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking... "):
            response = ask(user_input)
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
import streamlit as st
from rag import answer

st.title("PaperTrail")

# 1. Initialize history once
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Re-render the full history on every rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask a question about your documents")
if query:
    # 3. Add + show the user's message
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # 4. Add + show the assistant's answer
    response = answer(query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
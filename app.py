import streamlit as st
from rag import answer

st.title("PaperTrail")

# Helper function to display messages with sources
def show_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for chunk in msg["sources"]:
                    st.markdown(f"**{chunk.metadata['source']}** --- page {chunk.metadata.get('page_label')}")
                    st.write(chunk.page_content[:200] + "…")

# 1. Initialize history once
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Re-render the full history on every rerun
for msg in st.session_state.messages:
    show_message(msg)

query = st.chat_input("Ask a question about your documents")
if query:
    # 3. Add + show the user's message
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)
    show_message(user_msg)

    # 4. Add + show the assistant's answer
    response, sources = answer(query)
    assistant_msg = {"role": "assistant", "content": response, "sources": sources}
    st.session_state.messages.append(assistant_msg)
    show_message(assistant_msg)
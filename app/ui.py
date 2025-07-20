import streamlit as st
from core.retriever import Retriever
from core.generator import Generator

# # initialize
@st.cache_resource
def init_components():
    retriever = Retriever()
    generator = Generator()
    return retriever, generator

retriever, generator = init_components()


def app():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("RAG-Powered Chatbot")

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:")
    if st.button("Send") and user_input:
        docs = retriever.retrieve(user_input) 
        # docs = ""
        response = generator.generate(user_input, docs)
        # response = "hi"
        st.session_state.history.append((user_input, response))

    for user_q, bot_a in st.session_state.history:
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Bot:** {bot_a}")

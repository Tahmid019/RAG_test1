import streamlit as st
import requests



API_URL = "http://localhost:5000/query"

def get_answer(query):
    try:
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json().get('answer', 'An error occurred.')
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}. Is the backend server running?"

def ui():
    st.title(" RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


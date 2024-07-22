import streamlit as st
import traceback
from backend.src.rag_llama_index_langchain_backend.model import response_generator

st.title("Rag Application using llama_index and langchain")

with st.form("user inputs"):
    user_query = st.text_input("Enter Your Question related to Attention is all you need Or YOLO")
    button = st.form_submit_button("Answer")

    if button is not None and user_query:
        with st.spinner("Loading......"):
            try:
                final_response = response_generator(user_query=user_query)
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                st.write(final_response)





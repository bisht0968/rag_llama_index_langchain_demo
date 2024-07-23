import streamlit as st
import traceback
from backend.src.rag_llama_index_langchain_backend.model import response_generator,train_model,pdf_response_generator,llm_response_generator

st.title("Rag Application using llama_index and langchain")


@st.cache_resource
def get_trained_model():
    return train_model()

query_engine = get_trained_model()

with st.form("user inputs"):
    user_query = st.text_input("Enter Your Question related to Attention is all you need Or YOLO.")
    rag_query_button = st.form_submit_button("Answer")
    st.write("Or upload a PDF and ask a question related to it:")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    pdf_user_query = st.text_input("Enter Your Question related to the PDF")
    pdf_button = st.form_submit_button("Submit")
    st.write("Or Ask a general Question:")
    general_user_query = st.text_input("Enter Your Question?")
    general_button = st.form_submit_button("Answer the Question")

    if rag_query_button and user_query:
        with st.spinner("Loading......"):
            try:
                final_response = response_generator(query_engine=query_engine,user_query=user_query)
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                st.write(final_response)
    elif pdf_button and pdf is not None and pdf_user_query:
            with st.spinner("Loading......"):
                try:
                    pdf_final_response = pdf_response_generator(pdf_user_query=pdf_user_query,pdf = pdf)
                except Exception as e:
                    traceback.print_exception(type(e),e,e.__traceback__)
                    st.error("Error")
                else:
                    st.write(pdf_final_response)
    elif general_button and general_user_query:
        with st.spinner("Loading......"):
            try:
                general_final_response = llm_response_generator(general_user_query)
            except Exception as e:
                    traceback.print_exception(type(e),e,e.__traceback__)
                    st.error("Error")
            else:
                st.write(general_final_response)





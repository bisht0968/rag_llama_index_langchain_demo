import os
from dotenv import load_dotenv
from .utils import extract_pdfs, embedding_model, initiliaze_llm, service_context_model, query_formatting,response_fromatting,read_file,pdf_query_formatting,pdf_response_fromatting,llm_query_prompting,llm_response_formatting
from llama_index.core import VectorStoreIndex
from langchain.chains import LLMChain

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

def train_model():
    documents = extract_pdfs("pdfs")
    embed_model = embedding_model("sentence-transformers/all-mpnet-base-v2")
    llm = initiliaze_llm(HUGGINGFACE_API_TOKEN)
    service_context = service_context_model(embed_model = embed_model,llm = llm,chunk_size = 500,chunk_overlap = 150)
    index = VectorStoreIndex.from_documents(documents,service_context = service_context)
    query_engine = index.as_query_engine()
    return query_engine

def pdf_response_generator(pdf_user_query,pdf):
    documents = read_file(pdf)
    embed_model = embedding_model("sentence-transformers/all-mpnet-base-v2")
    llm = initiliaze_llm(HUGGINGFACE_API_TOKEN)
    service_context = service_context_model(embed_model = embed_model,llm = llm,chunk_size = 500,chunk_overlap = 150)
    index = VectorStoreIndex.from_documents(documents,service_context = service_context)
    pdf_query_engine = index.as_query_engine()
    formatted_pdf_query = pdf_query_formatting(pdf_user_query=pdf_user_query)
    pdf_response = pdf_query_engine.query(formatted_pdf_query)
    pdf_final_response = pdf_response_fromatting(pdf_response=pdf_response)
    return pdf_final_response


def response_generator(query_engine,user_query):
    formatted_query = query_formatting(user_query=user_query)
    response = query_engine.query(formatted_query)
    final_response = response_fromatting(response=response)
    return final_response

def llm_response_generator(llm_user_query):
    llm = initiliaze_llm(HUGGINGFACE_API_TOKEN)
    llm_prompt = llm_query_prompting()
    chain = LLMChain(llm=llm,prompt = llm_prompt)
    llm_response = chain.run({"llm_user_query" : llm_user_query})   
    format_llm_response = llm_response_formatting(llm_response)
    return format_llm_response
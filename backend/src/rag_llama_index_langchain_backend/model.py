import os
from dotenv import load_dotenv
from .utils import extract_pdfs, embedding_model, initiliaze_llm, service_context_model, query_formatting,response_fromatting
from llama_index.core import VectorStoreIndex

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
def response_generator(user_query):

    documents = extract_pdfs("pdfs")

    embed_model = embedding_model("sentence-transformers/all-mpnet-base-v2")

    llm = initiliaze_llm(HUGGINGFACE_API_TOKEN)

    service_context = service_context_model(embed_model = embed_model,llm = llm,chunk_size = 500,chunk_overlap = 150)

    index = VectorStoreIndex.from_documents(documents,service_context = service_context)

    query_engine = index.as_query_engine()

    formatted_query = query_formatting(user_query=user_query)

    response = query_engine.query(formatted_query)

    final_response = response_fromatting(response=response)
    return final_response
from llama_index.core import SimpleDirectoryReader,ServiceContext
from langchain_community.llms import HuggingFaceHub
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings

def initiliaze_llm(HUGGINGFACE_API_TOKEN):
    return HuggingFaceHub(
        huggingfacehub_api_token = HUGGINGFACE_API_TOKEN,
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs = {
            'temperature' : 0.8,
            'max_new_tokens' : 500,
            'do_sample' : False
        }
    )

def extract_pdfs(directory_path):
    return SimpleDirectoryReader(directory_path).load_data()

def query_formatting(user_query):
    system_prompt = """
    You are a Q&A assistant. 
    Your goal is to answer questions as accurately as possible based on the instructions and context provided.
    """
    query_wrapper_prompt = SimpleInputPrompt("""
    Question:
    {query_str}

    """
    )

    return f"{system_prompt}\n{query_wrapper_prompt.format(query_str = user_query)}"

def response_fromatting(response):
    response_text = response.response
    start_index = response_text.find("Answer:")
    answer = response_text[start_index:].strip()
    return answer

def embedding_model(model_name):
    return LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name = model_name
        )
    )

def service_context_model(llm,embed_model,chunk_size,chunk_overlap):
    return ServiceContext.from_defaults(
        llm = llm,
        embed_model = embed_model,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )


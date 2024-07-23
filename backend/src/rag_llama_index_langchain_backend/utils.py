import PyPDF2
from llama_index.core import SimpleDirectoryReader,ServiceContext,Document
from langchain_community.llms import HuggingFaceHub
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

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

def pdf_query_formatting(pdf_user_query):
    system_prompt = """
    You are a Q&A assistant. 
    Your goal is to answer questions as accurately as possible based on the instructions and context provided.
    """
    query_wrapper_prompt = SimpleInputPrompt("""
    Question:
    {pdf_query_str}

    """
    )

    return f"{system_prompt}\n{query_wrapper_prompt.format(pdf_query_str = pdf_user_query)}"

def pdf_response_fromatting(pdf_response):
    response_text = pdf_response.response
    start_index = response_text.find("Answer:")
    answer = response_text[start_index:].strip()
    return answer

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

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return [Document(text=text)]
        except Exception as e:
            raise Exception("Error reading the PDF file")
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
        return [Document(text=text)]
    else:
        raise Exception("Unsupported File Format, only PDF and TXT files are supported")
    
def llm_query_prompting():
    prompt = PromptTemplate(
        input_variables=['llm_user_query'],
        template="""
        You are a helpful assistant with extensive knowledge in various topics. Your goal is to provide accurate and clear answers to the questions asked. Here is the question you need to address:

        Question:
        {llm_user_query}

        Provide a comprehensive and well-structured response based on your knowledge and understanding.
        Answer:
        """
    )
    
    return prompt

def llm_response_formatting(llm_response):
    start_index = llm_response.find("Answer:")
    answer = llm_response[start_index:].strip()
    return answer
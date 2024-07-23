import PyPDF2
from llama_index.core import Document
file_path = "../../goodfellow.pdf"

import os
from dotenv import load_dotenv

from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return [Document(text)]
        except Exception as e:
            raise Exception("Error reading the PDF file")
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
        return [Document(text)]
    else:
        raise Exception("Unsupported File Format, only PDF and TXT files are supported")
    
def read_file_document(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception("Error reading the PDF file")
    elif file.name.endswith(".txt"):
        return file.read().decode("utf 8")
    else:
        raise Exception("Unsupported File Format, only PDF and TXT files are supported")

    

# Open the file and pass it to the function
# with open(file_path, "rb") as file:
#     documents = read_file_document(file)

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
    answer = llm_response[start_index + 7:].strip()
    return answer

def llm_response_generator(llm_user_query):
    llm = initiliaze_llm(HUGGINGFACE_API_TOKEN)
    llm_prompt = llm_query_prompting()
    chain = LLMChain(llm=llm,prompt = llm_prompt)
    llm_response = chain.run({"llm_user_query" : llm_user_query})   
    format_llm_response = llm_response_formatting(llm_response)
    return format_llm_response

user_query = "What is the capital of India"

response = llm_response_generator(user_query)

print(response)
#%%
import os
#import streamlit as st
#from PyPDF2 import PdfReader
#from dotenv import load_dotenv
#from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings # when using OpenAI
# Identify Hugging Face Version of Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.llms import HuggingFaceHub
# Identify Hugging Face Version of ChatOpenAI
from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import Chroma
#from langchain_google_vertexai import VertexAI
#import vertexai
#from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from google.oauth2 import service_account
#from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
#%%
# initializing google service account
#open ai api key = "sk-proj-K1eogesejaXYuTpLYriAT3BlbkFJgC5nnalRjGRjEBdK5Db4"

# Path to your service account key file
#SERVICE_ACCOUNT_FILE = '/Users/vamseekrishna/Desktop/AOSS_lambda/vertexRAG/vertexKeys.json'

# Define the scopes needed for the credentials
os.environ["OPENAI_API_KEY"] = "sk-proj-K1eogesejaXYuTpLYriAT3BlbkFJgC5nnalRjGRjEBdK5Db4"

#%%
with open('srs.txt', 'r') as file:
    # Read the contents of the file
    file_contents = file.read()

# Print the contents of the file
#print(file_contents)

#%%
#text = read_pdf(pdf_path_1)
text = file_contents
print(text)
#%%

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

# Process the PDF text and create the documents list
documents = text_splitter.split_text(text=text)
#%%
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",)







#%%
# Vectorize the documents and create vectorstore
#embeddings = HuggingFaceEmbeddings()
#embeddings = VertexAIEmbeddings("text-embedding-004")
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyArQeHPf5ePfkhUyjDRTx6kI7jG7YDzCdI" )
embeddings.embed_query("hello, world!")
"""

#%%
#embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
vectorstore = FAISS.from_texts(documents, embedding=embeddings) #error line
#vectorstore = Chroma.from_texts(documents, embedding=embeddings)
"""
loader = PyPDFLoader(pdf_path_1)
documents = loader.load_and_split()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
context = "\n\n".join(str(p.page_content) for p in documents)
texts = text_splitter.split_text(context)
vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
vector_index = Chroma.from_texts(texts, vertex_embeddings).as_retriever()
"""

#%%
#save the vector store locally
#retrieve the vector store again locally. 

index_path = "/Users/vamseekrishna/Desktop/RAG_demo/indices"
index_name = "va_qa"
vectorstore.save_local(folder_path = index_path,index_name=index_name)
vectorstore = FAISS.load_local(folder_path=index_path,embeddings=embeddings,allow_dangerous_deserialization=True,index_name=index_name)

#%%
#llm = VertexAI(model_name="gemini-1.0-pro-001",)
llm = OpenAI(openai_api_key="sk-proj-K1eogesejaXYuTpLYriAT3BlbkFJgC5nnalRjGRjEBdK5Db4",model="gpt-3.5-turbo-instruct")
#%%
"""
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)
"""
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
#%%
question = """how to get in touch with SRS"""
#%%
docs = vectorstore.similarity_search(question)
context = "context :"
for doc in docs:
    context = context + doc.page_content + "\n"
    print(doc)
#%%
prompt = "answer the question below, if you don't find any answer in the context, reply you have no answer. "
print(prompt)
context2 = f"""context : {context} 
            question : {question}"""
prompt_template = f""" {prompt} 
                    \n {context2}"""



#%%
def answer_from_pdf(question):
  result = qa({"question":question, "chat_history":""})
  return result["answer"]
#%%
print(answer_from_pdf(question = prompt_template))
# %%



# %%

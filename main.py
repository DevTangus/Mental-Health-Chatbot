import openai
import uvicorn

from fastapi import FastAPI, Request, Form
from langchain.chains import retrieval_qa
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import csv_loader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch

app = FastAPI()

#Set OpenAI API Key
OPENAI_API_KEY = "sk-proj-O5VKxVc51IgCEKdoa87TT3BlbkFJkf6XMmRVnLUEJVFIwT9n"

def setup_chain():
    #File path and template
    file = 'Mental_Health-FAQ.csv'
    template = """"""
    #Template contents
    """"""
    #Initializing embeddings, loader,and prompt
    embeddings = OpenAIEmbeddings()
    loader = csv_loader(file_path=file,encoding='utf-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template,input_variables)
    
    #create DocArrayInMenorySearch and retriever
    db = DocArrayInMemorySearch.from_documents(docs,embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs = {"prompt":prompt}
    #Initialize ChatOpenAI
    llm= ChatOpenAI(
        temperature=0)
    #setup Retrieval QA Chain
    chain = retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return chain
agent = setup_chain()




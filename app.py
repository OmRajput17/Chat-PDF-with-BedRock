import os 
import sys
import json
import boto3
import streamlit as st
from botocore.exceptions import ClientError

## Using titan Embedding model for generating embeddings

from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

## Libraries for data ingestiion

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embeddings and vector Stores

from langchain_community.vectorstores import FAISS

### LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock clients
bedrock = boto3.client(service_name = "bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id = "amazon.titan-embed-text-v1",
    client = bedrock,
)

## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    ## We use Recursive Character Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 1500,
    )

    docs = text_splitter.split_documents(documents=documents)

    return docs

## vector Embeddings and Vector Stores

def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(
            documents=docs,
            embedding=bedrock_embeddings,
        )

        vectorstore_faiss.save_local("faiss_index")

    except(ClientError, Exception) as e:
        print(f"ERROR: Can't invoke. Reason: {e}")
        exit(1)

def get_llama_llm():
    ### Create a Ollama Model
    model_id = "meta.llama3-70b-instruct-v1:0"
    try:
        llm = Bedrock(
            model_id = "meta.llama3-70b-instruct-v1:0",
            client = bedrock,
            model_kwargs = {
                "max_gen_len": 512,
                "temperature": 0.8,
            }
        )
    except(ClientError, Exception) as e:
        print(f"ERROR: Can't invoke'{model_id}'. Reason: {e}")
        exit(1)

    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question
at the end but use atleast summaroze with 250 words with detailed explanation. If you don't 
know the answer just say you don't know the answer.
<context>
{context}
<context>
Question : {question}

Assistant :
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context","question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt":PROMPT}
    )

    answer = qa({"query":query})

    return answer['result']


def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF using AWS BedRock")

    user_question = st.text_input("Ask a question from the pdf files")

    with st.sidebar:
        st.title("Update or Create Vector Stores:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs=docs)
                st.success("Done")

    if st.button("Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", 
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            llm = get_llama_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
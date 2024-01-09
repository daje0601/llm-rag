import time
from uuid import UUID
import streamlit as st
from typing import Any, Dict, List, Optional, Union

from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS

from langchain.chat_models import ChatOllama, ChatOpenAI

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output import ChatGenerationChunk, GenerationChunk 

from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(
    page_title="llm-rag",
    page_icon="🔒",
)


# 파일을 업로드하고, 
@st.cache_data(show_spinner="embed_file...")
def embed_file(file):
    file_content =file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f : 
        f.write(file_content)
    

    cached_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50, 
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(
        model="mistral:latest",
    )
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cached_dir)

    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    retriever = vectorstore.as_retriever(reduce_k_below_max_tokens=True)
    return retriever

def save_messages(message, role):
    st.session_state["messages"].append({"message" : message, "role" : role})
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save : 
        save_messages(message, role)


def pain_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the questio using ONLY the following context. If you don't know the answer just say you don't know. 
     Don't make anything up. Please answer in Korean.
    
    Context: {context}
    """),
    ("human", "{question}")
])

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_messages(self.message, "ai")
    
    
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1, 
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

st.title("llm-rag")
st.markdown("""
안녕하세요!

원하는 데이터를 올리고, 챗봇에서 자유롭게 질문해주세요. 
""")

with st.sidebar:
    file = st.file_uploader("upload a .txt, .pdf or .docx files ", type=["pdf", "txt", "docx"])



if file:
    retriever = embed_file(file)
    
    send_message("준비된 무엇이든 보내봐 ", "ai", save=False)
    pain_history()
    message = st.chat_input("파일에 대해서 무엇이든 물어보세요")

    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs), 
            "question" :  RunnablePassthrough()
        } | prompt | llm 
        with st.chat_message("ai"):
            response = chain.invoke(message)

else: 
    st.session_state["messages"] = [] 

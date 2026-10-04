import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
import uuid

load_dotenv()





prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "You have your pretrained knowledge and retrieved context. "
        "Use the retrieved context when it is relevant. "
        "If the context is irrelevant, use your own knowledge. "
        "If the context only partially answers the question, combine it "
        "with your own knowledge.\n\n"
        "Retrieved Context:\n{context}"
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "human",
        "{input}"
    )
])

@st.cache_resource
def  create_vector_db():
      loader  = PyPDFLoader("global_facts_2026.pdf")
      docs = loader.load()
      splitter= RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
      chunk_data = splitter.split_documents(docs)
      embedder  =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
      vectorDB = FAISS.from_documents(chunk_data, embedder)

      return vectorDB

vectorDB = create_vector_db()

retriever = vectorDB.as_retriever(search_kwargs={"k": 2})



st.title("LangChain + RAG + Hugging Face Q&A Chatbot")
input_text = st.text_area(
    "Ask a question or paste your code and error:",
    height=200
)
submit_button = st.button("Submit")



@st.cache_resource
def create_llm():

    llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    max_new_tokens=2024,
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)
    return ChatHuggingFace(llm=llm)



chat_llm = create_llm()

document_chain = create_stuff_documents_chain(chat_llm,prompt_template)

retriever_chain = create_retrieval_chain(retriever,document_chain)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def get_chat_history(session_id)->BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

chat_with_history = RunnableWithMessageHistory(retriever_chain,
                                               get_chat_history,
                                               input_messages_key="input",
                                               history_messages_key = "messages", 
                                               output_messages_key="answer")
session_id = st.session_state["session_id"]
config = {"configurable":{"session_id":session_id}}

if submit_button:

    if input_text:

        with st.spinner("Generating answer..."):

            try:

                response = chat_with_history.invoke({"input": input_text},config)

                st.write(response["answer"])

            except Exception as e:

                st.error("LLM Error:")
                st.exception(e)

    else:
        st.warning("Please enter a question.")

import streamlit as st
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import os

# Load environment variables
load_dotenv()

# Get API keys from env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Required to be set explicitly
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings and retriever
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot", embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    max_tokens=500,
)

# Prompt and RAG chain
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Streamlit UI
st.set_page_config(page_title="🩺 Medical Chatbot", layout="centered")
st.title("🩺 Medical Chatbot (Groq + Pinecone)")

query = st.text_input("Ask a medical question:")

if query:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": query})
        st.markdown("### 🤖 Answer")
        st.success(response["answer"])

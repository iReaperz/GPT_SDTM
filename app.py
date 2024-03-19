import streamlit as st
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import os
import shutil
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma as ChromaDB
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from streamlit.components.v1 import html

# Load the OpenAI API key from file and set it as an environment variable
api_key_path = 'openai_api_key.txt'
try:
    with open(api_key_path, 'r') as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
except FileNotFoundError:
    st.error(f"API key file '{api_key_path}' not found. Please make sure the file exists.")
    st.stop()

# Define constants and configurations
CHROMA_PATH = "chromadb"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Streamlit app title and layout configurations
st.set_page_config(page_title='Bank Information Retrieval Assistant', layout='wide')

# Custom CSS
custom_css = """
<style>
    /* Center the title text */
    h1 {
        text-align: center;
    }
    /* Style for the query input */
    .stTextInput > div > div > input {
        color: white;
        background-color: #333;
        border: 1px solid #fff;
        font-size: 1.1em;
    }
    /* Style for the response and sources containers */
    .response-container, .source-container {
        background-color: black;
        color: white;
        border: 1px solid #fff;
        padding: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
        margin-top: 5px;
        white-space: pre-wrap;
    }
    /* Style for the response and sources text */
    .response-container p, .source-container p {
        margin: 0;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title in the center
st.markdown('<h1>Bank Information Retrieval Assistant</h1>', unsafe_allow_html=True)

# User input for query
query_text = st.text_input("", placeholder="Ask a question...", key="query")

# Process the query when Enter is pressed or the button is clicked
if query_text:
    response_placeholder = st.empty()
    source_placeholder = st.empty()

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if not results or results[0][1] < 0.7:
        response_placeholder.markdown("<div class='response-container'><p>Unable to find matching results.</p></div>", unsafe_allow_html=True)
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ChatOpenAI()
        response_text = model.predict(prompt)
        
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        
        response_placeholder.markdown(
            f"<div class='response-container'><p>{response_text}</p></div>", 
            unsafe_allow_html=True
        )
        source_placeholder.markdown(
            f"<div class='source-container'><p>Sources: {', '.join(sources)}</p></div>",
            unsafe_allow_html=True
        )

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

from dotenv import load_dotenv

# Load the environment variables
# load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Adding the Title of the app
st.title("PDF Question Answering Chatbot 📄🤖💻")
st.markdown("<p style='color:blue'>Note: This is a demo version of the chatbot. It is trained on a pdf dataset and  able to answer all your questions related to the documents. 🙏</p>", unsafe_allow_html=True)
st.markdown("---")

# Create a container for the sidebar
sidebar = st.sidebar

# Set a background image or color for the sidebar

# Customize the sidebar with your desired layout
sidebar.title("PDF Chatbot")
sidebar.subheader("Interactive Q&A")
sidebar.markdown("---")

# Create a container for the main content
content = st.container()

with content:
    loader = DirectoryLoader('data', glob="**/*.pdf")
    docs = loader.load()
    # Printing the total number of documents
    st.write(f"Total number of documents: {len(docs)}")
    # make the color of the text red
     
    # Customize the layout of the main content
    col1, col2 = st.columns(2)
        
    # Add a form to input the user query
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your CSV data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        # Perform similarity search and retrieve the answer
        embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(docs, embeddings)
        output = vector_store.similarity_search(user_input)
        output = output[0].page_content
        
        # Store the user input and generated output in session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
            
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about your documents. 🤗"]
            
        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]
            
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        # Add a section title
        # st.header("Chat History")
        
        # Display the chat history
        response_container = st.container()
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

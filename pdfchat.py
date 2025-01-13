"""
PDF Chat Application
-------------------
A Streamlit application that enables conversational interaction with PDF documents
using Google's Generative AI with configurable models for different use cases.
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from enum import Enum
from typing import Dict, Any

# Load environment variables
load_dotenv()

class InteractionMode(Enum):
    """Defines different modes of interaction with the documents"""
    STRICT = "Strict (Factual Responses)"
    CREATIVE = "Creative (Elaborate Explanations)"
    ANALYSIS = "Analysis (Insights & Patterns)"

# Model configuration for different interaction modes
MODEL_CONFIGS: Dict[InteractionMode, Dict[str, Any]] = {
    InteractionMode.STRICT: {
        "temperature": 0.1,
        "prompt_template": """You are a precise and accurate AI assistant focused on providing factual information from the documents.
        Use the following context to answer the question. Stick strictly to the information provided.
        If the answer isn't directly supported by the context, say so clearly.
        
        {context}
        
        Question: {question}
        Factual Answer:"""
    },
    InteractionMode.CREATIVE: {
        "temperature": 0.7,
        "prompt_template": """You are a helpful AI tutor that explains concepts from documents in an engaging and detailed way.
        Use the following context as your source material, but feel free to elaborate with examples and explanations.
        
        {context}
        
        Question: {question}
        Detailed Explanation:"""
    },
    InteractionMode.ANALYSIS: {
        "temperature": 0.3,
        "prompt_template": """You are an analytical AI assistant that identifies patterns and insights from documents.
        Analyze the following context to provide thoughtful insights and connections.
        Support your analysis with specific references from the text.
        
        {context}
        
        Question: {question}
        Analysis:"""
    }
}

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Chat with your PDF 📚")
st.markdown("""
Upload PDF documents and interact with their content using natural language.
Choose different interaction modes to get responses tailored to your needs.
""")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = InteractionMode.STRICT

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the text into smaller chunks for processing"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore, mode: InteractionMode):
    """
    Create a conversation chain using Google's Generative AI
    
    Args:
        vectorstore: FAISS vector store containing document embeddings
        mode: InteractionMode determining the conversation behavior
    """
    config = MODEL_CONFIGS[mode]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=config["temperature"],
        convert_system_message_to_human=True
    )
    
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=config["prompt_template"]
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

def process_docs(pdf_docs):
    """Process uploaded PDF documents and initialize the conversation chain"""
    try:
        with st.spinner("Extracting text from PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            
        with st.spinner("Chunking text..."):
            text_chunks = get_text_chunks(raw_text)
            
        with st.spinner("Creating embeddings..."):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            
        with st.spinner("Setting up chat interface..."):
            st.session_state.vectorstore = vectorstore  # Store for mode switching
            st.session_state.conversation = get_conversation_chain(
                vectorstore,
                st.session_state.interaction_mode
            )
            st.session_state.processComplete = True
            
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar configuration
with st.sidebar:
    st.subheader("Document Upload")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDF files"
    )
    
    st.subheader("Interaction Settings")
    selected_mode = st.selectbox(
        "Choose interaction mode",
        options=[mode.value for mode in InteractionMode],
        help="""
        Strict: Direct, factual answers from the documents
        Creative: Detailed explanations with examples
        Analysis: Focus on patterns and insights
        """
    )
    
    # Update conversation chain if mode changes
    new_mode = InteractionMode(selected_mode)
    if st.session_state.get("interaction_mode") != new_mode:
        st.session_state.interaction_mode = new_mode
        if st.session_state.get("vectorstore"):
            st.session_state.conversation = get_conversation_chain(
                st.session_state.vectorstore,
                new_mode
            )
    
    if st.button("Process Documents", disabled=not pdf_docs):
        success = process_docs(pdf_docs)
        if success:
            st.success("Processing complete! You can now ask questions about your documents.")

# Main chat interface
if st.session_state.processComplete:
    # Display current mode
    st.info(f"Current mode: {st.session_state.interaction_mode.value}")
    
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.info("👈 Upload your PDFs in the sidebar and click 'Process Documents' to get started!")

"""
PDF Chat Application with Document Type Specialization
---------------------------------------------------
A Streamlit application that adapts its processing and interaction based on
document types and usage contexts.
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
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Load environment variables
load_dotenv()

class DocumentType(Enum):
    """Different types of documents with specific handling requirements"""
    TECHNICAL = "Technical Documentation"
    ACADEMIC = "Academic Papers"
    LEGAL = "Legal Documents"
    BUSINESS = "Business Reports"
    GENERAL = "General Content"

class UsageContext(Enum):
    """Different usage contexts that affect processing and interaction"""
    RESEARCH = "Research Analysis"
    SUMMARY = "Quick Summary"
    QA = "Question Answering"
    ANALYSIS = "Deep Analysis"

@dataclass
class DocumentConfig:
    """Configuration for document processing based on type"""
    chunk_size: int
    chunk_overlap: int
    temperature: float
    prompt_template: str
    model_name: str = "gemini-pro"  # Default model, can be overridden
    
# Configurations for different document types
DOCUMENT_CONFIGS: Dict[DocumentType, Dict[UsageContext, DocumentConfig]] = {
    DocumentType.TECHNICAL: {
        UsageContext.RESEARCH: DocumentConfig(
            chunk_size=1500,
            chunk_overlap=300,
            temperature=0.1,
            prompt_template="""You are a technical documentation expert. Analyze the following technical content 
            with precise attention to detail. Maintain technical accuracy and use proper terminology.
            {context}
            Question: {question}
            Technical Response:"""
        ),
        UsageContext.SUMMARY: DocumentConfig(
            chunk_size=2000,
            chunk_overlap=200,
            temperature=0.2,
            prompt_template="""Summarize this technical documentation clearly and concisely. 
            Focus on key technical concepts and implementation details.
            {context}
            Question: {question}
            Technical Summary:"""
        )
    },
    DocumentType.ACADEMIC: {
        UsageContext.RESEARCH: DocumentConfig(
            chunk_size=1000,
            chunk_overlap=200,
            temperature=0.1,
            prompt_template="""You are a research paper analyst. Examine the academic content thoroughly,
            maintaining academic rigor and precision in your analysis.
            {context}
            Question: {question}
            Academic Analysis:"""
        ),
        UsageContext.QA: DocumentConfig(
            chunk_size=800,
            chunk_overlap=150,
            temperature=0.1,
            prompt_template="""Provide precise answers based on the academic paper content.
            Cite specific sections when relevant.
            {context}
            Question: {question}
            Academic Response:"""
        )
    },
    DocumentType.LEGAL: {
        UsageContext.ANALYSIS: DocumentConfig(
            chunk_size=1200,
            chunk_overlap=300,
            temperature=0.1,
            prompt_template="""Analyze this legal document with attention to legal terminology and implications.
            Maintain precise legal language and context.
            {context}
            Question: {question}
            Legal Analysis:"""
        ),
        UsageContext.SUMMARY: DocumentConfig(
            chunk_size=1500,
            chunk_overlap=200,
            temperature=0.2,
            prompt_template="""Provide a clear summary of this legal document, highlighting key points
            while maintaining legal accuracy.
            {context}
            Question: {question}
            Legal Summary:"""
        )
    },
    DocumentType.BUSINESS: {
        UsageContext.ANALYSIS: DocumentConfig(
            chunk_size=1000,
            chunk_overlap=200,
            temperature=0.2,
            prompt_template="""Analyze this business document focusing on key business insights,
            metrics, and strategic implications.
            {context}
            Question: {question}
            Business Analysis:"""
        ),
        UsageContext.SUMMARY: DocumentConfig(
            chunk_size=1200,
            chunk_overlap=150,
            temperature=0.3,
            prompt_template="""Summarize this business document focusing on key findings,
            recommendations, and business impact.
            {context}
            Question: {question}
            Business Summary:"""
        )
    },
    DocumentType.GENERAL: {
        UsageContext.QA: DocumentConfig(
            chunk_size=1000,
            chunk_overlap=200,
            temperature=0.2,
            prompt_template="""Provide clear and helpful answers based on the document content.
            {context}
            Question: {question}
            Response:"""
        ),
        UsageContext.SUMMARY: DocumentConfig(
            chunk_size=1500,
            chunk_overlap=200,
            temperature=0.3,
            prompt_template="""Provide a clear and concise summary of the content.
            {context}
            Question: {question}
            Summary:"""
        )
    }
}

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Smart PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "doc_type" not in st.session_state:
    st.session_state.doc_type = DocumentType.GENERAL
if "usage_context" not in st.session_state:
    st.session_state.usage_context = UsageContext.QA

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str, config: DocumentConfig) -> list:
    """
    Split text into chunks based on document configuration
    
    Args:
        text: Raw text to split
        config: DocumentConfig containing chunk settings
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore, config: DocumentConfig):
    """
    Create a conversation chain using configuration-specific settings
    
    Args:
        vectorstore: FAISS vector store containing document embeddings
        config: DocumentConfig containing model settings
    """
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        temperature=config.temperature,
        convert_system_message_to_human=True
    )
    
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=config.prompt_template
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

def process_docs(pdf_docs, doc_type: DocumentType, usage_context: UsageContext):
    """
    Process documents using type-specific configurations
    
    Args:
        pdf_docs: List of uploaded PDF files
        doc_type: Type of document being processed
        usage_context: Context in which the document will be used
    """
    try:
        config = DOCUMENT_CONFIGS[doc_type][usage_context]
        
        with st.spinner("Extracting text from PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            
        with st.spinner(f"Chunking text (size: {config.chunk_size}, overlap: {config.chunk_overlap})..."):
            text_chunks = get_text_chunks(raw_text, config)
            
        with st.spinner("Creating embeddings..."):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            
        with st.spinner("Setting up specialized chat interface..."):
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore, config)
            st.session_state.processComplete = True
            
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Application title and description
st.title("Smart PDF Chat 📚")
st.markdown("""
This advanced PDF chat system adapts to your document type and usage needs.
Select the appropriate document type and usage context for optimal results.
""")

# Sidebar configuration
with st.sidebar:
    st.subheader("Document Configuration")
    
    # Document upload
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )
    
    # Document type selection
    selected_doc_type = st.selectbox(
        "Document Type",
        options=[dt.value for dt in DocumentType],
        help="Select the type of document you're uploading"
    )
    
    # Get available usage contexts for selected document type
    doc_type = DocumentType(selected_doc_type)
    available_contexts = [
        context.value for context in UsageContext
        if context in DOCUMENT_CONFIGS[doc_type]
    ]
    
    # Usage context selection
    selected_context = st.selectbox(
        "Usage Context",
        options=available_contexts,
        help="Select how you plan to use this document"
    )
    
    # Update session state
    new_doc_type = DocumentType(selected_doc_type)
    new_context = UsageContext(selected_context)
    
    # Process button
    if st.button("Process Documents", disabled=not pdf_docs):
        success = process_docs(pdf_docs, new_doc_type, new_context)
        if success:
            st.success("Processing complete! Documents configured for " +
                      f"{selected_doc_type} with {selected_context} context.")

# Main chat interface
if st.session_state.processComplete:
    # Display current configuration
    st.info(f"Document Type: {st.session_state.doc_type.value} | " +
            f"Usage Context: {st.session_state.usage_context.value}")
    
    user_question = st.chat_input("Ask about your documents:")
    
    if user_question:
        try:
            with st.spinner("Processing your question..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.info("👈 Start by uploading your documents and selecting appropriate settings in the sidebar.")

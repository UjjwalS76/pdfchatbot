import streamlit as st
from enum import Enum
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class AnalysisMode(Enum):
    COMPREHENSION = "Basic Comprehension"
    RISK = "Risk Analysis"
    CORNER_CASES = "Corner Case Exploration"
    COMPARISON = "Contract Comparison"

ANALYSIS_CONFIGS = {
    AnalysisMode.COMPREHENSION: {
        "temperature": 0.1,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "prompt": """You are a legal contract analyst focused on clear comprehension.
        Based on the following contract sections, provide a clear, straightforward explanation.
        Focus on key terms, obligations, and basic implications.
        
        Contract sections:
        {context}
        
        Question: {question}
        Clear Analysis:"""
    },
    AnalysisMode.RISK: {
        "temperature": 0.3,
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "prompt": """You are a risk-focused legal analyst.
        Based on the following contract sections, identify potential risks, ambiguities,
        and areas that could lead to disputes or complications.
        
        Contract sections:
        {context}
        
        Question: {question}
        Risk Analysis:"""
    },
    AnalysisMode.CORNER_CASES: {
        "temperature": 0.7,
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "prompt": """You are a creative legal analyst exploring edge cases.
        Based on the following contract sections, explore unexpected scenarios,
        creative interpretations, and potential loopholes. Think outside the box
        while maintaining legal relevance.
        
        Contract sections:
        {context}
        
        Question: {question}
        Creative Analysis:"""
    },
    AnalysisMode.COMPARISON: {
        "temperature": 0.2,
        "chunk_size": 800,
        "chunk_overlap": 200,
        "prompt": """You are a comparative legal analyst.
        Compare and contrast different sections of these contracts,
        identifying patterns, inconsistencies, and notable differences.
        
        Contract sections:
        {context}
        
        Question: {question}
        Comparative Analysis:"""
    }
}

st.set_page_config(page_title="Legal Contract Analyzer", page_icon="⚖️", layout="wide")
st.title("Legal Contract Analyzer ⚖️")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using OCR if needed"""
    try:
        # Try normal PDF text extraction first
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # If no text was extracted, try OCR
        if not text.strip():
            images = convert_from_bytes(pdf_file.getvalue())
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def process_text(text, mode):
    """Process text based on analysis mode"""
    config = ANALYSIS_CONFIGS[mode]
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    # Create conversation chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=config["temperature"],
        convert_system_message_to_human=True
    )
    
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=config["prompt"]
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

# Sidebar
with st.sidebar:
    st.subheader("Document Upload")
    pdf_docs = st.file_uploader(
        "Upload your contracts (PDF)",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more contract PDFs"
    )
    
    st.subheader("Analysis Settings")
    mode = st.selectbox(
        "Analysis Mode",
        options=[mode.value for mode in AnalysisMode],
        help="""
        Basic Comprehension: Clear understanding of terms
        Risk Analysis: Identify potential issues
        Corner Case Exploration: Creative scenario analysis
        Contract Comparison: Compare multiple contracts
        """
    )
    
    if st.button("Process Documents", disabled=not pdf_docs):
        with st.spinner("Processing contracts..."):
            try:
                # Extract text from all documents
                all_text = ""
                for pdf in pdf_docs:
                    with st.spinner(f"Processing {pdf.name}..."):
                        text = extract_text_from_pdf(pdf)
                        all_text += f"\n\n=== Document: {pdf.name} ===\n{text}"
                
                # Store extracted text
                st.session_state.extracted_text = all_text
                
                # Process text with selected mode
                process_text(all_text, AnalysisMode(mode))
                
                st.session_state.processComplete = True
                st.success("Processing complete!")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

# Main chat interface
if st.session_state.processComplete:
    # Display current mode
    st.info(f"Current Analysis Mode: {mode}")
    
    # Text preview option
    if st.checkbox("Show extracted text"):
        st.text_area("Extracted Content", st.session_state.extracted_text,
                    height=200)
    
    # Chat interface
    user_question = st.chat_input(
        "Ask about the contracts (e.g., 'What are the key obligations?')")
    
    if user_question:
        try:
            with st.spinner("Analyzing..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Assistant", response["answer"]))
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.info("👈 Start by uploading contracts and selecting an analysis mode.")

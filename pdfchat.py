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

# Set Google API key from secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

# Initialize session state variables for maintaining conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF documents
    Args:
        pdf_docs: List of uploaded PDF files
    Returns:
        str: Concatenated text from all PDFs
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Split the text into smaller chunks for processing
    Args:
        text: Raw text to be split
    Returns:
        list: List of text chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    """
    Create a conversation chain using Google's Generative AI
    Args:
        vectorstore: FAISS vector store containing document embeddings
    Returns:
        ConversationalRetrievalChain: Chain for managing conversation
    """
    # Initialize Google's Generative AI model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Define the conversation template
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    # Create and return the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

def process_docs(pdf_docs):
    """
    Process uploaded PDF documents
    Args:
        pdf_docs: List of uploaded PDF files
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Extract text from PDFs
        raw_text = get_pdf_text(pdf_docs)
        
        # Split text into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Initialize Google's embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Create vector store using FAISS
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Initialize conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.session_state.processComplete = True
        
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)
            if success:
                st.success("Processing complete!")

# Main chat interface
if st.session_state.processComplete:
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

# Display initial instructions
else:
    st.write("👈 Upload your PDFs in the sidebar to get started!")

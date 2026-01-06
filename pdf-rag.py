# RAG ChatBot - Conversational PDF Question Answering System
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="RAG ChatBot - PDF Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state - MUST BE EARLY
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

@st.cache_resource
def load_and_process_pdf(pdf_path, chunk_size=1000, chunk_overlap=100, model_name="all-MiniLM-L12-v2", k_results=3):
    """Load and process PDF document"""
    try:
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def create_conversation_chain(vectorstore, k_results=3):
    """Create conversational retrieval chain"""
    try:
        model = "llama-3.1-8b-instant"
        groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name=model,
            temperature=0.1
        )
        
        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Enhanced prompt template
        custom_prompt_template = """You are an expert AI assistant specialized in analyzing and answering questions about PDF documents with precision and clarity.

CORE INSTRUCTIONS:
1. **Primary Source**: Answer EXCLUSIVELY using information from the provided context below. This is your ONLY source of truth.

2. **Answer Quality**:
   - For SPECIFIC questions (e.g., "What is a for loop?"): Provide detailed, complete answers with explanations and examples if available in context
   - For GENERAL questions (e.g., "What topics are covered?"): Synthesize information from context to give comprehensive overviews
   - For CODE examples: Extract and format code snippets clearly if present in context
   - For DEFINITIONS: Provide the exact definition from the document, then explain it clearly

3. **When Information is Available**:
   - Give thorough, well-structured answers
   - Use bullet points or numbered lists for clarity when appropriate
   - Include relevant examples, code snippets, or explanations from the context
   - Quote important phrases when necessary
   - Connect related concepts if they appear in the context

4. **When Information is NOT Available**:
   - Clearly state: "I cannot find information about [topic] in the provided document."
   - If partially answered, say what you found and what's missing
   - Never invent, assume, or use outside knowledge

5. **Follow-up Questions**:
   - Reference chat history to maintain conversation context
   - Build upon previous answers when relevant
   - If a follow-up asks "more about that", refer to the previous topic

6. **Formatting**:
   - Use markdown for better readability
   - Format code with ```python blocks
   - Use **bold** for key terms
   - Use bullet points for lists of items

CONTEXT FROM DOCUMENT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION:
{question}

ANSWER (based solely on the document above):"""
        
        # Create the conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=groq_chat,
            retriever=vectorstore.as_retriever(search_kwargs={'k': k_results}),
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=custom_prompt_template,
                    input_variables=["context", "question", "chat_history"]
                )
            },
            return_source_documents=False,
            verbose=False
        )
        
        st.session_state.conversation_chain = qa_chain
        return qa_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def clear_conversation():
    """Clear conversation history"""
    st.session_state.messages = []
    if st.session_state.conversation_chain:
        st.session_state.conversation_chain.memory.clear()

# Title and description
st.title("📚 RAG ChatBot - PDF Question Answering")
st.markdown("Ask questions about your PDF document. The bot answers exclusively from the document content.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # PDF upload or path input
    pdf_option = st.radio(
        "Choose PDF input method:",
        ["Use sample PDF", "Upload your own PDF", "Use file path"]
    )
    
    pdf_path = None
    if pdf_option == "Upload your own PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file:
            # Save uploaded file temporarily
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_path = "temp_uploaded.pdf"
    elif pdf_option == "Use file path":
        pdf_path = st.text_input(
            "Enter PDF file path:",
            value="test_pdf.pdf"
        )
    else:
        pdf_path = "test_pdf.pdf"  # Default sample PDF
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 50, 500, 100, 50)
        k_results = st.slider("Number of results to retrieve", 1, 5, 3, 1)
        model_name = st.selectbox(
            "Embedding model",
            ["all-MiniLM-L12-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"]
        )
    
    # Load/Reload button
    if st.button("📥 Load/Reload PDF", use_container_width=True):
        if pdf_path:
            with st.spinner("Processing PDF..."):
                try:
                    vectorstore = load_and_process_pdf(pdf_path, chunk_size, chunk_overlap, model_name, k_results)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        create_conversation_chain(vectorstore, k_results)
                        st.success(f"✅ PDF loaded successfully! ({pdf_path})")
                except Exception as e:
                    st.error(f"❌ Error loading PDF: {e}")
        else:
            st.warning("Please provide a PDF file first!")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_conversation()
        st.rerun()
    
    # Debug information
    with st.expander("📊 Debug Info"):
        if st.session_state.vectorstore:
            st.write("✅ Vector store loaded")
        else:
            st.write("❌ No vector store")
        
        if st.session_state.conversation_chain:
            st.write("✅ Conversation chain ready")
        else:
            st.write("❌ No conversation chain")
        
        st.write(f"Messages in history: {len(st.session_state.messages)}")

# Custom CSS for sticky input at bottom
st.markdown("""
<style>
    /* Make the chat input stick to bottom */
    .stChatFloatingInputContainer {
        position: fixed !important;
        bottom: 20px !important;
        background-color: white;
        z-index: 1000;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Add padding to chat container so messages don't hide under input */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Style the chat messages container */
    .stChatMessage {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main chat interface - Scrollable messages area
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input - Always at bottom
if prompt := st.chat_input("Ask a question about the PDF...", key="chat_input"):
    # Check if PDF is loaded
    if not st.session_state.conversation_chain:
        st.warning("⚠️ Please load a PDF first using the sidebar!")
        st.stop()
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation_chain({"question": prompt})
                answer = response["answer"]
                
                # Display response
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Rerun to show new messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>📚 RAG ChatBot v2.0 | Uses only document content | No external knowledge</p>
    </div>
    """,
    unsafe_allow_html=True
)
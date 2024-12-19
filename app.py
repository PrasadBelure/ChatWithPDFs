import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os
import gc
import torch

# Clear GPU memory if available
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Function to extract text from PDFs with memory management
def get_pdf_text(pdf_docs, chunk_size=10):
    all_text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for i in range(0, len(pdf_reader.pages), chunk_size):
                chunk_pages = pdf_reader.pages[i:i + chunk_size]
                for page in chunk_pages:
                    all_text += page.extract_text() + "\n"
                gc.collect()
        return all_text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to split text into smaller chunks
def get_text_chunks(raw_text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

# Function to create vector store with batching
def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        batch_size = 100
        vectorstore = None
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
            else:
                vectorstore.add_texts(batch)
            gc.collect()
            
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        clear_gpu_memory()
        return None

# Initialize conversation chain with memory management
def get_conversation_chain(vectorstore):
    try:
        llm = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv('GROQ_API_KEY'),
            max_tokens=512
        )
        
        # Configure memory with explicit output key
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',  # Specify which key to store in memory
            max_token_limit=2000
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            chain_type="stuff",
            combine_docs_chain_kwargs={'prompt': None},  # Use default prompt
            verbose=True  # Enable verbose mode for debugging
        )
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

# Handle user input with error management
def handle_userinput(user_input):
    if not st.session_state.conversation:
        st.error("Please upload and process documents first.")
        return
    
    try:
        # Create proper input dictionary
        response = st.session_state.conversation({
            'question': user_input
        })
        
        # Extract answer and source documents
        answer = response.get('answer', '')
        sources = response.get('source_documents', [])
        
        # Update chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.write(f"🙋‍♂️ **You:** {message['content']}", unsafe_allow_html=True)
            else:
                st.write(f"🤖 **Assistant:** {message['content']}", unsafe_allow_html=True)
                
            # Optional: Display sources
            if message["role"] == "assistant" and sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"Source {i+1}:", source.page_content[:200] + "...")
                
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        clear_gpu_memory()

def main():
    try:
        load_dotenv()
        
        if not os.getenv('GROQ_API_KEY'):
            st.error("Please set your GROQ_API_KEY in the .env file")
            return

        st.set_page_config(
            page_title="PDF Chat Assistant",
            page_icon="📚",
            layout="wide"
        )

        st.header("Chat with your PDFs 📚")
        
        # Initialize session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Sidebar
        with st.sidebar:
            st.subheader("📁 Document Upload")
            pdf_docs = st.file_uploader(
                "Upload PDFs (Max 10MB each)",
                accept_multiple_files=True,
                type=['pdf']
            )
            
            if st.button("Process Documents"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF.")
                    return
                
                # Check file sizes
                for pdf in pdf_docs:
                    if pdf.size > 10 * 1024 * 1024:  # 10MB limit
                        st.error(f"File {pdf.name} is too large. Please upload files under 10MB.")
                        return
                
                with st.spinner("Processing documents..."):
                    # Clear previous conversation
                    st.session_state.conversation = None
                    st.session_state.messages = []
                    
                    # Process documents
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            vectorstore = get_vectorstore(text_chunks)
                            if vectorstore:
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                st.success("Ready to chat! Ask questions below.")
                    
                    clear_gpu_memory()

        # Chat interface
        user_input = st.text_input("Ask a question about your documents:")
        if user_input:
            handle_userinput(user_input)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        clear_gpu_memory()

if __name__ == '__main__':
    main()
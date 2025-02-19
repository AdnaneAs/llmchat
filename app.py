import streamlit as st
import requests
import json
from datetime import datetime
import time
from rag_pipeline import RAGPipeline, is_valid_file
import os

# Set page configuration
st.set_page_config(page_title="Ollama Chat", page_icon="💭", layout="wide")

# Add custom CSS for hover delete button
st.markdown("""
<style>
.chat-container {
    position: relative;
    padding-right: 30px;
    margin-bottom: 5px;
}
.chat-container:hover .delete-btn {
    opacity: 1;
}
.delete-btn {
    position: absolute;
    right: 5px;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0;
    transition: opacity 0.2s;
    background: none;
    border: none;
    color: #ff4b4b;
    cursor: pointer;
    padding: 0 5px;
}
.delete-btn:hover {
    color: #ff0000;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache model list for 5 minutes
def get_ollama_models():
    """Fetch available Ollama models"""
    with st.spinner("🔍 Loading available models..."):
        try:
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                return models
            return []
        except Exception as e:
            st.error(f"Error connecting to Ollama: {str(e)}")
            return []

def chat_with_ollama(model, message, context=[]):
    """Send a message to Ollama and get the response"""
    try:
        payload = {
            "model": model,
            "messages": context + [{"role": "user", "content": message}]
        }
        
        if st.session_state.get('debug_mode', False):
            st.info("Request Payload:")
            st.json(payload)
        
        response = requests.post('http://localhost:11434/api/chat', json=payload, stream=True)
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        response_data = json.loads(line)
                        if 'message' in response_data:
                            content = response_data['message'].get('content', '')
                            full_response += content
                            # Yield partial response for streaming
                            yield content
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Error: Unable to get response from model (Status code: {response.status_code})"
    except Exception as e:
        yield f"Error: {str(e)}"

def generate_chat_title(first_message):
    """Generate a title from the first message"""
    max_length = 30
    title = first_message[:max_length]
    if len(first_message) > max_length:
        title += "..."
    return title

# Initialize RAG pipeline
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline(persist_dir="./chroma_db", debug=st.session_state.get('debug_mode', False))

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}  # Dictionary to store all chats
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'rag_mode' not in st.session_state:
    st.session_state.rag_mode = False
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = False

# Main UI
st.title("💭 Ollama Chat Interface")

# Sidebar for model selection and controls
with st.sidebar:
    st.header("Model Selection")
    models = get_ollama_models()
    if not models:
        st.error("No Ollama models found. Please make sure Ollama is running.")
        selected_model = None
    else:
        selected_model = st.selectbox("Choose a model:", models)
    
    # Mode toggles
    st.header("Mode Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    with col2:
        st.session_state.rag_mode = st.checkbox("RAG Mode", value=st.session_state.rag_mode)
    
    # RAG Settings
    if st.session_state.rag_mode:
        st.subheader("RAG Settings")
        upload_type = st.radio("Upload Type", ["Files", "Directory Path"])
        
        if upload_type == "Files":
            uploaded_files = st.file_uploader(
                "Upload Documents", 
                accept_multiple_files=True,
                type=['txt', 'pdf', 'docx', 'doc', 'md']
            )
            
            if uploaded_files and st.button("Index Documents"):
                with st.spinner("📚 Processing and indexing documents..."):
                    rag = get_rag_pipeline()
                    # Save uploaded files temporarily
                    temp_files = []
                    for file in uploaded_files:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                            
                        # Verify file content
                        if st.session_state.debug_mode:
                            with open(temp_path, "r", encoding='utf-8') as f:
                                content = f.read()
                                st.expander(f"📄 Content of {file.name}", expanded=False).text(
                                    f"First 500 characters:\n{content[:500]}..."
                                )
                        
                        temp_files.append(temp_path)
                    
                    try:
                        num_docs = rag.index_documents(files=temp_files)
                        st.session_state.documents_indexed = True
                        st.success(f"Successfully indexed {num_docs} chunks from {len(temp_files)} documents!")
                        
                        if st.session_state.debug_mode:
                            collection_info = rag.collection.count()
                            st.info(f"Total chunks in collection: {collection_info}")
                            
                    except Exception as e:
                        st.error(f"Error indexing documents: {str(e)}")
                    finally:
                        # Cleanup temp files
                        for temp_file in temp_files:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
        
        else:  # Directory Path
            dir_path = st.text_input("Enter Directory Path")
            if dir_path and st.button("Index Documents"):
                with st.spinner("📚 Processing and indexing documents from directory..."):
                    try:
                        rag = get_rag_pipeline()
                        num_docs = rag.index_documents(directory_path=dir_path)
                        st.session_state.documents_indexed = True
                        st.success(f"Successfully indexed {num_docs} documents!")
                    except Exception as e:
                        st.error(f"Error indexing documents: {str(e)}")
        
        if st.session_state.documents_indexed:
            if st.button("Clear Index"):
                with st.spinner("🗑️ Clearing document index..."):
                    try:
                        rag = get_rag_pipeline()
                        rag.clear_index()
                        st.session_state.documents_indexed = False
                        st.success("Index cleared successfully!")
                    except Exception as e:
                        st.error(f"Error clearing index: {str(e)}")
    
    # New chat button
    if st.button("New Chat"):
        chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.chats[chat_id] = {
            "title": "New Chat",
            "model": selected_model,
            "messages": []
        }
        st.session_state.current_chat_id = chat_id
        st.rerun()
    
    # Chat history
    st.header("Chat History")
    for chat_id, chat_data in st.session_state.chats.items():
        chat_title = f"{chat_data['title']} ({chat_data['model']})"
        
        # Create a container for each chat with delete button
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            if st.button(chat_title, key=f"chat_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat_id == chat_id:
                    st.session_state.current_chat_id = None
                st.rerun()

    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Select a model from the dropdown
    2. Type your message in the chat input
    3. Press Enter or click Send to chat
    
    Make sure Ollama is running locally!
    """)

# Chat interface
if selected_model:
    current_chat = st.session_state.chats.get(st.session_state.current_chat_id, {
        "title": "New Chat",
        "model": selected_model,
        "messages": []
    })
    
    # Display chat history
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        current_chat["messages"].append({"role": "user", "content": prompt})
        
        # Generate title for new chats
        if len(current_chat["messages"]) == 1:
            current_chat["title"] = generate_chat_title(prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            if st.session_state.rag_mode and st.session_state.documents_indexed:
                with st.spinner("🤔 Searching through documents..."):
                    try:
                        rag = get_rag_pipeline()
                        rag_response = rag.query(prompt)
                        
                        if st.session_state.debug_mode:
                            st.expander("🔍 Debug Information", expanded=True).json(rag_response)

                        if rag_response.get('total_results', 0) > 0:
                            # Create a more focused prompt based on retrieved context
                            enhanced_prompt = f"""You are a knowledgeable assistant analyzing documents. Use the provided context to answer questions accurately.

Context Information:
{rag_response['context']}

User Question: {prompt}

Instructions:
1. Base your answer primarily on the provided context
2. When referencing specific information, cite the source using [Passage X]
3. If the context doesn't fully answer the question, clearly state what information is missing
4. If you use multiple passages, explain how they relate to each other
5. Keep your answer focused and relevant to the question

Please provide your detailed answer:"""
                        else:
                            enhanced_prompt = f"""No relevant documents were found in the collection for the query: "{prompt}"
Please inform the user that you couldn't find relevant information in the indexed documents and suggest they might want to:
1. Try rephrasing their question
2. Check if the relevant documents have been properly indexed
3. Consider if the question is related to the content of the indexed documents"""

                        # Add a small info box showing the sources being used
                        if rag_response.get('sources', []):
                            with st.expander("📚 Sources Used", expanded=False):
                                st.write("Retrieved passages from:")
                                for source in rag_response['sources']:
                                    st.write(f"- {source['source']} (Relevance Score: {source['score']:.2f})")
                                if st.session_state.debug_mode:
                                    st.write("\nFull source details:", rag_response['sources'])
                    except Exception as e:
                        st.error(f"Error querying RAG pipeline: {str(e)}")
                        enhanced_prompt = prompt
            else:
                enhanced_prompt = prompt
            
            # Display streaming response
            with st.spinner("🤖 Thinking..."):
                for response_chunk in chat_with_ollama(
                    selected_model,
                    enhanced_prompt,
                    [{"role": m["role"], "content": m["content"]} for m in current_chat["messages"][:-1]]
                ):
                    full_response += response_chunk
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            current_chat["messages"].append({"role": "assistant", "content": full_response})
        
        # Update chat in session state
        if st.session_state.current_chat_id is None:
            chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.current_chat_id = chat_id
        st.session_state.chats[st.session_state.current_chat_id] = current_chat

else:
    st.warning("Please select a model from the sidebar to start chatting.")
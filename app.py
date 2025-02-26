import streamlit as st
import requests
import json
from datetime import datetime
import time
from rag_pipeline import RAGPipeline, is_valid_file
import os
from openai import OpenAI
import asyncio

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
def get_models():
    """Fetch available models (both Ollama and OpenAI)"""
    models = []
    # Add OpenAI models
    models.extend(['gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo'])
    
    # Add Ollama models
    with st.spinner("🔍 Loading available models..."):
        try:
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code == 200:
                ollama_models = [model['name'] for model in response.json()['models']]
                models.extend(ollama_models)
            return models
        except Exception as e:
            st.error(f"Error connecting to Ollama: {str(e)}")
            return models

def chat_with_openai(model, message, context=[]):
    """Send a message to OpenAI and get the response"""
    try:
        if not st.session_state.openai_api_key:
            yield "Please enter your OpenAI API key in the sidebar first."
            return

        # Reinitialize client with updated API key and organization
        client = OpenAI(
            api_key=st.session_state.openai_api_key,
            timeout=60.0  # Add timeout for better error handling
        )

        messages = context + [{"role": "user", "content": message}]
        
        if st.session_state.get('debug_mode', False):
            st.info("Request Payload:")
            st.json({"model": model, "messages": messages})
        
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error: {str(e)}"
        if st.session_state.get('debug_mode', False):
            st.error(f"OpenAI API Error Details: {str(e)}")

def is_openai_model(model):
    """Check if the selected model is an OpenAI model"""
    return model.startswith('gpt-')

# Initialize session state for embedding model
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "nomic-embed-text"

# Initialize session states for OpenAI
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

# Initialize RAG pipeline with selected embedding model
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline(
        persist_dir="./chroma_db", 
        debug=st.session_state.get('debug_mode', False),
        embedding_model=st.session_state.embedding_model
    )

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
    models = get_models()
    
    if not models:
        st.error("No models found. Please make sure Ollama is running for local models.")
        selected_model = None
    else:
        selected_model = st.selectbox("Choose a chat model:", models)
        
        # Add OpenAI API key input if OpenAI model is selected
        if selected_model and is_openai_model(selected_model):
            with st.expander("OpenAI Settings", expanded=True):
                st.session_state.openai_api_key = st.text_input(
                    "OpenAI API Key", 
                    value=st.session_state.openai_api_key,
                    type="password",
                    help="Enter your OpenAI API key to use GPT models"
                )
        
        # Add embedding model selection with cache clearing
        previous_embedding_model = st.session_state.embedding_model
        st.session_state.embedding_model = st.selectbox(
            "Choose an embedding model:",
            models,
            index=models.index("nomic-embed-text") if "nomic-embed-text" in models else 0
        )
        
        # Clear cache if embedding model changed
        if previous_embedding_model != st.session_state.embedding_model:
            get_rag_pipeline.clear()
            if st.session_state.debug_mode:
                st.sidebar.info(f"Switched embedding model to: {st.session_state.embedding_model}")
    
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
                    try:
                        for file in uploaded_files:
                            # Create a filename safe path
                            temp_path = os.path.join(os.getcwd(), f"temp_{file.name}")
                            
                            # Write the file in binary mode
                            with open(temp_path, "wb") as f:
                                f.write(file.getbuffer())
                            
                            if st.session_state.debug_mode:
                                st.info(f"Saved file: {temp_path}")
                            
                            temp_files.append(temp_path)
                        
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
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                                    if st.session_state.debug_mode:
                                        st.info(f"Cleaned up: {temp_file}")
                            except Exception as e:
                                if st.session_state.debug_mode:
                                    st.warning(f"Error cleaning up {temp_file}: {str(e)}")
        
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
                with st.spinner("Clearing document index..."):
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
            if st.button("X", key=f"delete_{chat_id}", help="Delete chat"):
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
                try:
                    rag = get_rag_pipeline()
                    rag_response = rag.query(prompt)
                    
                    # Debug information without nested expanders
                    if st.session_state.debug_mode:
                        st.write("🔍 Debug Information")
                        st.json(rag_response.get('debug_info', {}))
                        
                        if 'max_similarity' in rag_response.get('debug_info', {}):
                            st.write("Similarity Analysis")
                            max_sim = rag_response['debug_info']['max_similarity']
                            threshold = rag_response['debug_info']['threshold_used']
                            st.progress(max_sim, f"Max Similarity: {max_sim:.3f}")
                            st.progress(threshold, f"Threshold: {threshold:.3f}")

                    # Simple source display without expanders
                    if rag_response.get('sources', []):
                        st.write("📚 Retrieved Sources:")
                        for idx, source in enumerate(rag_response['sources'], 1):
                            st.write(f"[{idx}] {source['source']} (Score: {source.get('final_score', source['score']):.2f})")
                            if st.session_state.debug_mode:
                                st.text_area(f"Content {idx}", value=source['text'][:500], height=100)
                        
                        # Construct enhanced prompt with retrieved information
                        context_info = "\n Retrieved Information : \n\n"
                        for idx, source in enumerate(rag_response['sources'], 1):
                            context_info += f"[{idx}] Score: {source.get('final_score', source['score']):.2f}\n{source['text']}\n\n"
                        
                        enhanced_prompt = f"""
Question: {prompt}

Before answering, determine if the input is a general conversational phrase (e.g., greetings, thank-yous, acknowledgments). 
If it is, respond accordingly without searching external sources.

Otherwise, understand the question and decide if it requires external resources. 
If not, answer directly as an assistant. If the answer is in the provided information, quote it and cite the source using [X]. 
If the answer isn't in the provided information, say so.

context info : {context_info} 
"""
                    else:
                        enhanced_prompt = prompt
                        st.warning("No relevant information found in the indexed documents.")
                    
                except Exception as e:
                    st.error(f"RAG Error: {str(e)}")
                    enhanced_prompt = prompt
            else:
                enhanced_prompt = prompt

            # Display streaming response with enhanced prompt
            with st.spinner(" 🤖 Thinking..."):
                chat_function = chat_with_openai if is_openai_model(selected_model) else chat_with_ollama
                for response_chunk in chat_function(
                    selected_model,
                    enhanced_prompt,
                    current_chat["messages"][:-1]  # Previous messages without current prompt
                ):
                    full_response += response_chunk
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Store messages in chat history (only store original prompt)
            current_chat["messages"].append({"role": "assistant", "content": full_response})

        # Update chat in session state
        if st.session_state.current_chat_id is None:
            chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.current_chat_id = chat_id
        st.session_state.chats[st.session_state.current_chat_id] = current_chat

else:
    st.warning("Please select a model from the sidebar to start chatting.")
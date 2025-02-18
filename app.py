import streamlit as st
import requests
import json
from datetime import datetime
import time

# Set page configuration
st.set_page_config(page_title="Ollama Chat", page_icon="💭", layout="wide")

@st.cache_data(ttl=300)  # Cache model list for 5 minutes
def get_ollama_models():
    """Fetch available Ollama models"""
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

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}  # Dictionary to store all chats
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

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
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    
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
        if st.sidebar.button(chat_title, key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id
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
            
            # Display streaming response
            for response_chunk in chat_with_ollama(
                selected_model,
                prompt,
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
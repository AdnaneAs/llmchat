import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(page_title="Ollama Chat", page_icon="💭", layout="wide")

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
        
        # Debug: Show the request payload
        if st.session_state.get('debug_mode', False):
            st.info("Request Payload:")
            st.json(payload)
        
        response = requests.post('http://localhost:11434/api/chat', json=payload)
        
        if st.session_state.get('debug_mode', False):
            st.info(f"Response Status Code: {response.status_code}")
            st.info("Raw Response:")
            st.text(response.text)
        
        if response.status_code == 200:
            # Handle streaming response by splitting into lines and parsing each JSON object
            full_response = ""
            for line in response.text.strip().split('\n'):
                try:
                    response_data = json.loads(line)
                    if 'message' in response_data:
                        full_response += response_data['message'].get('content', '')
                except json.JSONDecodeError:
                    continue
            return full_response
        return f"Error: Unable to get response from model (Status code: {response.status_code})"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize session state for chat history and debug mode
if 'messages' not in st.session_state:
    st.session_state.messages = []
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
        st.session_state.messages = []
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
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = [{"role": m["role"], "content": m["content"]} 
                          for m in st.session_state.messages[:-1]]
                response = chat_with_ollama(selected_model, prompt, context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Please select a model from the sidebar to start chatting.")
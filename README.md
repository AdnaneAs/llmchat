# Ollama Chat Interface with RAG Support

A streamlined chat interface for Ollama models with Retrieval-Augmented Generation (RAG) capabilities.

## Features

-  Interactive chat interface with Ollama models
-  RAG (Retrieval-Augmented Generation) support for document-based Q&A
-  Support for multiple document types:
  - PDF files
  - Word documents (DOCX, DOC)
  - Text files (TXT)
  - Markdown files (MD)
-  Multiple embedding model selection for RAG
-  Persistent chat history
-  Advanced semantic search and reranking
-  Debug mode for detailed information
-  Document chunking and indexing
-  Index clearing functionality

## Setup

1. Make sure you have Python 3.8+ installed
2. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
3. Clone this repository
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start Ollama server
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Select your preferred chat and embedding models from the sidebar
4. Choose between regular chat mode or RAG mode

### RAG Mode Features

- Upload individual files or specify a directory path
- Automatic text extraction from various document formats
- Document chunking and embedding
- Semantic search with reranking
- Source citation in responses

## Configuration

The application includes several configurable options in the sidebar:

- Chat Model Selection
- Embedding Model Selection
- Debug Mode Toggle
- RAG Mode Toggle
- Document Upload Options
- Index Management


## Notes

- Ensure Ollama is running before starting the application
- For PDF files, make sure they contain extractable text
- Large documents are automatically chunked for better processing
- Multiple encoding support for text files
from langchain_ollama import OllamaEmbeddings
import chromadb
import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings
import numpy as np
from chromadb.config import Settings

class RAGPipeline:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = "document_collection"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        
        # Initialize Ollama embeddings
        self.embed_model = OllamaEmbeddings(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        return self.embed_model.embed_documents(texts)

    def index_documents(self, directory_path: str = None, files: List[str] = None) -> None:
        """Index documents from either a directory or list of files"""
        try:
            texts = []
            ids = []
            
            # Read documents
            if directory_path:
                for file in os.listdir(directory_path):
                    if is_valid_file(file):
                        with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
                            text = f.read()
                            texts.append(text)
                            ids.append(str(len(ids)))
            elif files:
                for file in files:
                    if is_valid_file(file):
                        with open(file, 'r', encoding='utf-8') as f:
                            text = f.read()
                            texts.append(text)
                            ids.append(str(len(ids)))
            else:
                raise ValueError("Either directory_path or files must be provided")

            # Get embeddings
            embeddings = self._get_embeddings(texts)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids
            )
            
            return len(texts)
        except Exception as e:
            raise Exception(f"Error during indexing: {str(e)}")

    def query(self, query_text: str, num_results: int = 3, threshold: float = 0.7) -> dict:
        """Query using similarity search"""
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query_text])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results,
                include=["documents", "distances"]
            )
            
            # Filter results based on threshold if needed
            documents = results['documents'][0]
            distances = results['distances'][0]
            
            # Convert distances to similarities (ChromaDB returns distances, we convert to similarities)
            similarities = [1 - dist for dist in distances]
            
            # Filter based on threshold
            filtered_results = [
                {"text": doc, "score": sim}
                for doc, sim in zip(documents, similarities)
                if sim >= threshold
            ]
            
            if not filtered_results:
                return {
                    "answer": "No relevant documents found.",
                    "sources": []
                }
            
            # For now, we'll just return the most relevant text as the answer
            return {
                "answer": filtered_results[0]["text"],
                "sources": filtered_results
            }
        except Exception as e:
            raise Exception(f"Error during query: {str(e)}")

    def clear_index(self) -> None:
        """Clear the vector store"""
        try:
            self.collection.delete(where=None)
        except Exception as e:
            raise Exception(f"Error clearing index: {str(e)}")

# Helper functions for file handling
def get_file_extension(file_name: str) -> str:
    """Get file extension from file name"""
    return os.path.splitext(file_name)[1].lower()

def is_valid_file(file_name: str) -> bool:
    """Check if file type is supported"""
    valid_extensions = ['.txt', '.pdf', '.docx', '.doc', '.md']
    return get_file_extension(file_name) in valid_extensions
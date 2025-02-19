from langchain_ollama import OllamaEmbeddings
import chromadb
import os
from typing import List, Optional, Dict
from langchain_core.embeddings import Embeddings
import numpy as np
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import json

class RAGPipeline:
    def __init__(self, persist_dir: str = "./chroma_db", debug: bool = False):
        self.persist_dir = persist_dir
        self.debug = debug
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = "document_collection"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for better retrieval
            chunk_overlap=100,
            length_function=len,
        )
        
        # Initialize Ollama embeddings with correct parameters
        self.embed_model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        embeddings = self.embed_model.embed_documents(texts)
        if self.debug:
            print("\n=== Embedding Debug Info ===")
            for text, emb in zip(texts, embeddings):
                print(f"\nText: {text[:100]}...")
                print(f"Embedding shape: {len(emb)}")
                print(f"Embedding sample: {emb[:5]}")
                print(f"Embedding stats: min={min(emb):.4f}, max={max(emb):.4f}, mean={np.mean(emb):.4f}")
            print("=========================\n")
        return embeddings

    def _chunk_text(self, text: str, source_info: Dict) -> List[Dict]:
        """Split text into chunks and preserve source information"""
        chunks = self.text_splitter.split_text(text)
        if self.debug:
            print(f"\nChunking text from {source_info['file']}:")
            print(f"Total chunks created: {len(chunks)}")
            print(f"Sample chunk: {chunks[0][:200]}...")
        return [{
            "chunk": chunk,
            "source": source_info,
            "chunk_id": hashlib.md5(chunk.encode()).hexdigest()
        } for chunk in chunks]

    def index_documents(self, directory_path: str = None, files: List[str] = None) -> None:
        """Index documents from either a directory or list of files"""
        try:
            all_chunks = []
            
            # Process documents
            if directory_path:
                for file in os.listdir(directory_path):
                    if is_valid_file(file):
                        file_path = os.path.join(directory_path, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            source_info = {"file": file, "path": file_path}
                            chunks = self._chunk_text(text, source_info)
                            all_chunks.extend(chunks)
            elif files:
                for file in files:
                    if is_valid_file(file):
                        with open(file, 'r', encoding='utf-8') as f:
                            text = f.read()
                            source_info = {"file": os.path.basename(file), "path": file}
                            chunks = self._chunk_text(text, source_info)
                            all_chunks.extend(chunks)
            else:
                raise ValueError("Either directory_path or files must be provided")

            if not all_chunks:
                return 0

            # Prepare data for ChromaDB
            texts = [chunk["chunk"] for chunk in all_chunks]
            metadatas = [chunk["source"] for chunk in all_chunks]
            ids = [chunk["chunk_id"] for chunk in all_chunks]
            
            if self.debug:
                print(f"\nIndexing Summary:")
                print(f"Total chunks to index: {len(texts)}")
                print(f"Sample text: {texts[0][:200]}...")
            
            # Get embeddings
            embeddings = self._get_embeddings(texts)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Verify indexing
            if self.debug:
                count = self.collection.count()
                print(f"\nTotal documents in collection after indexing: {count}")
            
            return len(texts)
        except Exception as e:
            raise Exception(f"Error during indexing: {str(e)}")

    def query(self, query_text: str, num_results: int = 5, threshold: float = 0.5) -> dict:  # Lowered threshold
        """Query using similarity search"""
        try:
            # Get query embedding
            if self.debug:
                print(f"\n=== Query Debug Info ===")
                print(f"Query text: {query_text}")
                print(f"Collection size: {self.collection.count()}")
            
            query_embedding = self._get_embeddings([query_text])[0]
            
            # Search in ChromaDB with more results initially
            initial_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, self.collection.count()),  # Get more results initially
                include=["documents", "distances", "metadatas"]
            )
            
            if self.debug:
                print("\nChromaDB Query Results:")
                print(json.dumps(initial_results, indent=2))
            
            documents = initial_results['documents'][0]
            distances = initial_results['distances'][0]
            metadatas = initial_results['metadatas'][0]
            
            # Convert distances to similarities
            similarities = [1 - dist for dist in distances]
            
            if self.debug:
                print("\nSimilarity Scores:")
                for doc, sim in zip(documents, similarities):
                    print(f"Score: {sim:.4f} - Text: {doc[:100]}...")
            
            # Filter and format results
            filtered_results = []
            for doc, sim, meta in zip(documents, similarities, metadatas):
                if sim >= threshold:
                    filtered_results.append({
                        "text": doc,
                        "score": sim,
                        "source": meta["file"],
                        "path": meta["path"]
                    })
            
            # Sort by similarity score
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top N results
            filtered_results = filtered_results[:num_results]
            
            if not filtered_results:
                if self.debug:
                    print(f"\nNo results above threshold {threshold}")
                return {
                    "answer": "No relevant documents found.",
                    "sources": [],
                    "context": "No relevant context found in the documents."
                }
            
            # Format context for LLM
            context = "Here are the most relevant passages I found:\n\n"
            for i, result in enumerate(filtered_results, 1):
                context += f"[{i}] From {result['source']} (Relevance: {result['score']:.2f}):\n{result['text']}\n\n"
            
            if self.debug:
                print("\nFinal Response:")
                print(f"Number of filtered results: {len(filtered_results)}")
                print(f"Context length: {len(context)}")
                print("=========================\n")
            
            return {
                "context": context,
                "sources": filtered_results
            }
        except Exception as e:
            if self.debug:
                print(f"\nError in query: {str(e)}")
                import traceback
                print(traceback.format_exc())
            raise Exception(f"Error during query: {str(e)}")

    def clear_index(self) -> None:
        """Clear the vector store"""
        try:
            self.collection.delete(where=None)
            if self.debug:
                print(f"Index cleared. Collection size: {self.collection.count()}")
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
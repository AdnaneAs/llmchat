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
import shutil

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

    def _semantic_rerank(self, query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank results using semantic similarity with the query"""
        if not results:
            return []
        
        # Get embeddings for query and passages
        query_embedding = self._get_embeddings([query])[0]
        passage_embeddings = self._get_embeddings([r["text"] for r in results])
        
        # Calculate semantic similarities
        for idx, (result, embedding) in enumerate(zip(results, passage_embeddings)):
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            results[idx]["semantic_score"] = float(similarity)
            # Combine original score with semantic score
            results[idx]["final_score"] = (results[idx]["score"] + results[idx]["semantic_score"]) / 2
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def query(self, query_text: str, num_results: int = 5, threshold: float = 0.1) -> dict:
        """Query using similarity search"""
        try:
            # Verify collection exists and has documents
            collection_count = self.collection.count()
            
            if self.debug:
                print(f"\n=== Query Debug Info ===")
                print(f"Query text: {query_text}")
                print(f"Collection size: {collection_count}")
            
            if collection_count == 0:
                if self.debug:
                    print("Collection is empty!")
                return {
                    "answer": "No documents found in the collection.",
                    "sources": [],
                    "context": "The document collection is empty.",
                    "debug_info": {"collection_count": 0}
                }
            
            # Get query embedding
            query_embedding = self._get_embeddings([query_text])[0]
            
            # Search in ChromaDB with more results initially
            initial_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(20, collection_count),  # Get more initial results
                include=["documents", "distances", "metadatas"]
            )
            
            if self.debug:
                print("\nChromaDB Raw Query Results:")
                print(json.dumps({
                    "documents_count": len(initial_results['documents'][0]),
                    "distances": initial_results['distances'][0],
                    "metadatas": initial_results['metadatas'][0]
                }, indent=2))
            
            if not initial_results['documents'][0]:
                if self.debug:
                    print("No documents returned from query")
                return {
                    "answer": "No documents found in collection query.",
                    "sources": [],
                    "context": "Query returned no results.",
                    "debug_info": {
                        "collection_count": collection_count,
                        "query_embedding_sample": query_embedding[:5]
                    }
                }
            
            documents = initial_results['documents'][0]
            distances = initial_results['distances'][0]
            metadatas = initial_results['metadatas'][0]
            
            # Convert distances to similarities
            similarities = [1 - dist for dist in distances]
            
            if self.debug:
                print("\nSimilarity Scores Detail:")
                for doc, sim, meta in zip(documents, similarities, metadatas):
                    print(f"\nScore: {sim:.4f} - From {meta['file']}:")
                    print(f"Distance: {1-sim:.4f}")
                    print(f"Text Preview: {doc[:100]}...")
            
            # Filter and format results with lower threshold
            filtered_results = [
                {
                    "text": doc,
                    "score": sim,
                    "source": meta["file"],
                    "path": meta["path"]
                }
                for doc, sim, meta in zip(documents, similarities, metadatas)
                if sim >= threshold
            ]
            
            # Sort by similarity score
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            filtered_results = filtered_results[:num_results]
            
            if not filtered_results:
                if self.debug:
                    print(f"\nNo results above threshold {threshold}")
                    print("Highest similarity score:", max(similarities) if similarities else "No similarities")
                return {
                    "answer": "No sufficiently relevant documents found.",
                    "sources": [],
                    "context": "While documents exist in the collection, none were relevant enough to the query.",
                    "debug_info": {
                        "max_similarity": max(similarities) if similarities else None,
                        "threshold": threshold,
                        "total_results_before_filter": len(documents)
                    }
                }
            
            # Apply semantic reranking to filtered results
            if filtered_results:
                filtered_results = self._semantic_rerank(query_text, filtered_results)
                if self.debug:
                    print("\nReranking Results:")
                    for result in filtered_results:
                        print(f"Document: {result['source']}")
                        print(f"Original Score: {result['score']:.4f}")
                        print(f"Semantic Score: {result['semantic_score']:.4f}")
                        print(f"Final Score: {result['final_score']:.4f}")
                        print(f"Preview: {result['text'][:100]}...")
                        print("---")
            
            # Format context for LLM
            context = "Here are the relevant passages from your documents:\n\n"
            for i, result in enumerate(filtered_results, 1):
                score = result.get("final_score", result["score"])  # Fallback to original score if final_score not set
                context += (
                    f"[Passage {i}] From '{result['source']}' "
                    f"(Relevance: {score:.2f}):\n"
                    f"{result['text'].strip()}\n\n"
                )
            
            if self.debug:
                print("\nFinal Response Summary:")
                print(f"Number of reranked results: {len(filtered_results)}")
                print(f"Context length: {len(context)}")
                if filtered_results:
                    print("Top result score:", filtered_results[0]["final_score"])
            
            return {
                "context": context,
                "sources": filtered_results,
                "total_results": len(filtered_results),
                "debug_info": {
                    "collection_size": collection_count,
                    "initial_results": len(documents),
                    "filtered_results": len(filtered_results),
                    "threshold_used": threshold,
                    "max_similarity": max(result["final_score"] for result in filtered_results) if filtered_results else None
                }
            }
            
        except Exception as e:
            if self.debug:
                print(f"\nError in query: {str(e)}")
                import traceback
                print(traceback.format_exc())
            raise Exception(f"Error during query: {str(e)}")

    def clear_index(self) -> None:
        """Clear the vector store and reinitialize the collection"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)
            
            # Delete the persist directory contents
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir, exist_ok=True)
            
            # Reinitialize the collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            if self.debug:
                print(f"Index cleared and reinitialized. Collection size: {self.collection.count()}")
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
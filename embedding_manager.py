"""
Embedding Manager for RAG System
Handles creation and management of embeddings for document chunks using LangChain.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import numpy as np

class EmbeddingManager:
    """
    Manages the creation and storage of embeddings for document chunks.
    """
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 persist_directory: str = "chroma_db",
                 collection_name: str = "document_chunks"):
        """
        Initialize the EmbeddingManager.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection in the vector database
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding function
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"Vector store initialized: {self.persist_directory}")
            
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            raise
    
    def create_embeddings_from_chunks(self, 
                                    chunks: List[str], 
                                    metadata: List[Dict[str, Any]]) -> bool:
        """
        Create embeddings from chunks and store them in the vector database.
        
        Args:
            chunks: List of text chunks
            metadata: List of metadata dictionaries for each chunk
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Creating embeddings for {len(chunks)} chunks...")
            
            # Convert chunks and metadata to Document objects
            documents = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                # Create a copy of metadata and add chunk_id
                doc_metadata = meta.copy()
                doc_metadata['chunk_id'] = i
                doc_metadata['embedding_model'] = self.embedding_model
                
                # Create Document object
                doc = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                documents.append(doc)
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Persist the database
            self.vector_store.persist()
            
            print(f"✅ Successfully created embeddings for {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            return False
    
    def search_similar_chunks(self, 
                            query: str, 
                            k: int = 5, 
                            similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on a query.
        
        Args:
            query: Search query
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of dictionaries containing chunk content and metadata
        """
        try:
            # Search for similar documents
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=k
            )
            
            # Filter by similarity threshold and format results
            filtered_results = []
            seen_contents = set()
            
            for doc, score in results:
                if score >= similarity_threshold:
                    # Check for duplicate content
                    content_preview = doc.page_content[:100]  # First 100 chars for deduplication
                    
                    if content_preview not in seen_contents:
                        seen_contents.add(content_preview)
                        
                        result = {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'similarity_score': score
                        }
                        filtered_results.append(result)
                    else:
                        print(f"⚠️  Skipping duplicate chunk: {content_preview}...")
            
            print(f"Found {len(filtered_results)} unique similar chunks for query: '{query}'")
            return filtered_results
            
        except Exception as e:
            print(f"❌ Error searching chunks: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            stats = {
                'total_chunks': count,
                'embedding_model': self.embedding_model,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {str(e)}")
            return {}
    
    def clear_database(self) -> bool:
        """
        Clear all embeddings from the database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.vector_store._collection.delete()
            
            # Reinitialize the vector store
            self._initialize_vector_store()
            
            print("✅ Database cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing database: {str(e)}")
            return False
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension or None if error
        """
        try:
            # Create a test embedding to get dimension
            test_text = "This is a test sentence."
            test_embedding = self.embeddings.embed_query(test_text)
            return len(test_embedding)
            
        except Exception as e:
            print(f"❌ Error getting embedding dimension: {str(e)}")
            return None


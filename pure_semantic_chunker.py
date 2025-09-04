from helper_utils import word_wrap
import os
import numpy as np

# Import semantic chunking components
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Import helper functions
from rag_helper import (
    load_and_preprocess_documents,
    save_chunks_to_files,
    analyze_chunk_quality
)

def create_semantic_chunker(embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"Initializing semantic chunker with {embedding_model}...")
    
    # Initialize embedding function
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create semantic chunker with configurable parameters
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # or "interquartile"
        breakpoint_threshold_amount=0.1,  # 10th percentile for breakpoint detection
        min_chunk_size=1000,  # Minimum chunk size in characters
    )
    
    return semantic_chunker

def apply_semantic_chunking(texts, chunker):
    """
    Apply semantic chunking to a list of texts.
    """
    print("Applying semantic chunking...")
    
    all_chunks = []
    chunk_metadata = []
    
    for i, text in enumerate(texts):
        try:
            # Apply semantic chunking
            chunks = chunker.split_text(text)
            
            # Store chunks with metadata
            for j, chunk in enumerate(chunks):
                # Handle both string and object chunks
                chunk_content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
                all_chunks.append(chunk_content)
                chunk_metadata.append({
                    'source_text': i,
                    'chunk_index': j,
                    'start_index': None,  # SemanticChunker doesn't provide start_index
                    'end_index': None,    # SemanticChunker doesn't provide end_index
                    'chunk_size': len(chunk_content)
                })
                
        except Exception as e:
            print(f"Error processing text {i+1}: {str(e)}")
            # Fallback to simple character splitting
            fallback_chunks = text.split('\n\n')
            for j, chunk in enumerate(fallback_chunks):
                if chunk.strip():
                    all_chunks.append(chunk.strip())
                    chunk_metadata.append({
                        'source_text': i,
                        'chunk_index': j,
                        'start_index': None,
                        'end_index': None,
                        'chunk_size': len(chunk.strip())
                    })
    
    print(f"Created {len(all_chunks)} semantic chunks")
    return all_chunks, chunk_metadata

def main():
    print("=== Pure Semantic Chunking Implementation ===")
    
    # Load documents
    txt_texts = load_and_preprocess_documents()
    
    # Create semantic chunker
    semantic_chunker = create_semantic_chunker()
    
    # Apply semantic chunking
    semantic_chunks, semantic_metadata = apply_semantic_chunking(txt_texts, semantic_chunker)
    
    # Analyze results
    analyze_chunk_quality(semantic_chunks, semantic_metadata, "Pure Semantic Chunking")
    
    # Save results to files
    summary_file, chunks_dir = save_chunks_to_files(
        semantic_chunks, 
        semantic_metadata, 
        chunking_type="Pure Semantic Chunking"
    )
    
    print(f"Semantic chunks created: {len(semantic_chunks)}")
    print(f"Results saved to: {chunks_dir}")
    
    # Return results for further processing
    return {
        'semantic_chunks': semantic_chunks,
        'semantic_metadata': semantic_metadata,
        'files_created': {
            'summary_file': summary_file,
            'chunks_dir': chunks_dir
        }
    }

if __name__ == "__main__":
    chunking_results = main()

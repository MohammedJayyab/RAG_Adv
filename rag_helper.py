import os
import numpy as np
from helper_utils import word_wrap

def save_chunks_to_files(chunks, metadata, output_dir="chunking_results", chunking_type="Unknown"):
    """
    Save chunks to separate text files with metadata.
    
    Args:
        chunks: List of text chunks
        metadata: List of metadata dictionaries for each chunk
        output_dir: Directory to save results
        chunking_type: Type of chunking method used
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary file
    summary_file = os.path.join(output_dir, f"{chunking_type.lower().replace(' ', '_')}_summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {chunking_type} Summary ===\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Average chunk size: {np.mean([meta['chunk_size'] for meta in metadata]):.1f} characters\n")
        f.write(f"Min chunk size: {min([meta['chunk_size'] for meta in metadata])} characters\n")
        f.write(f"Max chunk size: {max([meta['chunk_size'] for meta in metadata])} characters\n\n")
        
        # Add chunking method distribution if available
        if 'chunking_method' in metadata[0]:
            methods = {}
            for meta in metadata:
                method = meta['chunking_method']
                methods[method] = methods.get(method, 0) + 1
            
            f.write("Chunking method distribution:\n")
            for method, count in methods.items():
                f.write(f"  {method}: {count} chunks\n")
            f.write("\n")
        
        f.write("=" * 50 + "\n\n")
        
        # Write each chunk with metadata
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Size: {meta['chunk_size']} characters\n")
            
            # Add source text info if available
            if 'source_text' in meta:
                f.write(f"Source text: {meta['source_text']}\n")
            if 'chunk_index' in meta:
                f.write(f"Chunk index: {meta['chunk_index']}\n")
            
            # Add method info if available
            if 'chunking_method' in meta:
                f.write(f"Method: {meta['chunking_method']}\n")
            
            # Add hybrid-specific info if available
            if 'original_chunk' in meta:
                f.write(f"Original chunk: {meta['original_chunk']}\n")
            if 'refined_index' in meta:
                f.write(f"Refined index: {meta['refined_index']}\n")
            
            f.write("\nContent:\n")
            f.write(chunk)
            f.write("\n\n" + "=" * 50 + "\n\n")
    
    print(f"Saved {chunking_type} summary to: {summary_file}")
    
    # Create individual chunk files
    chunks_dir = os.path.join(output_dir, f"{chunking_type.lower().replace(' ', '_')}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        chunk_file = os.path.join(chunks_dir, f"chunk_{i+1:03d}.txt")
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(f"Chunk {i+1}\n")
            f.write(f"Size: {meta['chunk_size']} characters\n")
            
            # Add source info if available
            if 'source_text' in meta:
                f.write(f"Source: {meta['source_text']}\n")
            
            # Add method info if available
            if 'chunking_method' in meta:
                f.write(f"Method: {meta['chunking_method']}\n")
            else:
                f.write(f"Method: {chunking_type}\n")
            
            # Add hybrid-specific info if available
            if 'original_chunk' in meta:
                f.write(f"Original chunk: {meta['original_chunk']}\n")
            if 'refined_index' in meta:
                f.write(f"Refined index: {meta['refined_index']}\n")
            
            f.write("-" * 30 + "\n\n")
            f.write(chunk)
    
    print(f"Saved {len(chunks)} individual chunk files to: {chunks_dir}")
    
    return summary_file, chunks_dir

def save_comparison_analysis(semantic_chunks, semantic_metadata, hybrid_chunks, hybrid_metadata, output_dir="chunking_results"):
    """
    Create a comparison analysis file between different chunking methods.
    
    Args:
        semantic_chunks: List of semantic chunks
        semantic_metadata: Metadata for semantic chunks
        hybrid_chunks: List of hybrid chunks
        hybrid_metadata: Metadata for hybrid chunks
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_file = os.path.join(output_dir, "comparison_analysis.txt")
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("=== Chunking Methods Comparison ===\n\n")
        
        f.write("Pure Semantic Chunking:\n")
        f.write(f"  Total chunks: {len(semantic_chunks)}\n")
        f.write(f"  Average size: {np.mean([meta['chunk_size'] for meta in semantic_metadata]):.1f} chars\n")
        f.write(f"  Summary file: pure_semantic_chunking_summary.txt\n")
        f.write(f"  Chunks directory: pure_semantic_chunking_chunks\n\n")
        
        f.write("Hybrid Chunking:\n")
        f.write(f"  Total chunks: {len(hybrid_chunks)}\n")
        f.write(f"  Average size: {np.mean([meta['chunk_size'] for meta in hybrid_metadata]):.1f} chars\n")
        f.write(f"  Summary file: hybrid_chunking_summary.txt\n")
        f.write(f"  Chunks directory: hybrid_chunking_chunks\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("Files created:\n")
        f.write("1. pure_semantic_chunking_summary.txt - Pure semantic chunking summary\n")
        f.write("2. hybrid_chunking_summary.txt - Hybrid chunking summary\n")
        f.write("3. comparison_analysis.txt - This comparison file\n")
        f.write("4. pure_semantic_chunking_chunks/ - Individual semantic chunks\n")
        f.write("5. hybrid_chunking_chunks/ - Individual hybrid chunks\n")
    
    print(f"Saved comparison analysis to: {comparison_file}")
    return comparison_file

def analyze_chunk_quality(chunks, metadata, chunking_type="Unknown"):
    """
    Analyze the quality and characteristics of generated chunks.
    
    Args:
        chunks: List of text chunks
        metadata: List of metadata dictionaries
        chunking_type: Type of chunking method used
    """
    print(f"\n=== Chunk Quality Analysis for {chunking_type} ===")
    
    # Size distribution
    sizes = [meta['chunk_size'] for meta in metadata]
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {np.mean(sizes):.1f} characters")
    print(f"Min chunk size: {min(sizes)} characters")
    print(f"Max chunk size: {max(sizes)} characters")
    
    # Chunking method distribution (only if available)
    if 'chunking_method' in metadata[0]:
        methods = {}
        for meta in metadata:
            method = meta['chunking_method']
            methods[method] = methods.get(method, 0) + 1
        
        print("\nChunking method distribution:")
        for method, count in methods.items():
            print(f"  {method}: {count} chunks")
    
    # Show sample chunks
    #print("\nSample chunks:")
    #for i in range(min(3, len(chunks))):
    #    print(f"\nChunk {i+1} ({metadata[i]['chunk_size']} chars):")
    #    print(word_wrap(chunks[i][:200]) + "..." if len(chunks[i]) > 200 else word_wrap(chunks[i]))

def create_chunk_metadata(chunk_content, source_text, chunk_index, chunk_size, **kwargs):
    """
    Create standardized metadata for a chunk.
    
    Args:
        chunk_content: The text content of the chunk
        source_text: Source text identifier
        chunk_index: Index of the chunk within the source
        chunk_size: Size of the chunk in characters
        **kwargs: Additional metadata fields
    
    Returns:
        Dictionary containing chunk metadata
    """
    metadata = {
        'source_text': source_text,
        'chunk_index': chunk_index,
        'chunk_size': chunk_size,
        'start_index': None,
        'end_index': None
    }
    
    # Add any additional metadata fields
    metadata.update(kwargs)
    
    return metadata

def load_and_preprocess_documents(txt_path="data/knowledge.txt"):
    """
    Load text documents and extract content.
    
    Args:
        txt_path: Path to the text file
        
    Returns:
        List of text sections
    """
    print("Loading text documents...")
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into paragraphs or sections based on double newlines
        texts = [section.strip() for section in content.split('\n\n') if section.strip()]
        
        print(f"Loaded {len(texts)} non-empty sections from {txt_path}")
        return texts
        
    except FileNotFoundError:
        print(f"Error: File {txt_path} not found")
        return []
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []

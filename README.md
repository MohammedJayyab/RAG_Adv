# Advanced RAG Techniques - Semantic Chunking Pipeline

## Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline that leverages pure semantic chunking to provide accurate and contextually relevant answers. The system processes source documents, divides them into meaningful semantic chunks, generates embeddings, and stores them in a ChromaDB vector database. When a user asks a question, the pipeline retrieves the most relevant chunks and uses a Large Language Model (LLM) to generate a comprehensive answer.

A key feature of this pipeline is the generation of similar questions to enhance the search query, leading to more accurate and diverse retrieval results.

## Features

- **Semantic Chunking**: Instead of fixed-size chunks, the project uses a pure semantic chunking mechanism to divide documents based on meaning and context.
- **Vector Embeddings**: Generates and stores embeddings for each semantic chunk.
- **ChromaDB Integration**: Uses ChromaDB as the vector store for efficient similarity searches.
- **Interactive Search Mode**: Provides a command-line interface for users to ask questions and get answers.
- **Similar Question Generation**: Augments user queries by generating similar questions to improve retrieval accuracy.
- **LLM-Powered Responses**: Utilizes a Large Language Model to generate final answers based on the retrieved context.

## How It Works

The pipeline follows these steps:

1.  **Database Check**: On startup, the application checks if a ChromaDB database already exists.
2.  **Indexing Pipeline (if no database is found)**:
    a. **Semantic Chunking**: The source documents in the `data/` directory are processed by the `pure_semantic_chunker.py` script.
    b. **Embedding Generation**: The `embedding_manager.py` takes the semantic chunks and creates vector embeddings for each one.
    c. **Database Storage**: The embeddings and their corresponding metadata are stored in a ChromaDB collection.
3.  **Search & Answer Pipeline**:
    a. The application enters an interactive search mode.
    b. The user inputs a question.
    c. The `llm_manager.py` generates several similar questions based on the user's input.
    d. The original question and the generated questions are combined into an enhanced query.
    e. The `embedding_manager.py` performs a similarity search in ChromaDB to find the most relevant chunks.
    f. The retrieved chunks are presented to the user, along with their content and similarity scores.
    g. The `llm_manager.py` uses the original question and the retrieved chunks to generate a final, context-aware answer.

## Project Structure

```
.
├── chroma_db/              # ChromaDB vector store
├── chunking_results/       # Output of the chunking process
├── data/                   # Source documents (e.g., knowledge.pdf)
├── .gitignore
├── embedding_manager.py    # Manages embeddings and ChromaDB
├── helper_utils.py         # Utility functions
├── llm_manager.py          # Handles interactions with the LLM
├── main.py                 # Main entry point of the application
├── pure_semantic_chunker.py # Implements the semantic chunking logic
├── rag_helper.py           # Helper functions for the RAG pipeline
└── requirements.txt        # Python dependencies
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd advanced-rag-techniques
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Add your data:**
    Place your source documents (e.g., `knowledge.pdf`) inside the `data/` directory.

## Usage

To run the pipeline, execute the `main.py` script:

```bash
python main.py
```

- If no database is found, you will be prompted to create one. Type `y` to start the indexing process.
- Once the database is ready, the application will enter the "Interactive Search Mode".
- Type your question and press Enter.
- To exit the application, type `quit` or `q`.

#!/usr/bin/env python3

import os
from llm_manager import LLMManager
from pure_semantic_chunker import main as run_semantic_chunker
from embedding_manager import EmbeddingManager

def check_database_exists():
    db_path = "chroma_db/chroma.sqlite3"
    return os.path.exists(db_path)

def main():
    print("🚀 Advanced RAG Techniques - Semantic Chunking Pipeline")
    
    if check_database_exists():
        print("✅ Using existing database. Starting search mode...")
        embedding_manager = EmbeddingManager()
        return run_search_mode(embedding_manager)
    
    user_choice = input("❓ Database not found. Create new database? (y/N): ").strip().lower()
    if user_choice not in ['y', 'yes']:
        print("👋 Goodbye!")
        return None
    
    try:
        chunking_results = run_semantic_chunker()
        if not chunking_results or 'semantic_chunks' not in chunking_results:
            print("❌ Error: Semantic chunking failed")
            return None
        
        semantic_chunks = chunking_results['semantic_chunks']
        semantic_metadata = chunking_results['semantic_metadata']
        
        embedding_manager = EmbeddingManager()
        success = embedding_manager.create_embeddings_from_chunks(semantic_chunks, semantic_metadata)
        
        if not success:
            print("❌ Error: Failed to create embeddings")
            return None
        
        print(f"✅ Pipeline completed: {len(semantic_chunks)} chunks created")
        return run_search_mode(embedding_manager)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def run_search_mode(embedding_manager):
    print("\n🔍 Interactive Search Mode")
    print("-" * 30)
    
    while True:
        user_question = input("\n❓ Question (or 'quit'): ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            continue
        

         #generate similar questions
        llm_manager = LLMManager()
        similar_questions = llm_manager.generate_similar_question(user_question)
        print(f"\nSimilar Questions: {similar_questions}")
        # join similar questions with user question
        user_question_with_similar_questions = user_question + "\n" + "\n".join(similar_questions)

        #search for similar chunks
        print(f"\n🔍 Searching: '{user_question}'")
        search_results = embedding_manager.search_similar_chunks(
            query=user_question_with_similar_questions,
            k=5,
            similarity_threshold=0.9
        )
       
        if search_results:
            print(f"\n📋 Found {len(search_results)} chunks:")
            for i, result in enumerate(search_results, 1):
                print(f"\n--- Chunk {i} ---")
                print(f"📄 Content: {result['content'][:1000]}...")
                print(f"📊 Similarity: {result['similarity_score']:.3f}")
                print(f"🏷️  Source: {result['metadata'].get('source_text', 'Unknown')}")
                print(f"🔢 Size: {result['metadata'].get('chunk_size', 'Unknown')} chars")
           
            print("LLM now answering, please wait...")
            # Generate response using LLMManager
            llm_manager = LLMManager()
            ai_response = llm_manager.generate_response(user_question, search_results)
            print("============================================================")
            
            print(f"\nUser Question: {user_question}")
            print(f"\n(AI Answer): {ai_response}")
            print("============================================================")
            ##print(f"\n🤖 AI Response: {ai_response}")
        else:
            print("❌ No relevant chunks found.")
            

    
    return True

if __name__ == "__main__":
    results = main()
    if not results:
        exit(1)

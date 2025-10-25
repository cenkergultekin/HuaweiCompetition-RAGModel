"""
rag_query.py
FAISS vektör indeksi üzerinden RAG sorguları yapar.
Modüler yapı ile yeniden düzenlenmiştir.
"""

# Import modular components
from config import API_KEY, API_BASE, MODEL_NAME, EMBEDDING_MODEL, INDEX_PATH, TOP_K, TEMPERATURE, MAX_HISTORY
from vectorstore import load_vectorstore
from llm_utils import initialize_llm
from rag_engine import query_rag_system
from diagram_handler import handle_diagram_query
from chat_history import ChatHistory


def main():
    """Ana fonksiyon - RAG query loop"""
    print("HUAWEI CLOUD RAG - Q&A SYSTEM")
    
    # 1. Vektör deposunu yükle
    vectorstore = load_vectorstore(INDEX_PATH, EMBEDDING_MODEL)
    
    # 2. LLM'i başlat
    llm = initialize_llm(API_KEY, API_BASE, MODEL_NAME, TEMPERATURE)
    
    # 3. Chat history'yi başlat
    chat_history = ChatHistory(llm, MAX_HISTORY)

    print("="*60)
    print("System ready! You can start asking questions.")
    print("   To exit: type 'quit', 'exit', or 'q'")
    print("="*60)
    
    # 4. Soru-cevap döngüsü
    query_count = 0
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            # Çıkış kontrolü
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\nTotal {query_count} questions asked. Goodbye!")
                break
            
            # Boş sorgu kontrolü
            if not query:
                print("Please enter a question!")
                continue
            
            # Diagram etiketi kontrolü
            if query.startswith("@diagram"):
                handle_diagram_query(query, vectorstore, llm, TOP_K)
                continue  # Diagram tamamlandı, normal RAG'a gitme
            
            # Normal RAG sorgusu
            query_count += 1
            query_rag_system(vectorstore, llm, query, TOP_K, chat_history)
            
        except KeyboardInterrupt:
            print(f"\n\nShutting down... (Total {query_count} questions)")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
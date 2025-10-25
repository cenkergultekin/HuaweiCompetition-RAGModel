"""
vectorstore.py
FAISS vector store operations and management.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_vectorstore(index_path: str, embedding_model_name: str) -> FAISS:
    """FAISS vektör deposunu yükler."""
    print("\n" + "="*60)
    print("LOADING FAISS INDEX...")
    print("="*60)
    
    try:
        # Embedding modelini yükle
        print(f"Embedding model: {embedding_model_name}")
        emb_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} # cosine similarity için / vector buyuklugunden dolayi yanliligi yok eder, daha kararli sonuclar verir.
        )
        
        # FAISS indeksini yükle
        print(f"Index path: {index_path}")
        vectorstore = FAISS.load_local(
            index_path, 
            emb_model, 
            allow_dangerous_deserialization=True # pkl dosyası icin guvenlik engelini kapatiyor.
        )
        
        print(f"FAISS index loaded successfully!") 
        print(f"   Total vectors: {vectorstore.index.ntotal}")
        print(f"   Vector dimension: {vectorstore.index.d}")
        print("="*60 + "\n")
        
        return vectorstore
        
    except Exception as e:
        print(f"ERROR: Failed to load FAISS index!")
        print(f"   Details: {e}")
        print("\nPlease run 'embed_builder.py' first to create the index.\n")
        exit(1)


def display_sources(docs: list, show_content: bool = False):
    """Kaynak dokümanları formatlanmış şekilde gösterir."""
    print("\n" + "="*60)
    print("SOURCES:")
    print("="*60)
    
    seen_sources = set()
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        
        # Aynı kaynağı tekrar gösterme
        source_key = f"{source}_p{page}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        
        # Kaynak bilgisi
        source_name = os.path.basename(source) if source != 'Unknown' else 'Unknown'
        print(f"\n{i}. {source_name}")
        print(f"   Page: {page}")
        
        # İsteğe bağlı: İçerik önizlemesi
        if show_content:
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"   Preview: {preview}...")
    
    print("\n" + "="*60)

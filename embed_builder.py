"""
embed_builder.py
PDF dosyalarından FAISS vektör indeksi oluşturur.
"""

import os
import glob
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===============================
# CONFIGURATION
# ===============================
PDF_FOLDER = "./docs"
EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_PATH = "embeddings/faiss_index"

# Chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ".", " ", ""]

# Batch size for embedding (daha küçük yaparsanız daha sık güncelleme görürsünüz)
BATCH_SIZE = 16  # 32'den 16'ya düşürdüm, daha sık progress görülsün


def load_pdfs(folder_path: str) -> list:
    """Klasördeki tüm PDF dosyalarını yükler."""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"'{folder_path}' klasöründe PDF dosyası bulunamadı!")
    
    print(f"\n{'='*60}")
    print(f"📁 {len(pdf_files)} PDF dosyası bulundu:")
    for pdf in pdf_files:
        print(f"   • {os.path.basename(pdf)}")
    print(f"{'='*60}\n")
    
    all_texts = []
    failed_files = []
    
    for file in tqdm(pdf_files, desc="📄 PDF'ler yükleniyor", unit="dosya"):
        try:
            loader = PyPDFLoader(file)
            docs = loader.load()
            all_texts.extend(docs)
        except Exception as e:
            failed_files.append((os.path.basename(file), str(e)))
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} dosya yüklenemedi:")
        for filename, error in failed_files:
            print(f"   ✗ {filename}: {error}")
    
    print(f"\n✅ Toplam {len(all_texts)} sayfa başarıyla yüklendi.\n")
    return all_texts


def create_chunks(documents: list, chunk_size: int = CHUNK_SIZE, 
                  chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Dökümanları RAG için optimize edilmiş chunk'lara böler."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=SEPARATORS,
        add_start_index=True,
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"{'='*60}")
    print(f"✂️  CHUNKING SONUÇLARI:")
    print(f"   • Toplam chunk: {len(chunks)}")
    print(f"   • Chunk boyutu: {chunk_size} karakter")
    print(f"   • Overlap: {chunk_overlap} karakter")
    print(f"{'='*60}\n")
    
    if chunks:
        print("📝 Örnek Chunk:")
        print("-" * 60)
        print(f"Kaynak: {chunks[0].metadata.get('source', 'N/A')}")
        print(f"Sayfa: {chunks[0].metadata.get('page', 'N/A')}")
        print(f"\nİçerik (ilk 300 karakter):")
        print(chunks[0].page_content[:300] + "...\n")
        print("-" * 60 + "\n")
    
    return chunks


def build_vector_store_with_progress(chunks: list, model_name: str = EMBEDDING_MODEL, 
                                     index_path: str = INDEX_PATH,
                                     batch_size: int = BATCH_SIZE) -> FAISS:
    """
    Embedding modeli ile FAISS vektör deposu oluşturur - Progress bar ile!
    
    Args:
        chunks: Embedding yapılacak chunk'lar
        model_name: Kullanılacak embedding modeli
        index_path: İndeksin kaydedileceği yol
        batch_size: Her batch'te kaç chunk işlenecek
        
    Returns:
        FAISS: Oluşturulan vektör deposu
    """
    print(f"{'='*60}")
    print(f"🤖 EMBEDDING MODELİ YÜKLENİYOR...")
    print(f"   Model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        # Embedding modelini yükle
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': batch_size,
            }
        )
        
        print(f"🔄 EMBEDDING İŞLEMİ BAŞLIYOR...")
        print(f"   • Toplam chunk: {len(chunks)}")
        print(f"   • Batch boyutu: {batch_size}")
        print(f"   • Tahmini batch sayısı: {(len(chunks) + batch_size - 1) // batch_size}\n")
        
        # Batch'ler halinde embedding yap
        vectorstore = None
        
        for i in tqdm(range(0, len(chunks), batch_size), 
                     desc="🧮 Embedding yapılıyor",
                     unit="batch",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            
            batch = chunks[i:i + batch_size]
            
            if vectorstore is None:
                # İlk batch - yeni vectorstore oluştur
                vectorstore = FAISS.from_documents(batch, embedding_model)
            else:
                # Sonraki batch'ler - mevcut vectorstore'a ekle
                batch_vectorstore = FAISS.from_documents(batch, embedding_model)
                vectorstore.merge_from(batch_vectorstore)
        
        print(f"\n✅ Embedding tamamlandı!\n")
        
        # İndeksi kaydet
        print("💾 İndeks kaydediliyor...")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vectorstore.save_local(index_path)
        
        print(f"\n{'='*60}")
        print(f"✅ BAŞARILI!")
        print(f"   • İndeks konumu: {index_path}")
        print(f"   • Toplam vektör: {vectorstore.index.ntotal}")
        print(f"   • Vektör boyutu: {vectorstore.index.d}")
        print(f"   • Dosya boyutu: {get_folder_size(index_path):.2f} MB")
        print(f"{'='*60}\n")
        
        return vectorstore
        
    except Exception as e:
        print(f"\n❌ EMBEDDING HATASI: {e}")
        raise


def get_folder_size(folder_path: str) -> float:
    """Klasör boyutunu MB cinsinden döndürür"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # MB'ye çevir
    except:
        return 0.0


def main():
    """Ana fonksiyon - tüm pipeline'ı çalıştırır"""
    print("\n" + "="*60)
    print("🚀 HUAWEI CLOUD RAG - VEKTÖR İNDEKSİ OLUŞTURMA")
    print("="*60 + "\n")
    
    try:
        # 1. PDF'leri yükle
        documents = load_pdfs(PDF_FOLDER)
        
        # 2. Chunk'lara böl
        chunks = create_chunks(documents)
        
        # 3. FAISS vektör deposu oluştur (Progress bar ile!)
        vectorstore = build_vector_store_with_progress(chunks)
        
        print("\n" + "="*60)
        print("🎉 İŞLEM TAMAMLANDI!")
        print("="*60)
        print(f"\nŞimdi 'rag_query.py' ile sorgu yapabilirsiniz.\n")
        
    except Exception as e:
        print(f"\n❌ HATA: {e}\n")
        raise


if __name__ == "__main__":
    main()
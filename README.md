# Huawei Cloud RAG - Q&A System

Huawei Cloud dokümantasyonu üzerinden RAG tabanlı soru-cevap sistemi.

## Proje Yapısı

- **`main.py`** - Ana uygulama, soru-cevap döngüsü
- **`rag_engine.py`** - RAG motoru, doküman retrieval
- **`llm_utils.py`** - LLM yönetimi ve prompt oluşturma
- **`vectorstore.py`** - FAISS vektör deposu işlemleri
- **`chat_history.py`** - Chat geçmişi ve bağlam analizi
- **`diagram_handler.py`** - @diagram sorguları yönetimi
- **`diagram_chat.py`** - Diagram oluşturma fonksiyonları
- **`embed_builder.py`** - PDF'lerden vektör indeksi oluşturma
- **`config.py`** - Sistem konfigürasyonu

## Kurulum Adımları

### 1. Sanal Ortam
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 2. Bağımlılıklar
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
`.env` dosyası oluşturun:
```env
QWEN_API_KEY=your_api_key
QWEN_API_BASE=your_api_base
QWEN_MODEL=your_model
```

### 4. PDF Dosyaları
```bash
# PDF dosyalarınızı docs/ klasörüne koyun, repoda varsa gerek yok.
```

### 5. Vektör İndeksi (İlk Kurulum)
```bash
python embed_builder.py
```
Repoda eğer embeddings/faiss_index/index.faiss varsa bunu çalıştırmana gerek yok.

### 6. Çalıştırma
```bash
python main.py
```

## Kullanım Şekli

**Normal Soru:**
```
Question: Huawei Cloud güvenlik özellikleri nelerdir? /İngilizce istem kabul ediyoruz.
```

**Diagram Oluşturma:**
```
Question: @diagram mobile app deployment
```

## 🔧 Ayarlar

- `TOP_K`: Doküman sayısı (varsayılan: 20)
- `TEMPERATURE`: LLM yaratıcılık (varsayılan: 0)
- `MAX_HISTORY`: Chat geçmişi (varsayılan: 5)

## 🐛 Sorun Giderme

- **"FAISS index not found"** → `python embed_builder.py`
- **"API key not found"** → `.env` dosyasını kontrol edin
- **"No PDF files found"** → `docs/` klasörüne PDF ekleyin

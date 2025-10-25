"""
rag_engine.py
RAG (Retrieval Augmented Generation) engine operations.
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from vectorstore import display_sources
from llm_utils import create_rag_prompt
from chat_history import ChatHistory


def query_rag_system(vectorstore: FAISS, llm: ChatOpenAI, query: str, top_k: int = 20, chat_history: ChatHistory = None):
    """RAG sistemine sorgu yapar ve sonucu döndürür."""
    try:
        # Bağlam kontrolü: Önceki soruyla ilişkili mi?
        try:
            is_related, contextualized_query = chat_history.is_related_to_previous(query)
            query_to_use = contextualized_query if (is_related and contextualized_query) else query

            if is_related:
                print("Bağlam kuruldu. Sorun, önceki soruyla ilişkili.")
            
        except (NameError, AttributeError):
            # Eğer chat_history tanımlı değilse bağlam kontrolünü atla
            query_to_use = query

        # 1. İlgili dokümanları bul (retrieval)
        print(f"Query: '{query}'")
        print(f"   Retrieving top-{top_k} documents...\n")
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        relevant_docs = retriever.invoke(query_to_use)
        
        if not relevant_docs:
            print("No relevant documents found!")
            return None
        
        print(f"Found {len(relevant_docs)} relevant documents.")
        
        # 2. Context oluştur
        context = "\n\n---\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        # 3. Prompt hazırla
        prompt = create_rag_prompt(context, query)
        
        # 4. LLM'den cevap al
        print("Generating answer...\n")
        response = llm.invoke(prompt)
        
        # 5. Sonucu göster
        print("ANSWER:")
        print(response.content)
        
        if chat_history:
            chat_history.add_exchange(query, response.content)
        
        # 6. Kaynakları göster
        display_sources(relevant_docs, show_content=False)
        
        return response.content
        
    except Exception as e:
        print(f"\nError processing query: {e}")
        return None

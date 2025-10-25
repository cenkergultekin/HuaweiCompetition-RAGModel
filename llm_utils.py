"""
llm_utils.py
LLM initialization and utility functions.
"""

import os
from langchain_openai import ChatOpenAI


def initialize_llm(api_key: str, api_base: str, model_name: str, temperature: float) -> ChatOpenAI:
    """LLM modelini başlatır."""
    # LangChain API ortam değişkenleri
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = api_base
      
    llm = ChatOpenAI(
        model=model_name, 
        temperature=temperature,
        max_tokens=5000
    )
    
    print("LLM ready!\n")
    return llm


def create_rag_prompt(context: str, query: str) -> str:
    """RAG için optimize edilmiş prompt oluşturur."""
    prompt = f"""You are a Senior Cloud Engineer specialized in Huawei Cloud. Answer the question based on the provided documentation with your expertise.

RULES:
1. Use ONLY the information from the provided documents
2. Do not make up information not present in the documents
3. If you're unsure or the information is not in the documents, say "This information is not available in the provided documentation"
4. Use technical terms correctly and professionally
5. Structure your answer clearly and concisely
6. Provide practical insights when relevant based on your cloud engineering expertise
7. Help a student who wants to learn about the cloud by acting as a teacher


DOCUMENTATION:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

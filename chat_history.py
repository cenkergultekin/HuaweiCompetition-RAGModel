"""
chat_history.py
Chat history management and context analysis.
"""

from langchain_openai import ChatOpenAI
from typing import Optional, List, Dict


class ChatHistory:
    """Chat geçmişini yönetir ve ilişkili sorguları tespit eder."""
    
    def __init__(self, llm: ChatOpenAI, max_history: int = 5):
        self.llm = llm
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_exchange(self, query: str, answer: str):
        """Soru-cevap çiftini history'ye ekler."""
        self.history.append({"query": query, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_history_context(self) -> str:
        """History'yi text formatına çevirir."""
        if not self.history:
            return ""
        return "\n".join(
            f"Q{i}: {h['query']}\nA{i}: {h['answer'][:200]}..."
            for i, h in enumerate(self.history, 1)
        )

    def is_related_to_previous(self, current_query: str) -> tuple[bool, Optional[str]]:
        """Sorgunun önceki sorguyla ilişkili olup olmadığını kontrol eder."""
        if not self.history:
            return False, None
        
        last_query = self.history[-1]["query"] 
        prompt = f"""
You are analyzing if two queries are related.

PREVIOUS QUERY: {last_query}
CURRENT QUERY: {current_query}

Answer ONLY:
STATUS: [RELATED/UNRELATED]
STANDALONE_QUERY: [Rewritten query if RELATED, else empty]
"""
        # yapay zekanın önceki sorguyla yeni sorguyu ilişkili mi diye kontrol eder.
        try:
            res = self.llm.invoke(prompt).content.strip().splitlines()
            status = next((l.split(":")[1].strip() for l in res if l.startswith("STATUS:")), "")
            standalone = next((l.split(":")[1].strip() for l in res if l.startswith("STANDALONE_QUERY:")), "")
            return (status == "RELATED", standalone or None)
        except Exception:
            return False, None
    
    def clear(self):
        """History'yi temizler."""
        self.history.clear()

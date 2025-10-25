"""
diagram_handler.py
Diagram generation and clarification handling.
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from diagram_chat import generate_diagram_flow, get_clarification_questions, enhance_query_with_answers


def handle_diagram_query(query: str, vectorstore: FAISS, llm: ChatOpenAI, top_k: int = 20):
    """Handle @diagram queries with clarification flow."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    # Clarification döngüsü - 6 soru için
    clarification_answers = {}
    question_index = 0
    max_questions = 6
    
    while question_index < max_questions:
        diagram_result = generate_diagram_flow(
            query, retriever, llm, 
            top_k=top_k, 
            clarification_answers=clarification_answers, 
            question_index=question_index
        )
        
        # Eğer soru döndüyse
        if isinstance(diagram_result, str) and "Options:" in diagram_result:
            print(f"\n{diagram_result}")
            answer = input("Your answer: ").strip()
            
            # Default değer işleme
            if not answer:
                # Default değeri al
                question_info = get_clarification_questions("", question_index)
                if question_info and "default" in question_info:
                    answer = question_info["default"]
                    print(f"Using default: {answer}")
            
            if answer:
                clarification_answers[f"question_{question_index}"] = answer
                question_index += 1
        else:
            # JSON döndü, diagram tamamlandı
            print("\n=== DIAGRAM JSON ===")
            print(diagram_result)
            return diagram_result
    
    # 6 soru tamamlandıysa direkt RAG'a git
    if len(clarification_answers) >= 6:
        # Enhanced query'yi göster
        enhanced_query = enhance_query_with_answers(query, clarification_answers)
        print(f"\nYour query: {enhanced_query}")
        
        diagram_result = generate_diagram_flow(
            query, retriever, llm, 
            top_k=top_k, 
            clarification_answers=clarification_answers, 
            question_index=question_index
        )
        print("\n=== DIAGRAM JSON ===")
        print(diagram_result)
        return diagram_result
    
    return None

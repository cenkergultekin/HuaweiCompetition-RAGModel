import json
import os
from typing import List, Dict, Tuple

# ---- Şema sabitleri ----
TECH_JSON_REQUIRED_KEYS = ["technologies", "relationships", "explanation"]
TECH_ITEM_KEYS = ["name", "category", "description", "node_id", "node_label"]
REL_ITEM_KEYS = ["from", "to", "type"]

DEFAULTS = {
    "assumptions": {
        "region": "eu-west-0",
        "traffic_rps": 1000,
        "gpu": 0,
        "privacy": "public-non-sensitive",
        "realtime": True
    }
}

# Load clarification configuration
def load_clarification_config():
    """Load clarification configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'clarification_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

CLARIFICATION_CONFIG = load_clarification_config()

def strip_diagram_intent(query_text: str) -> str:
    """@diagram önekini temizler."""
    if not query_text:
        return ""
    q = query_text.strip()
    if q.lower().startswith("@diagram"):
        q = q[len("@diagram"):].strip()
    return q or query_text

def retrieve_context(retriever, query_text: str, k: int = 20) -> Tuple[List, str]:
    """RAG: ilgili dokümanları getirip tek metin haline dönüştürür."""
    docs = retriever.invoke(query_text)
    documentation = "\n\n---\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    return docs, documentation

def build_prompt(query_text: str, documentation: str) -> str:
    """LLM için yalnız JSON döndürecek sıkı prompt."""
    return f"""
You are a Cloud Solution Architect. The user requests a deployment flow for a *MOBILE* project on Huawei Cloud.
Use ONLY the DOCUMENTATION below and the user request. Output MUST be VALID JSON and nothing else.

TARGET JSON SCHEMA (strict):
{{
  "technologies": [
    {{
      "name": "Huawei Cloud Servis Adı",
      "category": "Kategori",
      "description": "Açıklama",
      "node_id": 1,
      "node_label": "Servis Adı"
    }}
  ],
  "relationships": [
    {{
      "from": "Servis1",
      "to": "Servis2",
      "type": "veri akışı türü"
    }}
  ],
  "explanation": "Açıklama"
}}

RULES:
- Only return the JSON (no extra text).
- Use services strictly supported by DOCUMENTATION; avoid hallucinations.
- If info is missing, use these defaults and mention them in "explanation":
{json.dumps(DEFAULTS, ensure_ascii=False)}
- "technologies": each item MUST include exactly these keys: {TECH_ITEM_KEYS}.
- "relationships": each item MUST include exactly these keys: {REL_ITEM_KEYS}.
- "relationships.from"/"to" MUST match a "technologies.name".
- Prefer MOBILE deployment blocks when applicable: API Gateway, FunctionGraph or CCE, GaussDB, OBS, CDN, Cloud Eye, WAF/IAM (only if DOCUMENTATION supports).

DOCUMENTATION:
{documentation}

USER_REQUEST:
{query_text}
""".strip()

def invoke_llm_json(llm, prompt: str) -> str:
    """LLM çağrısı: ham string döner (parse burada yapılmaz)."""
    return llm.invoke(prompt).content.strip()

def parse_json_strict(payload_str: str) -> Dict:
    """Sıkı JSON parse. Geçersizse ValueError fırlatır."""
    return json.loads(payload_str)

def validate_payload(payload: Dict) -> Tuple[bool, List[str]]:
    """Şemayı doğrular; hataları döndürür."""
    errors: List[str] = []

    # Kök anahtarlar
    for k in TECH_JSON_REQUIRED_KEYS:
        if k not in payload:
            errors.append(f"missing key: {k}")

    # technologies
    techs = payload.get("technologies", [])
    if not isinstance(techs, list):
        errors.append("technologies must be a list")
        techs = []

    names = set()
    for i, t in enumerate(techs, 1):
        if not isinstance(t, dict):
            errors.append(f"technologies[{i}] must be an object")
            continue
        for req in TECH_ITEM_KEYS:
            if req not in t:
                errors.append(f"technologies[{i}] missing key: {req}")
        # tip kontrolleri
        if "node_id" in t and not isinstance(t["node_id"], int):
            errors.append(f"technologies[{i}].node_id must be int")
        # name topla
        if "name" in t and isinstance(t["name"], str) and t["name"].strip():
            names.add(t["name"].strip())

    # relationships
    rels = payload.get("relationships", [])
    if not isinstance(rels, list):
        errors.append("relationships must be a list")
        rels = []

    for j, r in enumerate(rels, 1):
        if not isinstance(r, dict):
            errors.append(f"relationships[{j}] must be an object")
            continue
        for req in REL_ITEM_KEYS:
            if req not in r:
                errors.append(f"relationships[{j}] missing key: {req}")
        # isim eşleşmesi
        f = (r.get("from") or "").strip()
        t = (r.get("to") or "").strip()
        if f and f not in names:
            errors.append(f'relationships[{j}].from "{f}" not in technologies.name')
        if t and t not in names:
            errors.append(f'relationships[{j}].to "{t}" not in technologies.name')

    return (len(errors) == 0, errors)

def normalize_payload(payload: Dict) -> Dict:
    """Ufak onarımlar: node_id sıralama, boş label, trim vb."""
    
    tech_names = {t["name"] for t in payload.get("technologies", []) if isinstance(t, dict)}
    extra_nodes = set()
    for rel in payload.get("relationships", []):
        for endpoint in [rel.get("from"), rel.get("to")]:
            if endpoint and endpoint not in tech_names and endpoint not in ["All Services"]:
                extra_nodes.add(endpoint)

    for name in extra_nodes:
        if name in tech_names:  # zaten varsa ekleme
            continue
        tech_names.add(name)
        payload["technologies"].append({
        "name": name,
        "category": "External",
        "description": "User-defined or external component.",
        "node_id": len(payload["technologies"]) + 1,
        "node_label": name
    })

    techs = payload.get("technologies", [])
    # node_id yoksa/çakışırsa 1..N yeniden ata
    for idx, t in enumerate(techs, 1):
        if not isinstance(t, dict):
            continue
        t["node_id"] = idx
        # label boşsa name kopyala
        if not t.get("node_label"):
            t["node_label"] = (t.get("name") or "").strip()
        # trim alanlar
        if "name" in t and isinstance(t["name"], str):
            t["name"] = t["name"].strip()
        if "category" in t and isinstance(t["category"], str):
            t["category"] = t["category"].strip()
        if "description" in t and isinstance(t["description"], str):
            t["description"] = t["description"].strip()
        if "node_label" in t and isinstance(t["node_label"], str):
            t["node_label"] = t["node_label"].strip()

    # relationships trim
    rels = payload.get("relationships", [])
    for r in rels:
        if not isinstance(r, dict):
            continue
        if "from" in r and isinstance(r["from"], str):
            r["from"] = r["from"].strip()
        if "to" in r and isinstance(r["to"], str):
            r["to"] = r["to"].strip()
        if "type" in r and isinstance(r["type"], str):
            r["type"] = r["type"].strip()

    # explanation zorunlu alan
    if "explanation" not in payload or not isinstance(payload["explanation"], str):
        payload["explanation"] = "Generated with defaults; please review."

    return payload

def get_clarification_questions(query: str, current_question_index: int = 0) -> Dict:
    """Tek tek soru döner, sırasıyla."""
    config = CLARIFICATION_CONFIG
    priority_order = config["collection_strategy"]["priority_order"]
    max_questions = config["collection_strategy"]["max_questions"]
    
    if current_question_index >= max_questions:
        return None  # Tüm sorular bitti
    
    # Mevcut soruyu bul
    param_path = priority_order[current_question_index]
    param_config = None
    
    for category in config["required_parameters"].values():
        if param_path in category:
            param_config = category[param_path]
            break
    
    if not param_config:
        return None
    
    return {
        "question": param_config["question"],
        "options": param_config["options"],
        "default": param_config["default"],
        "criticality": param_config["criticality"],
        "unknown_warning": param_config["unknown_warning"],
        "current_index": current_question_index,
        "total_questions": max_questions
    }

def enhance_query_with_answers(original_query: str, answers: Dict[str, str]) -> str:
    """Cevapları optimize query'ye dönüştürür."""
    
    # Kısa ve öz mapping
    keyword_mappings = {
        "mobile-backend": "mobile backend",
        "web-application": "web app", 
        "structured-db": "SQL database",
        "file-storage": "file storage",
        "small-1k": "small scale",
        "medium-10k": "medium scale",
        "basic-security": "basic security",
        "balanced": "balanced performance",
        "basic-uptime": "basic uptime"
    }
    
    # Optimize context
    context_parts = []
    for param, answer in answers.items():
        if answer in keyword_mappings:
            context_parts.append(keyword_mappings[answer])
        else:
            context_parts.append(answer)
    
    # Kısa query
    enhanced = f"{original_query} {' '.join(context_parts)} Huawei Cloud"
    return enhanced

def has_sufficient_info(query: str, documentation: str) -> bool:
    """Dokümanlarda yeterli bilgi var mı kontrol eder."""
    query_lower = query.lower()
    
    # Mobil deployment için gerekli anahtar kelimeler
    mobile_keywords = ['mobile', 'app', 'deployment', 'api', 'database', 'storage']
    found_keywords = sum(1 for keyword in mobile_keywords if keyword in query_lower)
    
    # Dokümanlarda ilgili servisler var mı?
    doc_lower = documentation.lower()
    huawei_services = ['huawei cloud', 'obs', 'gaussdb', 'functiongraph', 'cce', 'api gateway']
    found_services = sum(1 for service in huawei_services if service in doc_lower)
    
    # En az 2 anahtar kelime ve 1 servis bulunmalı
    return found_keywords >= 2 and found_services >= 1

def generate_diagram_flow(query_text: str, retriever, llm, top_k: int = 20, clarification_answers: Dict = None, question_index: int = 0) -> Dict:
    """@diagram isteği için: bağlam topla → promptla LLM → JSON üret → validate/normalize."""
    clean_query = strip_diagram_intent(query_text)
    
    # İlk çağrı veya henüz tüm sorular sorulmadıysa
    if not clarification_answers or len(clarification_answers) < 6:
        # Hangi soruyu soracağını belirle
        current_index = len(clarification_answers) if clarification_answers else 0
        question = get_clarification_questions(clean_query, current_index)
        
        if question:
            options_str = " ".join([f"[{opt}]" for opt in question["options"]])
            return f"{question['question']}\nOptions: {options_str}"
    
    # Sadece tüm cevaplar toplandığında RAG + LLM'e git
    if clarification_answers and len(clarification_answers) >= 6:
        clean_query = enhance_query_with_answers(clean_query, clarification_answers)
        
        docs, documentation = retrieve_context(retriever, clean_query, k=top_k)

        if not docs:
            return {
                "technologies": [],
                "relationships": [],
                "explanation": "No relevant documents found. Provide more details (e.g., traffic, region).",
                "clarification_needed": True,
                "questions": get_clarification_questions(clean_query)
            }
        
        # Geçici olarak has_sufficient_info kontrolünü devre dışı bırak
        # if not has_sufficient_info(clean_query, documentation):
        #     return {
        #         "technologies": [],
        #         "relationships": [],
        #         "explanation": "Insufficient information in knowledge base for this specific request. Please provide more context or try a different approach.",
        #         "clarification_needed": True,
        #         "questions": get_clarification_questions(clean_query)
        #     }

        prompt = build_prompt(clean_query, documentation)

        raw = invoke_llm_json(llm, prompt)

        try:
            payload = parse_json_strict(raw)
        except ValueError:
            return {
                "technologies": [],
                "relationships": [],
                "explanation": "Model returned invalid JSON. Please retry."
            }

        ok, errs = validate_payload(payload)
        if not ok:
            # validation hatalarını explanation'a ekle
            msg = " | validation_errors=" + "; ".join(errs)
            if "explanation" in payload and isinstance(payload["explanation"], str):
                payload["explanation"] = (payload["explanation"] or "").strip() + msg
            else:
                payload["explanation"] = "Generated with issues." + msg

        payload = normalize_payload(payload)
        return payload












"""
config.py
Configuration management for the RAG system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("QWEN_API_KEY")
API_BASE = os.getenv("QWEN_API_BASE")
MODEL_NAME = os.getenv("QWEN_MODEL")

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_PATH = "embeddings/faiss_index"

# Retrieval Parameters
TOP_K = 20
TEMPERATURE = 0

# Chat History
MAX_HISTORY = 5

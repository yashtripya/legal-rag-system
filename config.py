"""
Configuration Settings for Indian Legal RAG System.
Provides centralized path management, environment defaults, model settings, and RAG hyperparameters.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DB_DIR = STORAGE_DIR / "chroma_db"

# Ensure essential directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, STORAGE_DIR, CHROMA_DB_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ChromaDB Settings
COLLECTION_NAME = os.getenv("LEGAL_RAG_COLLECTION_NAME", "indian_legal_docs")

# Embedding Model Configuration
# Uses lightweight local model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Text Splitter Settings
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval Hyperparameters
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.05"))

# LLM Configuration (Local Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "llama3.2:1b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))

# Supported LLM Models
SUPPORTED_LLM_MODELS = [
    "llama3.2:1b",
    "qwen2.5:1.5b",
    "tinyllama",
    "llama3.2",
    "mistral",
]

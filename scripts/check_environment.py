"""
Environment Verification Utility for Legal RAG System.
Validates dependencies, storage directory access, sentence transformer loading, and Ollama connectivity.
"""

import sys
import io
import os
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, STORAGE_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL
from src.embeddings import LocalLegalEmbeddings
from src.llm_service import LocalLegalLLMService


def check_env():
    print("==================================================")
    print("🔍  LEGAL RAG SYSTEM - ENVIRONMENT & DEPENDENCY CHECK")
    print("==================================================")

    # 1. Check Python version
    py_ver = sys.version.split()[0]
    print(f"✅ Python Version: {py_ver}")

    # 2. Check Directories
    for d_name, d_path in [("Data Dir", DATA_DIR), ("Storage Dir", STORAGE_DIR), ("ChromaDB Dir", CHROMA_DB_DIR)]:
        if d_path.exists():
            print(f"✅ Directory Found: {d_name} -> {d_path}")
        else:
            print(f"❌ Directory Missing: {d_name} -> {d_path}")

    # 3. Test Sentence Transformers Embeddings
    print(f"\n🧠 Testing Local Embedding Model ({EMBEDDING_MODEL_NAME})...")
    try:
        embeddings = LocalLegalEmbeddings()
        sample_vector = embeddings.embed_query("Supreme Court Basic Structure")
        print(f"✅ Embeddings Working! Vector Dimensions: {len(sample_vector)}")
    except Exception as e:
        print(f"❌ Embeddings Error: {str(e)}")

    # 4. Check Ollama Status
    print(f"\n🤖 Checking Local Ollama Service ({OLLAMA_BASE_URL})...")
    llm = LocalLegalLLMService()
    health = llm.check_health()
    if health.get("status") == "online":
        print(f"✅ Ollama Status: ONLINE")
        print(f"   Available Models: {', '.join(health.get('available_models', []))}")
    else:
        print(f"⚠️  Ollama Status: OFFLINE")
        print(f"   Note: The app will run in Direct Retrieval Fallback Mode.")
        print(f"   To enable AI answers, start Ollama: `ollama serve` & `ollama pull llama3.2:1b`")

    print("\n==================================================")
    print("✨ Environment Check Complete!")
    print("==================================================\n")


if __name__ == "__main__":
    check_env()

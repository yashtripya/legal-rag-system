"""
CLI Script to Ingest Legal Documents into ChromaDB Vector Database.
Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --dir path/to/legal/docs
    python scripts/ingest_documents.py --file path/to/document.pdf
"""

import sys
import io
import argparse
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_DATA_DIR
from src.rag_pipeline import LegalRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest Legal Documents into Vector Database")
    parser.add_argument("--dir", type=str, default=str(SAMPLE_DATA_DIR), help="Directory containing PDF/TXT files")
    parser.add_argument("--file", type=str, default=None, help="Single file path to ingest")
    parser.add_argument("--clear", action="store_true", help="Clear existing index before ingestion")

    args = parser.parse_args()

    print("==================================================")
    print("⚖️  INDIAN LEGAL RAG - DOCUMENT INGESTION PIPELINE")
    print("==================================================")

    pipeline = LegalRAGPipeline()

    if args.clear:
        print("\n🗑️  Clearing existing vector index...")
        pipeline.clear_index()
        print("✅ Vector collection reset.")

    if args.file:
        file_p = Path(args.file).resolve()
        print(f"\n📄 Ingesting single file: {file_p.name}...")
        res = pipeline.ingest_file(file_p)
        print(f"Status: {res.get('status').upper()}")
        print(f"Chunks Processed: {res.get('chunks_processed', 0)} | Added: {res.get('chunks_added', 0)}")
        if res.get("status") == "error":
            print(f"❌ Error: {res.get('error')}")
    else:
        dir_p = Path(args.dir).resolve()
        print(f"\n📁 Ingesting directory: {dir_p}...")
        summary = pipeline.ingest_directory(dir_p)
        
        print(f"\nTotal Files Found: {summary.get('total_files_found')}")
        print(f"✅ Successful Ingestions: {summary.get('successful_files')}")
        print(f"❌ Failed Ingestions: {summary.get('failed_files')}")

        for item in summary.get("details", []):
            status_icon = "✅" if item.get("status") == "success" else "❌"
            print(f"  {status_icon} {item.get('filename')}: Chunks Added={item.get('chunks_added', 0)}")
            if item.get("status") == "error":
                print(f"     Reason: {item.get('error')}")

    stats = pipeline.vector_store.get_stats()
    print("\n--------------------------------------------------")
    print(f"📊 Vector Database Summary:")
    print(f"   Collection Name: {stats.get('collection_name')}")
    print(f"   Embedding Model: {stats.get('embedding_model')}")
    print(f"   Total Indexed Documents: {stats.get('total_documents')}")
    print(f"   Total Indexed Chunks:    {stats.get('total_chunks')}")
    print("==================================================\n")


if __name__ == "__main__":
    main()

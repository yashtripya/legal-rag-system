"""
RAG Evaluation Benchmark Script for Legal RAG System.
Measures Retrieval Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and Citation Coverage.
"""

import sys
import io
import json
from pathlib import Path
from typing import List, Dict, Any

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DATA_DIR, SAMPLE_DATA_DIR
from src.rag_pipeline import LegalRAGPipeline


# Standard Benchmark Questions over Sample Legal Dataset
BENCHMARK_DATASET = [
    {
        "id": "Q1",
        "question": "What is the Basic Structure Doctrine established by the Supreme Court of India?",
        "expected_doc": "Kesavananda Bharati v. State of Kerala",
        "keywords": ["basic structure", "article 368", "amend", "essential features"],
    },
    {
        "id": "Q2",
        "question": "Is the Right to Privacy a fundamental right under the Constitution of India?",
        "expected_doc": "Justice K.S. Puttaswamy v. Union of India",
        "keywords": ["privacy", "article 21", "fundamental right", "proportionality"],
    },
    {
        "id": "Q3",
        "question": "What constitutes the offense of Murder under Section 300 of the Indian Penal Code?",
        "expected_doc": "Indian Penal Code 1860 - Sections 299, 300, 302",
        "keywords": ["section 300", "intention of causing death", "culpable homicide", "bodily injury"],
    },
    {
        "id": "Q4",
        "question": "What are the exceptions where culpable homicide is not considered murder under IPC?",
        "expected_doc": "Indian Penal Code 1860 - Sections 299, 300, 302",
        "keywords": ["grave and sudden provocation", "private defense", "sudden fight", "exception"],
    },
    {
        "id": "Q5",
        "question": "What protection is guaranteed under Article 21 of the Indian Constitution?",
        "expected_doc": "Constitution of India - Articles 14, 19, 21",
        "keywords": ["article 21", "life", "personal liberty", "procedure established by law"],
    },
    {
        "id": "Q6",
        "question": "How did the Supreme Court expand the scope of Article 21 in landmark decisions?",
        "expected_doc": "Constitution of India - Articles 14, 19, 21",
        "keywords": ["human dignity", "maneka gandhi", "privacy", "fair procedure"],
    },
    {
        "id": "Q7",
        "question": "What is the three-pronged test for permissible state restriction on privacy under Puttaswamy?",
        "expected_doc": "Justice K.S. Puttaswamy v. Union of India",
        "keywords": ["legality", "legitimate state aim", "proportionality", "three-pronged"],
    },
    {
        "id": "Q8",
        "question": "What is the punishment for murder specified in Section 302 of the IPC?",
        "expected_doc": "Indian Penal Code 1860 - Sections 299, 300, 302",
        "keywords": ["death", "imprisonment for life", "section 302", "fine"],
    },
    {
        "id": "Q9",
        "question": "What are the reasonable restrictions on Freedom of Speech under Article 19(2)?",
        "expected_doc": "Constitution of India - Articles 14, 19, 21",
        "keywords": ["sovereignty", "security of the state", "public order", "decency"],
    },
    {
        "id": "Q10",
        "question": "What limits exist on Parliament's power to amend the Constitution under Article 368?",
        "expected_doc": "Kesavananda Bharati v. State of Kerala",
        "keywords": ["article 368", "abrogating", "judicial review", "amend"],
    },
]


def run_evaluation():
    print("==================================================")
    print("🧪  INDIAN LEGAL RAG - RETRIEVAL EVALUATION BENCHMARK")
    print("==================================================")

    pipeline = LegalRAGPipeline()

    # Ensure sample data is ingested
    stats = pipeline.vector_store.get_stats()
    if stats.get("total_chunks", 0) == 0:
        print("📦 Ingesting sample dataset for evaluation...")
        pipeline.ingest_directory(SAMPLE_DATA_DIR)

    results = []
    total_precision = 0.0
    total_recall = 0.0
    reciprocal_ranks = []
    citations_present = 0
    top_k = 4

    print(f"\nEvaluating {len(BENCHMARK_DATASET)} legal queries (top_k={top_k})...\n")

    for item in BENCHMARK_DATASET:
        qid = item["id"]
        query = item["question"]
        expected_doc = item["expected_doc"]
        keywords = item["keywords"]

        # Run pipeline retrieval & Q&A
        response = pipeline.query(user_query=query, top_k=top_k)
        raw_passages = response.get("raw_passages", [])
        citations = response.get("citations", [])

        # Evaluate Precision & Recall
        relevant_chunks = 0
        first_rel_rank = 0

        for rank, pass_data in enumerate(raw_passages, start=1):
            content = pass_data["content"].lower()
            doc_title = pass_data["metadata"].get("document_title", "")
            
            # Match condition: expected doc match or at least 2 keywords match
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
            is_relevant = (expected_doc.lower() in doc_title.lower()) or (keyword_matches >= 2)

            if is_relevant:
                relevant_chunks += 1
                if first_rel_rank == 0:
                    first_rel_rank = rank

        precision = relevant_chunks / len(raw_passages) if raw_passages else 0.0
        recall = 1.0 if relevant_chunks > 0 else 0.0
        mrr = (1.0 / first_rel_rank) if first_rel_rank > 0 else 0.0

        if citations:
            citations_present += 1

        total_precision += precision
        total_recall += recall
        reciprocal_ranks.append(mrr)

        query_res = {
            "id": qid,
            "query": query,
            "retrieved_count": len(raw_passages),
            "relevant_retrieved": relevant_chunks,
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "mrr": round(mrr, 4),
            "citations_generated": len(citations),
        }
        results.append(query_res)

        print(f"[{qid}] P@{top_k}={precision:.2f} | R@{top_k}={recall:.2f} | MRR={mrr:.2f} | Query: '{query[:50]}...'")

    num_queries = len(BENCHMARK_DATASET)
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    mean_mrr = sum(reciprocal_ranks) / num_queries
    citation_coverage = (citations_present / num_queries) * 100.0

    eval_summary = {
        "dataset_size": num_queries,
        "top_k": top_k,
        "mean_precision_at_k": round(avg_precision, 4),
        "mean_recall_at_k": round(avg_recall, 4),
        "mean_reciprocal_rank_mrr": round(mean_mrr, 4),
        "citation_coverage_pct": round(citation_coverage, 2),
        "query_breakdown": results,
    }

    # Save metrics JSON
    out_file = PROCESSED_DATA_DIR / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n--------------------------------------------------")
    print("📊 AGGREGATE BENCHMARK EVALUATION RESULTS:")
    print(f"   Precision@{top_k}:               {avg_precision*100:.1f}%")
    print(f"   Recall@{top_k}:                  {avg_recall*100:.1f}%")
    print(f"   Mean Reciprocal Rank (MRR): {mean_mrr:.4f}")
    print(f"   Citation Coverage:          {citation_coverage:.1f}%")
    print(f"   Detailed Report Saved To:   {out_file}")
    print("==================================================\n")


if __name__ == "__main__":
    run_evaluation()

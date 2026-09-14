"""
demo_task13.py — RAG Triad Evaluation Demo (Capstone Task 13)

Demonstrates the RAG Triad evaluation pipeline:
1. Runs 15 queries (12 in-scope KB topics + 3 out-of-scope) through answer_query()
2. Computes per-query RAG Triad metrics (Context Relevance, Groundedness, Answer Relevance)
3. Computes aggregate averages for all queries and in-scope subset
4. Verifies out-of-scope queries are correctly refused
5. Saves transcript to transcripts/task13_rag_triad.json
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["MOCK_LLM"] = "true"

from evaluation.evaluate_agent import run_rag_triad_evaluation


def main():
    print("=" * 75)
    print("CAPSTONE TASK 13 — RAG TRIAD EVALUATION DEMO")
    print("=" * 75)
    print()

    transcript = run_rag_triad_evaluation()

    print()
    print("=" * 75)
    print("DEMO COMPLETE")
    print("=" * 75)

    agg = transcript["aggregate_averages"]
    print(f"\nFinal Averages (All {transcript['total_queries']} Queries):")
    print(f"  Context Relevance: {agg['all_queries']['context_relevance']:.4f}")
    print(f"  Groundedness:      {agg['all_queries']['groundedness']:.4f}")
    print(f"  Answer Relevance:  {agg['all_queries']['answer_relevance']:.4f}")

    print(f"\nIn-Scope Averages ({transcript['in_scope_queries']} Queries):")
    print(f"  Context Relevance: {agg['in_scope_only']['context_relevance']:.4f}")
    print(f"  Groundedness:      {agg['in_scope_only']['groundedness']:.4f}")
    print(f"  Answer Relevance:  {agg['in_scope_only']['answer_relevance']:.4f}")

    print(f"\nOut-of-Scope: {transcript['out_of_scope_correctly_refused']}/{transcript['out_of_scope_queries']} correctly refused")
    print(f"\nTranscript: transcripts/task13_rag_triad.json")


if __name__ == "__main__":
    main()

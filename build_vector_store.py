"""
build_vector_store.py — Build the Chroma Vector Database

This script is the main entry point for Phase 4.
It connects the full pipeline:

    knowledge_base/  (12 .txt files)
        ↓
    rag/chunking.py  (two chunking strategies)
        ↓
    rag/embeddings.py  (all-MiniLM-L6-v2 embeddings)
        ↓
    Chroma  (two persistent collections)

Usage:
    python build_vector_store.py

Output:
    - chroma_db/  directory with persistent vector storage
    - Collection "fixed_size_chunks"
    - Collection "sentence_chunks"
"""

import os
import sys
import shutil

import chromadb

# Add project root to path so we can import rag modules
sys.path.insert(0, os.path.dirname(__file__))

from rag.chunking import load_documents, chunk_all_documents
from rag.embeddings import load_model, generate_embeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

FIXED_COLLECTION = "fixed_size_chunks"
SENTENCE_COLLECTION = "sentence_chunks"

CHUNK_SIZE = 200   # characters per fixed-size chunk
OVERLAP = 50       # character overlap between consecutive fixed-size chunks


def build():
    """Run the full pipeline: load → chunk → embed → store."""

    # ------------------------------------------------------------------
    # Step 1: Load documents
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading documents from knowledge_base/")
    print("=" * 60)

    documents = load_documents(KB_DIR)
    print(f"  Documents loaded: {len(documents)}")
    for doc in documents:
        print(f"    - {doc['source']} ({len(doc['text'])} chars)")
    print()

    # ------------------------------------------------------------------
    # Step 2: Chunk documents using both strategies
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Chunking documents")
    print("=" * 60)

    fixed_chunks, sent_chunks = chunk_all_documents(
        documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP
    )

    print(f"  Fixed-size chunks: {len(fixed_chunks)}")
    print(f"  Sentence-based chunks: {len(sent_chunks)}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Load embedding model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Loading SentenceTransformer model")
    print("=" * 60)

    model = load_model()
    print(f"  Model: all-MiniLM-L6-v2")
    print(f"  Embedding dimension: {model.get_embedding_dimension()}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Generate embeddings for both chunk sets
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Generating embeddings")
    print("=" * 60)

    fixed_texts = [c["text"] for c in fixed_chunks]
    sent_texts = [c["text"] for c in sent_chunks]

    print(f"  Embedding {len(fixed_texts)} fixed-size chunks...")
    fixed_embeddings = generate_embeddings(model, fixed_texts)

    print(f"  Embedding {len(sent_texts)} sentence-based chunks...")
    sent_embeddings = generate_embeddings(model, sent_texts)

    print(f"  Fixed-size embeddings: {len(fixed_embeddings)} vectors of dim {len(fixed_embeddings[0])}")
    print(f"  Sentence embeddings:   {len(sent_embeddings)} vectors of dim {len(sent_embeddings[0])}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Store in Chroma (two separate collections)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Storing in Chroma")
    print("=" * 60)

    # Remove old database if it exists so we start fresh
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"  Removed old chroma_db/")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # --- Collection 1: Fixed-size chunks ---
    fixed_col = client.create_collection(name=FIXED_COLLECTION)
    fixed_col.add(
        ids=[f"fixed_{i}" for i in range(len(fixed_chunks))],
        documents=fixed_texts,
        embeddings=fixed_embeddings,
        metadatas=[c["metadata"] for c in fixed_chunks],
    )
    print(f"  Collection '{FIXED_COLLECTION}': {fixed_col.count()} chunks stored")

    # --- Collection 2: Sentence-based chunks ---
    sent_col = client.create_collection(name=SENTENCE_COLLECTION)
    sent_col.add(
        ids=[f"sent_{i}" for i in range(len(sent_chunks))],
        documents=sent_texts,
        embeddings=sent_embeddings,
        metadatas=[c["metadata"] for c in sent_chunks],
    )
    print(f"  Collection '{SENTENCE_COLLECTION}': {sent_col.count()} chunks stored")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 4 BUILD COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Documents loaded:         {len(documents)}")
    print(f"  Fixed-size chunks:        {len(fixed_chunks)}")
    print(f"  Sentence-based chunks:    {len(sent_chunks)}")
    print(f"  Embedding dimension:      {model.get_embedding_dimension()}")
    print(f"  Chroma collection '{FIXED_COLLECTION}': {fixed_col.count()} chunks")
    print(f"  Chroma collection '{SENTENCE_COLLECTION}': {sent_col.count()} chunks")
    print(f"  Chroma storage:           {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build()

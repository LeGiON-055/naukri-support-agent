"""
chunking.py — Document Chunking Strategies

Implements two chunking strategies for the RAG pipeline:
1. Fixed-size-with-overlap chunking
2. Sentence-based chunking

Each function returns a list of dictionaries with:
    - "text": the chunk text
    - "metadata": dict containing source filename, chunk index, and strategy name
"""

import os
import re


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size with overlap
# ---------------------------------------------------------------------------

def fixed_size_chunks(text, source, chunk_size=200, overlap=50):
    """
    Split text into fixed-size character chunks with overlap.

    Args:
        text (str): The full document text.
        source (str): The source filename (e.g. "probation_period.txt").
        chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of characters repeated between consecutive chunks.

    Returns:
        list[dict]: Each dict has "text" and "metadata" keys.

    Example with chunk_size=30, overlap=5 on a 100-char string:
        Chunk 0 → characters  0..29
        Chunk 1 → characters 25..54   (overlaps 5 chars with chunk 0)
        Chunk 2 → characters 50..79   (overlaps 5 chars with chunk 1)
        Chunk 3 → characters 75..99   (overlaps 5 chars with chunk 2)
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        # Only keep non-empty chunks
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "strategy": "fixed_size",
                },
            })
            chunk_index += 1

        # Move the window forward by (chunk_size - overlap)
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-based chunking
# ---------------------------------------------------------------------------

def sentence_chunks(text, source):
    """
    Split text into chunks where each chunk is one sentence.

    Sentences are detected by splitting on period, question mark, or
    exclamation mark followed by a space or end-of-string.

    Args:
        text (str): The full document text.
        source (str): The source filename.

    Returns:
        list[dict]: Each dict has "text" and "metadata" keys.
    """
    # Split on sentence-ending punctuation followed by whitespace or end
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    chunk_index = 0

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if sentence:
            chunks.append({
                "text": sentence,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "strategy": "sentence",
                },
            })
            chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Load all documents from the knowledge base directory
# ---------------------------------------------------------------------------

def load_documents(kb_dir):
    """
    Read every .txt file from the knowledge base directory.

    Args:
        kb_dir (str): Absolute or relative path to the knowledge_base folder.

    Returns:
        list[dict]: Each dict has "source" (filename) and "text" (file content).
    """
    documents = []

    for filename in sorted(os.listdir(kb_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                documents.append({"source": filename, "text": text})

    return documents


# ---------------------------------------------------------------------------
# Apply both strategies to all documents
# ---------------------------------------------------------------------------

def chunk_all_documents(documents, chunk_size=200, overlap=50):
    """
    Apply both chunking strategies to a list of documents.

    Args:
        documents (list[dict]): Output of load_documents().
        chunk_size (int): For fixed-size strategy.
        overlap (int): For fixed-size strategy.

    Returns:
        tuple: (fixed_chunks_list, sentence_chunks_list)
            Each is a flat list of chunk dicts from all documents.
    """
    all_fixed = []
    all_sentence = []

    for doc in documents:
        source = doc["source"]
        text = doc["text"]

        all_fixed.extend(fixed_size_chunks(text, source, chunk_size, overlap))
        all_sentence.extend(sentence_chunks(text, source))

    return all_fixed, all_sentence

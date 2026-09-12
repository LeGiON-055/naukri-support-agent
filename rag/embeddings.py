"""
embeddings.py — Text Embedding Generation

Uses a local SentenceTransformer model (all-MiniLM-L6-v2) to convert
text chunks into numerical vectors (embeddings).

No external API key is required — the model runs entirely on the local machine.

Key facts about all-MiniLM-L6-v2:
    - Embedding dimension: 384
    - Suitable for semantic similarity tasks
    - Small and fast enough for a capstone project
"""

from sentence_transformers import SentenceTransformer

# Default model name — used consistently across the project
MODEL_NAME = "all-MiniLM-L6-v2"


def load_model(model_name=MODEL_NAME):
    """
    Load and return a SentenceTransformer model.

    The first call downloads the model (~80 MB) and caches it locally.
    Subsequent calls load from cache instantly.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        SentenceTransformer: The loaded model.
    """
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(model, texts):
    """
    Convert a list of text strings into embedding vectors.

    Args:
        model (SentenceTransformer): A loaded SentenceTransformer model.
        texts (list[str]): The text strings to embed.

    Returns:
        list[list[float]]: One embedding vector per input text.
            Each vector has 384 dimensions for all-MiniLM-L6-v2.
    """
    # encode() returns a numpy array — convert to plain Python lists for Chroma
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

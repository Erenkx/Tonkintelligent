"""
Performs Top-K semantic search using a FAISS index.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_index(
    index_path: str
) -> faiss.Index:
    """
    Loads a FAISS index from disk.

    Args:
        index_path (str): Path to the FAISS index file.

    Returns:
        faiss.Index: Loaded FAISS index.
    """
    return faiss.read_index(index_path)


def embed_query(
    query: str,
    model_name: str = 'all-MiniLM-L6-v2'
) -> np.ndarray:
    """
    Embeds a query string using a sentence transformer model.

    Args:
        query (str): The query string to embed.
        model_name (str): Name of the SentenceTransformer model to use.
            Default is 'all-MiniLM-L6-v2'.

    Returns:
        np.ndarray: The embedded query vector.
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode([query], normalize_embeddings=True)

    return embedding.astype('float32')


def search_faiss_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int = 5
) -> np.ndarray:
    """
    Performs a Top-K search in the FAISS index.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_embedding (np.ndarray): The embedded query vector.
        top_k (int): Number of top results to return. Default is 5.

    Returns:
        np.ndarray: Indices of the top K nearest neighbors.
    """
    _, indices = index.search(query_embedding, top_k)

    return indices[0].tolist()

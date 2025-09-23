"""
Performs Top-K semantic search using a FAISS index.
"""

import faiss
import numpy as np
from openai import OpenAI


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
    client: OpenAI,
    query: str,
    model_name: str = 'text-embedding-3-small'
) -> np.ndarray:
    """
    Embeds a query string using the OpenAI API.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        query (str): The query string to embed.
        model_name (str): Name of the OpenAI model to use.
            Default is 'text-embedding-3-small'.

    Returns:
        np.ndarray: The embedded query vector.
    """
    response = client.embeddings.create(
        input=[query],
        model=model_name
    )
    embedding = np.array([response.data[0].embedding], dtype=np.float32)

    return embedding


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

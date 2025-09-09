"""
Embeds text chunks using SentenceTransformer and saves the embeddings to
a .npy file or embedded_chunks.json with metadata.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer


def _load_chunks(
    path: str
) -> list[dict]:
    """
    Loads text chunks from a JSON file.
    
    Args:
        path (str): Path to the JSON file containing text chunks.
        
    Returns:
        list[dict]: List of dictionaries containing text chunks and 
            metadata.
    """
    with open(path, 'r', encoding='utf-8') as fin:
        return json.load(fin)
    

def _save_embeddings_npy(
    embeddings: np.ndarray,
    output_path: str
) -> None:
    """
    Saves embeddings to a .npy file.
    
    Args:
        embeddings (np.ndarray): Array of embeddings.
        output_path (str): Path to save the .npy file.
    """
    np.save(output_path, embeddings)
    print(f'Saved NumPy embeddings to {output_path}')


def _save_embedded_chunks(
    chunks: list[dict],
    embeddings: np.ndarray,
    output_path: str
) -> None:
    """
    Saves text chunks along with their embeddings to a JSON file.
    
    Args:
        chunks (list[dict]): List of dictionaries containing text 
            chunks and metadata.
        embeddings (np.ndarray): Array of embeddings.
        output_path (str): Path to save the JSON file.
    """
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding.tolist()
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(chunks, fout, ensure_ascii=False, indent=4)
    
    print(f'Saved embedded chunks to {output_path}')


def embed_chunks(
    input_path: str,
    out_npy: str = None,
    out_json: str = None,
    model_name: str = 'all-MiniLM-L6-v2'
) -> None:
    """
    Embeds text chunks from a JSON file using SentenceTransformer and
    saves the embeddings to a .npy file or embedded_chunks.json with
    metadata.
    
    Args:
        input_path (str): Path to the JSON file containing text chunks.
        out_npy (str): Path to save the .npy file. Default is None.
        out_json (str): Path to save the JSON file with embeddings.
            Default is None.
        model_name (str): Name of the SentenceTransformer model to use.
            Default is 'all-MiniLM-L6-v2'.
    """
    print('-' * 72)

    print(f'Loading chunks from {input_path}...')
    chunks = _load_chunks(input_path)
    texts = [chunk['content'] for chunk in chunks]
    print(f'Loaded {len(chunks)} chunks.')

    print('-' * 72)

    print(f'Loading model {model_name}...')
    model = SentenceTransformer(model_name)
    print('Model loaded.')

    print('-' * 72)

    print('Embedding chunks...')
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    if out_npy:
        _save_embeddings_npy(embeddings, out_npy)
    
    if out_json:
        _save_embedded_chunks(chunks, embeddings, out_json)

    print('-' * 72)

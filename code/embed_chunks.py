"""
Embeds text chunks using OpenAI and saves the embeddings to
a .npy file or embedded_chunks.json with metadata.
"""

import json

import numpy as np
from openai import OpenAI


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


def _embed_with_openai(
    client: OpenAI,
    texts: list[str],
    model_name: str = 'text-embedding-3-small'
) -> np.ndarray:
    """
    Embeds a list of texts using the OpenAI API.
    
    Args:
        client (OpenAI): An instance of the OpenAI client.
        texts (list[str]): List of texts to embed.
        model_name (str): Name of the OpenAI embedding model to use.
            Default is 'text-embedding-3-small'.
        
    Returns:
        np.ndarray: Array of embeddings.
    """
    response = client.embeddings.create(
        input=texts,
        model=model_name
    )
    embeddings = [data.embedding for data in response.data]

    return np.array(embeddings, dtype=np.float32)


def embed_chunks(
    client: OpenAI,
    input_path: str,
    out_npy: str = None,
    out_json: str = None,
    model_name: str = 'text-embedding-3-small'
) -> None:
    """
    Embeds text chunks from a JSON file using OpenAI and
    saves the embeddings to a .npy file or embedded_chunks.json with
    metadata.
    
    Args:
        client (OpenAI): An instance of the OpenAI client.
        input_path (str): Path to the JSON file containing text chunks.
        out_npy (str): Path to save the .npy file. Default is None.
        out_json (str): Path to save the JSON file with embeddings.
            Default is None.
        model_name (str): Name of the OpenAI model to use.
            Default is 'text-embedding-3-small'.
    """
    print('-' * 72)

    print(f'Loading chunks from {input_path}...')
    chunks = _load_chunks(input_path)
    texts = [chunk['content'] for chunk in chunks]
    print(f'Loaded {len(chunks)} chunks.')

    print('-' * 72)

    print('Embedding chunks...')
    embeddings = _embed_with_openai(client, texts, model_name)

    if out_npy:
        _save_embeddings_npy(embeddings, out_npy)
    
    if out_json:
        _save_embedded_chunks(chunks, embeddings, out_json)

    print('-' * 72)

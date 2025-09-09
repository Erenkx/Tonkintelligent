"""
Builds a FAISS index from .npy embeddings and saves it to disk.
"""

import faiss
import numpy as np


def build_faiss_index(
    embeddings_path: str, 
    output_path: str
) -> None:
    """
    Builds a FAISS index from .npy embeddings and saves it to disk.

    Args:
        embeddings_path (str): Path to the .npy file containing 
            embeddings.
        output_path (str): Path to save the FAISS index file.
    """
    print(f'Loading embeddings from {embeddings_path}...')
    embeddings = np.load(embeddings_path).astype('float32')
    print(f'Loaded {embeddings.shape[0]} embeddings of dimension '
          f'{embeddings.shape[1]}.')
    
    print('-' * 72)

    print('Normalizing vectors for cosine similarity...')
    faiss.normalize_L2(embeddings)
    print('Normalization completed.')

    print('-' * 72) 

    print('Building FAISS index...')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f'FAISS index built with {index.ntotal} vectors.')

    print('-' * 72)

    print(f'Saving FAISS index to {output_path}...')
    faiss.write_index(index, output_path)
    print('FAISS index saved successfully.')

    print('-' * 72)

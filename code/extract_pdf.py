"""
Extracts text from PDF files in a specified folder, splits the text 
into chunks, and saves the chunks along with metadata into JSON files.
"""

import os
import json

import fitz # PyMuPDF
from tqdm import tqdm


def _extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:
    """
    Extracts text from a PDF and splits it into chunks with metadata.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Maximum number of words in each chunk. 
            Default is 500.
        chunk_overlap (int): Number of overlapping words between chunks.
            Default is 50.

    Returns:
        list[dict]: List of dictionaries containing text chunks and 
            metadata.
    """
    chunks = []
    file_name = os.path.basename(pdf_path)
    file_path = os.path.abspath(pdf_path)

    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue

        words = text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append({
                'content': chunk,
                'metadata': {
                    'source': file_name,
                    'path': file_path,
                    'page_number': page_num
                }
            })

    return chunks


def extract_folder(
    folder_path: str,
    output_dir: str
) -> None:
    """
    Processes all PDF files in a folder, extracts text chunks, and saves
    them as JSON files.
    
    Args:
        folder_path (str): Path to the folder containing PDF files.
        output_dir (str): Directory path to save the JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')
    ]
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(folder_path, pdf_file)
        chunks = _extract_pdf_chunks(pdf_path)

        output_path = os.path.join(
            output_dir, pdf_file.replace('.pdf', '.json')
        )
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(chunks, fout, ensure_ascii=False, indent=4)


def merge_chunks(
    chunk_dir: str,
    output_path: str
) -> None:
    """
    Merges all JSON chunk files in a directory into a single JSON
    file.

    Args:
        chunk_dir (str): Directory containing JSON chunk files.
        output_path (str): Path to save the merged JSON file.
    """
    all_chunks = []

    json_files = [
        f for f in os.listdir(chunk_dir) if f.lower().endswith('.json')
    ]
    for json_file in tqdm(json_files):
        json_path = os.path.join(chunk_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as fin:
            chunks = json.load(fin)
            all_chunks.extend(chunks)

    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(all_chunks, fout, ensure_ascii=False, indent=4)

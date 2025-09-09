import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import json

from extract_pdf import extract_folder, merge_chunks
from embed_chunks import embed_chunks
from faiss_index import build_faiss_index
from faiss_search import load_index, embed_query, search_faiss_index
from rag import format_prompt, call_local_model


MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


def main():
    project_name = 'project_1'
    project_path = os.path.join(
        os.getcwd(), 'data', 'projects', project_name
    )

    project_pdfs = os.path.join(project_path, 'pdfs')
    project_chunks = os.path.join(project_path, 'chunks')
    merged_chunks = os.path.join(project_path, 'merged_chunks.json')
    out_npy = os.path.join(project_path, 'embeddings.npy')
    out_json = os.path.join(project_path, 'embedded_chunks.json')
    faiss_index = os.path.join(project_path, 'faiss_index.index')

    num_pdfs = len([
        f for f in os.listdir(project_pdfs) if f.endswith('.pdf')
    ])
    num_chunks = len([
        f for f in os.listdir(project_chunks) if f.endswith('.json')
    ])

    if num_pdfs != num_chunks:
        extract_folder(project_pdfs, project_chunks)
        merge_chunks(project_chunks, merged_chunks)
        embed_chunks(
            input_path=merged_chunks,
            out_npy=out_npy,
            out_json=out_json,
            model_name='all-MiniLM-L6-v2'
        )
        build_faiss_index(
            embeddings_path=out_npy,
            output_path=faiss_index
        )

    index = load_index(faiss_index)
    query = 'What does Yilin Chen do well?'
    query_embedding = embed_query(query)
    top_indices = search_faiss_index(
        index=index,
        query_embedding=query_embedding,
        top_k=3
    )
    chunks = json.load(open(out_json, 'r'))
    prompt = format_prompt(
        query=query,
        chunks=chunks,
        top_indices=top_indices
    )
    response = call_local_model(
        prompt=prompt,
        model_name=MODEL_NAME
    )
    print(query)
    print(response)


if __name__ == "__main__":
    main()

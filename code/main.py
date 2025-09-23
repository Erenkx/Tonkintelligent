import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import json

from extract_pdf import extract_folder, merge_chunks
from embed_chunks import embed_chunks
from faiss_index import build_faiss_index
from faiss_search import load_index, embed_query, search_faiss_index
from rag import format_prompt, load_openai_api_key, call_openai_model


MODEL_NAME = 'gpt-3.5-turbo'


def main():
    project_name = '240365 - Mitcham Enviro Monitoring'
    project_path = os.path.join(
        os.getcwd(), 'data', 'projects', project_name
    )

    project_pdfs = os.path.join(project_path, 'pdfs')
    project_chunks = os.path.join(project_path, 'chunks')
    merged_chunks = os.path.join(project_path, 'merged_chunks.json')
    out_npy = os.path.join(project_path, 'embeddings.npy')
    out_json = os.path.join(project_path, 'embedded_chunks.json')
    faiss_index = os.path.join(project_path, 'faiss_index.index')
    openai_key_path = os.path.join(
        os.path.dirname(__file__), 'OPENAI_API_KEY.txt'
    )

    # extract_folder(project_pdfs, project_chunks)
    # merge_chunks(project_chunks, merged_chunks)
    # embed_chunks(
    #     input_path=merged_chunks,
    #     out_npy=out_npy,
    #     out_json=out_json,
    #     model_name='all-MiniLM-L6-v2'
    # )
    # build_faiss_index(
    #     embeddings_path=out_npy,
    #     output_path=faiss_index
    # )

    client = load_openai_api_key(openai_key_path)

    index = load_index(faiss_index)
    query = 'What are the scopes of this work?'
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
    response = call_openai_model(
        client=client,
        prompt=prompt,
        model_name=MODEL_NAME
    )
    print(query)
    print(response)


if __name__ == "__main__":
    main()

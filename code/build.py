import os

from rag import load_openai_api_key
from faiss_index import build_faiss_index
from extract_pdf import extract_folder, merge_chunks
from embed_chunks import embed_chunks, merge_embedded_chunks


def main():
    project_folder = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'projects'
    )
    projects = os.listdir(project_folder)
    projects.insert(-1, 'global')

    client = load_openai_api_key(
        os.path.join(
            os.path.dirname(__file__), 'OPENAI_API_KEY.txt'
        )
    )

    for project in projects:
        project_path = os.path.join(project_folder, project)

        merged_chunks = os.path.join(project_path, 'merged_chunks.json')
        out_npy = os.path.join(project_path, 'embeddings.npy')
        out_json = os.path.join(project_path, 'embedded_chunks.json')
        faiss_index = os.path.join(project_path, 'faiss_index.index')

        if project != 'global':
            project_pdfs = os.path.join(project_path, '6_Issued')
            project_chunks = os.path.join(project_path, 'chunks')

            extract_folder(project_pdfs, project_chunks)
            merge_chunks(project_chunks, merged_chunks)
            embed_chunks(
                client=client,
                input_path=merged_chunks,
                out_npy=out_npy,
                out_json=out_json,
                model_name='text-embedding-3-small'
            )
        else:
            merge_embedded_chunks(
                project_folder=project_folder,
                global_folder=project_path
            )

        build_faiss_index(
            embeddings_path=out_npy,
            output_path=faiss_index
        )


if __name__ == "__main__":
    main()

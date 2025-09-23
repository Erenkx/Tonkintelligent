import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json

import streamlit as st

from faiss_search import load_index, embed_query, search_faiss_index
from rag import format_prompt, load_openai_api_key, call_openai_model


MODEL_ENUM = {
    'OpenAI GPT-3.5': 'gpt-3.5-turbo',
    'OpenAI GPT-4': 'gpt-4'
}


def main():
    st.title('Tonkintelligent')
    st.write("Search Tonkin's legacy project data with AI-powered retrieval.")

    model = st.radio(
        'Select Model:',
        ('OpenAI GPT-3.5', 'OpenAI GPT-4'),
        horizontal=True
    )

    project_folder = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'projects'
    )
    projects = os.listdir(project_folder)
    project_name = st.selectbox(
        'Select Project:',
        projects
    )

    query = st.text_input('Enter your query:')

    project_path = os.path.join(project_folder, project_name)
    out_json = os.path.join(project_path, 'embedded_chunks.json')
    faiss_index = os.path.join(project_path, 'faiss_index.index')
    openai_key_path = os.path.join(
        os.path.dirname(__file__), 'OPENAI_API_KEY.txt'
    )

    if st.button('Search'):
        if not query.strip():
            st.warning('Please enter a query before searching.')
        else:
            client = load_openai_api_key(openai_key_path)
            index = load_index(faiss_index)

            query_embedding = embed_query(client, query)
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
                model_name=MODEL_ENUM[model]
            )

            st.subheader('Response:')
            st.success(response)

            st.subheader('Related Chunks:')
            for idx in top_indices:
                st.markdown(
                    f'**Source**: {chunks[idx]["metadata"]["source"]} '
                    f'(Page {chunks[idx]["metadata"]["page_number"]})'
                )


if __name__ == "__main__":
    main()

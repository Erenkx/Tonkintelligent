"""
Agent to decide and execute actions based on observations.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json

import transformers

from rag import call_local_model, format_prompt
from faiss_search import load_index, embed_query, search_faiss_index

MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
AGENT_PROMPT = (
"""You are an intelligent assisteant that analyzes user queries to decide what
action to take.

You must always respond in only a valid JSON format as described below:

{{
    "action": "summarize" | "find_files" | "both",
    "project": "<project_name> or null"
}}

Instructions:
- If the user is asking for a summary of a project, return "summarize" as the
    action, and the project name if specified, otherwise null.
- If the user wants to find files in a project, return "find_files" as the
    action, and the project name if specified, otherwise null.
- If the user wants both a summary and to find files, return "both" as the
    action, and the project name if specified, otherwise null.

User Query: {query}

JSON Response:
"""
)


def load_file_index(
    project_root: str
) -> dict:
    """
    Loads the file index from the given project root directory.
    
    Args:
        project_root (str): The root directory containing project 
            folders.
    
    Returns:
        dict: A dictionary mapping project names to lists of file 
            paths.
    """
    index_path = os.path.join(project_root, 'file_index.json')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f'File index not found at {index_path}')

    with open(index_path, 'r') as fin:
        file_index = json.load(fin)
    
    return file_index


def decide_action(
    prompt: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer
) -> tuple[str, str]:
    """
    Uses the language model to decide the next action based on the 
    query.

    Args:
        prompt (str): The input prompt for the model.
        model (transformers.PreTrainedModel): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer 
            for the model.

    Returns:
        tuple[str, str]: The decided action and project name.
    """
    inputs = tokenizer(
        prompt, return_tensors='pt', truncation=True
    )
    outputs = model.generate(**inputs, max_new_tokens=25)
    decision = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decision = json.loads(decision.split(prompt)[-1].strip())

    action = decision.get('action')
    project = decision.get('project')

    return action, project


def run_rag(
    query: str,
    project: str,
    project_root: str
) -> str:
    """
    Runs the RAG pipeline to get a response for the query.

    Args:
        query (str): The user query.
        project (str): The project name.
        project_root (str): The root directory containing project 
            folders.

    Returns:
        str: The response from the RAG pipeline.
    """
    embedded_chunks_path = os.path.join(
        project_root, project, 'embedded_chunks.json'
    )
    faiss_index = os.path.join(
        project_root, project, 'faiss_index.index'
    )

    chunks = json.load(open(embedded_chunks_path, 'r'))
    index = load_index(faiss_index)
    query_embedding = embed_query(query)
    top_indices = search_faiss_index(
        index=index,
        query_embedding=query_embedding,
        top_k=3
    )
    prompt = format_prompt(
        query=query,
        chunks=chunks,
        top_indices=top_indices
    )

    return call_local_model(prompt, MODEL_NAME)


def execute_action(
    action: str,
    project: str | None,
    project_root: str,
    query: str,
    file_index: dict
):
    """
    Executes the decided action.

    Args:
        action (str): The action to execute ("summarize", 
            "find_files", or "both").
        project (str | None): The project name, or None if not 
            specified.
        project_root (str): The root directory containing project
            folders.
        query (str): The user query.
        file_index (dict): The file index mapping project names 
            to file paths.
    """
    response = []
    if action in {'summarize', 'both'}:
        if project and project in file_index:
            response.append(run_rag(query, project, project_root))
        else:
            print('Please specify a valid project name.')

    if action in {'find_files', 'both'}:
        if project and project in file_index:
            files = file_index[project]
            res = f'Files in project {project}:\n'
            for f in files:
                res += f'- {f}\n'
            response.append(res.strip())
        else:
            print('Please specify a valid project name.')

    return '\n\n'.join(response)


if __name__ == "__main__":
    project_root = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'projects'
    )
    file_index = load_file_index(project_root)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    print('Agent is ready. Enter your query below and type "exit" to quit.')
    while True:
        query = input('> ')
        if query.lower() == 'exit':
            break

        prompt = AGENT_PROMPT.format(query=query)
        action, project = decide_action(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer
        )
        response = execute_action(
            action=action,
            project=project,
            project_root=project_root,
            query=query,
            file_index=file_index
        )
        print(response)
        print()

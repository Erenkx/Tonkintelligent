"""
Retrieval-Augmented Generation (RAG) implementation.
"""

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a helpful assistant that provides accurate and concise answers "
    "based on the provided context. If the context does not contain enough "
    "information to answer the question, respond with 'I don't know.'"
)

USER_PROMPT = (
"""Context: 
{context}

User Question: {query}

Answer: """
)


def load_openai_api_key(key_path: str) -> OpenAI:
    """
    Loads the OpenAI API key from a file and sets it in the openai module.

    Args:
        key_path (str): The path to the file containing the API key.

    Returns:
        OpenAI: An instance of the OpenAI client initialized with the API key.
    """
    with open(key_path, 'r') as fin:
        api_key = fin.read().strip()
    
    return OpenAI(api_key=api_key)


def format_prompt(
    query: str,
    chunks: list[dict],
    top_indices: list[int]
) -> str:
    """
    Formats the prompt for the language model.

    Args:
        query (str): The user query.
        chunks (list[dict]): The list of document chunks with metadata.
        top_indices (list[int]): The indices of the top chunks.

    Returns:
        str: The formatted prompt.
    """
    context = ''
    for idx in top_indices:
        chunk = chunks[idx]
        metadata = chunk['metadata']
        content = chunk['content'].strip().replace('\n', ' ')
        context += f'[Source: {metadata["source"]} - Page: '
        context += f'{metadata["page_number"]}]\n{content}\n\n'
    context = context.strip()

    return USER_PROMPT.format(context=context, query=query)


def call_openai_model(
    client: OpenAI,
    prompt: str,
    model_name: str = 'gpt-3.5-turbo',
    max_tokens: int = 512
) -> str:
    """
    Calls the OpenAI API to generate a response.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        prompt (str): The input prompt for the model.
        model_name (str): The name of the OpenAI model to use.
        max_tokens (int): Maximum number of tokens to generate. 
            Default is 512.

    Returns:
        str: The generated response from the model.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

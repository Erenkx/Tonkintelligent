"""
Retrieval-Augmented Generation (RAG) implementation.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT = (
"""You are a helpful assistant. Use the following context to reason and
answer the question.

Context: 
{context}

User Question: {query}

Answer: """
)


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

    return PROMPT.format(context=context, query=query)


def call_local_model(
    prompt: str,
    model_name: str,
    max_tokens: int = 512
) -> str:
    """
    Calls a local language model to generate a response.

    Args:
        prompt (str): The input prompt for the model.
        model_name (str): The name or path of the local model.
        max_tokens (int): Maximum number of tokens to generate. 
            Default is 512.

    Returns:
        str: The generated response from the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response.split(prompt)[-1].strip()

from FM2DS.lvlm.gpt.get_response import get_gpt_response
from FM2DS.lvlm.claude.get_response import get_claude_response
from FM2DS.lvlm.llama.get_response import get_llama_response

def generate_query(model_name, question, answer, documents):
    """
    Generates a query using the specified model.

    Args:
        model_name (str): The model to use ("gpt", "claude", or "llama").
        question (str): The question being addressed.
        answer (str): The answer to the question.
        documents (list): A list of documents. Each document is a dict with:
            - "title" (str): Title of the document.
            - "content" (list): List of dicts, where each dict contains:
                - "type" (str): Type of content ("text", "image", etc.).
                - "value" (str): The content value.

    Returns:
        str: The generated query.
    """
    # Format the documents for the prompt
    def format_document(doc):
        formatted = f"Title: {doc['title']}\n"
        for item in doc['content']:
            if item['type'] == "text":
                formatted += f"Text: {item['value']}\n"
            elif item['type'] == "image":
                if item['value'].startswith("http"):
                    formatted += f"Image (URL): {item['value']}\n"
                elif item['value'].startswith("data:image"):
                    formatted += "Image (Base64 Encoded): [Base64 data included]\n"
                else:
                    formatted += f"Image (File Path): {item['value']}\n"
        return formatted.strip()

    formatted_documents = "\n\n".join([format_document(doc) for doc in documents])

    # Prompt for query generation
    system_message = """You are an advanced AI assistant capable of explaining complex reasoning processes.
Your task is to analyze multiple documents and explain the step-by-step process used to extract
and verify answers to questions using information across documents and modalities."""

    prompt = f"""You are provided with multiple
documents, a question, and the
answer. Your task is to explain
the step-by-step process you would
use to extract and verify the
answer using information from the
documents and various modalities.
Clearly identify each document
title and relevant sections, and
describe how you locate, interpret,
and integrate information across
both documents to derive the
correct answer.

Question:
{question}

Answer:
{answer}

Here are the documents:
{formatted_documents}
"""

    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # Call the selected model
    if model_name == "gpt":
        return get_gpt_response(messages)
    elif model_name == "claude":
        return get_claude_response(messages)
    elif model_name == "llama":
        return get_llama_response(messages)
    else:
        raise ValueError("Invalid model_name. Choose from 'gpt', 'claude', or 'llama'.")
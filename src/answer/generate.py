from FM2DS.lvlm.gpt.get_response import get_gpt_response
from FM2DS.lvlm.claude.get_response import get_claude_response
from FM2DS.lvlm.llama.get_response import get_llama_response

def generate_answer(model_name, question, documents, few_shot_examples):
    """
    Generates an answer using the specified model.

    Args:
        model_name (str): The model to use ("gpt", "claude", or "llama").
        question (str): The question to be answered.
        documents (list): A list of documents. Each document is a dict with:
            - "title" (str): Title of the document.
            - "content" (list): List of dicts, where each dict contains:
                - "type" (str): Type of content ("text", "image", etc.).
                - "value" (str): The content value.
        few_shot_examples (list): A list of few-shot examples. Each example is a dict with:
            - "question" (str): The question.
            - "answer" (str): The answer.
            - "documents" (list): Documents structured like the `documents` parameter.

    Returns:
        str: The generated answer.
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
    formatted_examples = "\n\n".join([
        f"Question: {example['question']}\nAnswer: {example['answer']}\nDocuments:\n" +
        "\n\n".join([format_document(doc) for doc in example['documents']])
        for example in few_shot_examples
    ])

    # Prompt for answer generation
    system_message = """You are an advanced AI assistant capable of analyzing multimodal content.
Your task is to answer questions by synthesizing information from multiple documents and modalities,
such as text and images. You must combine insights across all sources to deliver the most accurate
and comprehensive response possible."""

    prompt = f"""You are provided with multiple
documents, including both textual
content and images, along with
a question. Your task is to
carefully review each document,
analyze the images, and derive an
answer based on the information
contained across all sources. Aim
to combine insights from both
documents and across modalities
to deliver the most accurate and
comprehensive response possible.

Question:
{question}

Here are the documents:
{formatted_documents}

Here are examples:
{formatted_examples}
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

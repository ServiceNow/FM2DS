from FM2DS.lvlm.gpt.get_response import get_gpt_response
from FM2DS.lvlm.claude.get_response import get_claude_response
from FM2DS.lvlm.llama.get_response import get_llama_response

def generate_question(model_name, documents, few_shot_examples, redundant_examples=None):
    """
    Generates a multi-hop question using the specified model.

    Args:
        model_name (str): The model to use ("gpt", "claude", or "llama").
        documents (list): A list of documents. Each document is a dict with:
            - "title" (str): Title of the document.
            - "content" (list): List of dicts, where each dict contains:
                - "type" (str): Type of content ("text", "image", etc.).
                - "value" (str): The content value.
        few_shot_examples (list): A list of few-shot examples. Each example is a dict with:
            - "question" (str): The question.
            - "answer" (str): The answer.
            - "documents" (list): Documents structured like the `documents` parameter.
        redundant_examples (list): A list of previously generated questions to avoid duplication.

    Returns:
        str: The generated multi-hop question.
    """
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

    system_message = """You are an advanced AI assistant capable of analyzing multimodal content.
Your task is to generate complex, multi-hop questions that require synthesizing information
from multiple documents and modalities (e.g., text and images). Focus on ensuring the questions
are unanswerable if any single document or modality is missing."""

    prompt = f"""Generate a multi-hop question
based on the provided information.
A multi-hop question requires
the model to utilize information
from all available documents
in combination to reach the
correct answer. Specifically,
the question should be designed to
be unanswerable if any one of the
documents is missing. Furthermore,
focus on creating questions that
compel the model to extract and
synthesize relevant information
across multiple modalities|such as
images and text. This means that
answering the question correctly
will demand integrating insights
from each source and modality,
making it impossible to arrive at
an accurate answer using any single
document or modality alone.

Here are the documents:
{formatted_documents}

Here are examples:
{formatted_examples}
"""

    if redundant_examples is not None:
        prompt += "\n\nYou previously generated questions that already existed in my samples. Do not generate these again:\n"
        for i, redundant_example in enumerate(redundant_examples):
            prompt += f"{i+1}- {redundant_example}\n"

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

if __name__ == "__main__":
    # Example usage
    documents = [
        {
            "title": "Document 1",
            "content": [
                {"type": "text", "value": "The Eiffel Tower is in Paris."},
                {"type": "image", "value": "http://example.com/eiffel.jpg"}
            ]
        },
        {
            "title": "Document 2",
            "content": [
                {"type": "text", "value": "The Taj Mahal is in India."},
                {"type": "image", "value": "/images/taj_mahal.png"}
            ]
        }
    ]

    few_shot_examples = [
        {
            "question": "What are the main landmarks mentioned in the documents?",
            "answer": "The Eiffel Tower and the Taj Mahal.",
            "documents": [
                {
                    "title": "Example Document",
                    "content": [
                        {"type": "text", "value": "Landmarks include the Eiffel Tower and the Taj Mahal."},
                        {"type": "image", "value": "http://example.com/landmarks.jpg"}
                    ]
                }
            ]
        }
    ]

    redundant_examples = ["What is the location of the Eiffel Tower?"]

    model_name = "gpt"  # Change to "claude" or "llama" as needed
    generated_question = generate_question(model_name, documents, few_shot_examples, redundant_examples)
    print("Generated Question:")
    print(generated_question)

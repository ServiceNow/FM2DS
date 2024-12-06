from FM2DS.lvlm.llama.get_response import get_llama_response
from FM2DS.lvlm.gpt.get_response import get_gpt_response
from FM2DS.src.answer.generate import generate_answer
from bs4 import BeautifulSoup

def break_question_into_components(question):
    """
    Breaks a question into simpler components using LLama.

    Args:
        question (str): The question to split.

    Returns:
        list: List of simpler components extracted from the question.
    """
    prompt = f"Break down the following question into simpler components:\n\nQuestion: {question}"
    response = get_llama_response([{"role": "user", "content": prompt}], "http://localhost:8085")
    components = response.split("\n")
    return [component.strip() for component in components if component.strip()]

def contains_table(text):
    """
    Checks if the text contains an HTML table.

    Args:
        text (str): The text content to analyze.

    Returns:
        bool: True if the text contains a table, False otherwise.
    """
    soup = BeautifulSoup(text, "html.parser")
    return bool(soup.find("table"))

def is_question_multimodal(question, documents, model_name):
    """
    Checks if a question is multimodal by testing if it can be answered with a single modality.

    Args:
        question (str): The question to validate.
        documents (list): List of documents with multiple modalities.
        model_name (str): The model name used to generate answers.

    Returns:
        bool: True if the question is multimodal, False otherwise.
    """
    modalities = ["text", "image", "table"]

    for modality in modalities:
        filtered_documents = []
        for doc in documents:
            filtered_content = []
            for item in doc["content"]:
                if modality == "text" and item["type"] == "text":
                    if not contains_table(item["value"]):
                        filtered_content.append(item)
                elif modality == "table" and item["type"] == "text":
                    if contains_table(item["value"]):
                        filtered_content.append(item)
                elif modality == item["type"]:
                    filtered_content.append(item)
            if filtered_content:
                filtered_documents.append({"title": doc["title"], "content": filtered_content})

        # Filter out empty documents
        if not filtered_documents:
            continue

        # Generate an answer using only the filtered documents
        generated_answer = generate_answer(
            model_name=model_name,
            question=question,
            documents=filtered_documents,
            few_shot_examples=[]
        )

        if generated_answer.strip():  # If the model generates an answer
            return False

    return True

def rephrase_multihop_question(question):
    """
    Rephrases a multihop question into a concise format using GPT-4o.

    Args:
        question (str): The multihop question to rephrase.

    Returns:
        str: The rephrased question.
    """
    prompt = f"Rephrase the following question into a concise multihop format without conjunctions:\n\nQuestion: {question}"
    response = get_gpt_response([{"role": "user", "content": prompt}])
    return response.strip()

def validate_question(question, documents, model_name):
    """
    Validates a question to ensure it meets multihop and multimodal criteria.

    Args:
        question (str): The question to validate.
        documents (list): List of documents containing multimodal content.
        model_name (str): The model name used to generate answers.

    Returns:
        bool: True if the question passes validation, False otherwise.
    """
    # Step 1: Break question into components and check multihop reasoning
    components = break_question_into_components(question)

    # Check if all components are solvable with one document
    all_solvable = all(any(
        component in doc["content"][0]["value"] for doc in documents
        if component) for component in components)

    if all_solvable:
        print(f"Discarding question: {question} (all components solvable with one document)")
        return False

    # Step 2: Rephrase the question for concise multihop reasoning
    question = rephrase_multihop_question(question)

    # Step 3: Check if the question is multimodal
    if not is_question_multimodal(question, documents, model_name):
        print(f"Discarding question: {question} (not multimodal)")
        return False

    return True

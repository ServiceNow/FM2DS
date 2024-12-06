import spacy
from FM2DS.lvlm.gpt.get_response import get_gpt_response

def extract_named_entities(text):
    """
    Extract named entities from a text using spaCy.

    Args:
        text (str): The text to process.

    Returns:
        dict: A dictionary of named entities by type (e.g., PERSON, ORG, LOC).
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def validate_entities(answer_entities, document_entities):
    """
    Validate named entities between the answer and the documents.

    Args:
        answer_entities (dict): Entities extracted from the answer.
        document_entities (dict): Entities extracted from the documents.

    Returns:
        bool: True if all entities in the answer are found in the documents, False otherwise.
    """
    for entity_type, entities in answer_entities.items():
        document_set = set(document_entities.get(entity_type, []))
        if not all(entity in document_set for entity in entities):
            return False
    return True

def extract_relations(text):
    """
    Extract relations between entities from a text using spaCy's dependency parsing.

    Args:
        text (str): The text to process.

    Returns:
        list: A list of relations between entities.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    relations = []
    for token in doc:
        if token.dep_ == "prep" and token.head.ent_type_ and token.ent_type_:
            relations.append((token.head.text, token.text, token.text))
    return relations

def validate_relations(answer_relations, document_relations):
    """
    Validate relationships between entities in the answer and the documents.

    Args:
        answer_relations (list): Relations extracted from the answer.
        document_relations (list): Relations extracted from the documents.

    Returns:
        bool: True if all relations in the answer are found in the documents, False otherwise.
    """
    document_set = set(document_relations)
    return all(relation in document_set for relation in answer_relations)

def generate_image_caption(image_url, question):
    """
    Generate a caption for an image using GPT-4o.

    Args:
        image_url (str): The URL of the image.
        question (str): The question context for the image.

    Returns:
        str: The generated caption.
    """
    prompt = [
        {"role": "system", "content": "You are an AI assistant capable of analyzing images and providing detailed captions."},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "user", "content": {"type": "image", "image_url": image_url}}
    ]

    response = get_gpt_response(prompt)
    return response

def validate_answer(answer, documents, question):
    """
    Validate the generated answer against the documents.

    Args:
        answer (str): The generated answer.
        documents (list): List of documents containing text and image content.
        question (str): The question being answered.

    Returns:
        bool: True if the answer passes validation, False otherwise.
    """

    document_text = " ".join([content['value'] for doc in documents for content in doc['content'] if content['type'] == "text"])
    answer_entities = extract_named_entities(answer)
    document_entities = extract_named_entities(document_text)

    if not validate_entities(answer_entities, document_entities):
        return False

    answer_relations = extract_relations(answer)
    document_relations = extract_relations(document_text)

    if not validate_relations(answer_relations, document_relations):
        return False

    images = [content['value'] for doc in documents for content in doc['content'] if content['type'] == "image"]

    if images:
        for image in images:
            captions = [generate_image_caption(image, question) for _ in range(5)]
            if len(set(captions)) > 1:
                return False

    return True